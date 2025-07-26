# app.py
# -*- coding: utf-8 -*-

import os
from dotenv import load_dotenv

# ✅ Carrega variáveis de ambiente do .env (local) ou do Render
load_dotenv()
GROQ_API = os.getenv("GROQ_API")

# -----------------------------
# 📂 Importações do LlamaIndex
# -----------------------------
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.llms.groq import Groq
from llama_index.core.memory import ChatSummaryMemoryBuffer

# -----------------------------
# 📂 Importações auxiliares
# -----------------------------
import chromadb
import gradio as gr

# ==============================================================
# 1. Carregar os documentos locais
# ==============================================================

documentos = SimpleDirectoryReader(input_dir='./documentos')
docs = documentos.load_data()

# Fazendo o chunking do arquivo
node_parser = SentenceSplitter(chunk_size=1200)
nodes = node_parser.get_nodes_from_documents(docs, show_progress=True)

# ==============================================================
# 2. Criar a base vetorial com ChromaDB
# ==============================================================

db = chromadb.PersistentClient(path="./chroma_db")
collection_name = 'documentos_serenatto'

try:
    chroma_collection = db.get_or_create_collection(name=collection_name)
except Exception as e:
    print(f"Não foi possível carregar/criar a coleção: {e}")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

embed_model = HuggingFaceEmbedding(model_name='intfloat/multilingual-e5-large')

# Cria ou carrega índice
if os.path.exists("./chroma_db"):
    index = load_index_from_storage(storage_context, embed_model=embed_model)
else:
    index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)

# ==============================================================
# 3. Configuração do modelo LLM Groq
# ==============================================================

llm = Groq(model='llama3-70b-8192', api_key=GROQ_API)

# ==============================================================
# 4. Criar Chat Engine com memória resumida
# ==============================================================

memory = ChatSummaryMemoryBuffer(llm=llm, token_limit=256)
chat_engine = index.as_chat_engine(
    chat_mode='context',
    llm=llm,
    memory=memory,
    system_prompt=(
        "Você é especialista em cafés especiais da Serenatto, uma loja online que vende grãos de cafés torrados. "
        "Sua função é tirar dúvidas de forma simpática e natural sobre os grãos disponíveis."
    )
)

# ==============================================================
# 5. Funções para Gradio
# ==============================================================

def converse_com_bot(message, chat_history):
    response = chat_engine.chat(message)

    if chat_history is None:
        chat_history = []

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": response.response})

    return "", chat_history

def resetar_chat():
    chat_engine.reset()
    return []

# ==============================================================
# 6. Interface Gradio
# ==============================================================

with gr.Blocks() as demo:
    gr.Markdown('# ☕ Chatbot Serenatto - Cafés Especiais')

    chatbot = gr.Chatbot(type='messages')
    msg = gr.Textbox(label='Digite a sua mensagem')
    limpar = gr.Button('Limpar')

    msg.submit(converse_com_bot, [msg, chatbot], [msg, chatbot])
    limpar.click(resetar_chat, None, chatbot, queue=False)

# ==============================================================
# 7. Executar a aplicação
# ==============================================================

if __name__ == "__main__":
    # Render precisa rodar em 0.0.0.0 e porta 8080
    demo.launch(server_name="0.0.0.0", server_port=8080)
