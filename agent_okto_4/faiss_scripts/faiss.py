from langchain_community.vectorstores import FAISS
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

_data_cache = {}

def carregar_faiss(collection_name):
    global _data_cache
    
    if collection_name in _data_cache:
        print(f"Usando dados carregados do cache para '{collection_name}'.")
        return _data_cache[collection_name]

    faiss_file = f'faiss_container/faiss_index_{collection_name}.pkl'
    
    if not os.path.exists(faiss_file):
        raise FileNotFoundError(f"O arquivo {faiss_file} não foi encontrado.")
    
    with open(faiss_file, "rb") as f:
        vectordb = pickle.load(f)
    
    _data_cache[collection_name] = vectordb

    return vectordb

def load_existing_faiss_index(collection_name):
    faiss_index_path = f"faiss_container/faiss_index_{collection_name}.pkl"
    if os.path.exists(faiss_index_path):
        print(f"Carregando índice FAISS existente para {collection_name}...")
        with open(faiss_index_path, "rb") as f:
            vectordb = pickle.load(f)
        return vectordb
    return None

def atualizar_faiss(novo_texto, faiss_index_path, embedding_model):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    embedding_novo_texto = embeddings.embed_documents([novo_texto])

    faiss_index = FAISS.load_local(faiss_index_path, embeddings)

    faiss_index.add_documents([novo_texto], embedding_novo_texto)

    faiss_index.save_local(faiss_index_path)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50, 
    separators=['\n\n', '\n', '.', '?', ' ', '']
)