import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50, 
    separators=['\n\n', '\n', '.', '?', ' ', '']
)

def process_user_document(file_path):
    user_faiss_path = f"faiss_container/faiss_index_user_{os.path.basename(file_path)}.pkl"
    
    if os.path.exists(user_faiss_path):
        print(f"O arquivo FAISS já existe para {file_path}, carregando-o.")
        with open(user_faiss_path, "rb") as f:
            vectordb = pickle.load(f)
        return vectordb
    else:
        print(f"Processando o documento {file_path} e gerando FAISS...")

        with open(file_path, "r", encoding="utf-8", errors='ignore') as file:
            content = file.read().lower()

        document = Document(page_content=content, metadata={"source": file_path})

        split_docs = text_splitter.split_documents([document])

        vectordb = FAISS.from_documents(
            documents=split_docs,
            embedding=embedding_model
        )

        with open(user_faiss_path, "wb") as f:
            pickle.dump(vectordb, f)

        print(f"Índice FAISS para o documento {file_path} salvo em {user_faiss_path}")
        return vectordb