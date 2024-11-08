# Script responsável por extrair os dados contidos nos arquivos de texto referente aos contratos e documentos
# gerar a base de dados vetorial Chroma para pesquisa e uso da ferramenta do Agente.

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from faiss_scripts.faiss import load_existing_faiss_index
from documentos_scripts.leitura_documento import load_txt_files
 
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50, 
    separators=['\n\n', '\n', '.', '?', ' ', '']
)

faiss_index_path = "faiss_index"

txt_files = {
    "faq": ["okto_documents/apostas_de_quota_fixa.txt",
            "okto_documents/autorizacoes_aposta.txt",
            "okto_documents/loterias.txt",
            "okto_documents/promocao_comercial.txt",
            "okto_documents/promocoes_comerciais.txt"],
    
    "lei": ["okto_documents/leis.txt"],
    
    "portaria": ["okto_documents/portarias.txt"],
    
    "decreto": ["okto_documents/decretos.txt"]
}

for collection_name, txt_file_list in txt_files.items():

    existing_vectordb = load_existing_faiss_index(collection_name)

    if existing_vectordb:
        pass
    else:
        documents = load_txt_files(txt_file_list)
        split_docs = text_splitter.split_documents(documents)

        vectordb = FAISS.from_documents(
            documents=split_docs,
            embedding=embedding_model
        )
        with open(f"{faiss_index_path}_{collection_name}.pkl", "wb") as f:
            pickle.dump(vectordb, f)
