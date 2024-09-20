# Script responsável por extrair os dados contidos nos arquivos de texto referente aos contratos e documentos
# gerar a base de dados vetorial Chroma para pesquisa e uso da ferramenta do Agente.

from langchain_community.embeddings.sentence_transformer import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import pickle

def load_txt_files(file_paths):
    documents = []
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read().lower()
            documents.append(Document(page_content=content, metadata={"source": file_path}))
    return documents

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50, 
    separators=['\n\n', '\n', '.', ' ', '']
)

faiss_index_path = "faiss_index"

txt_files = {
    "Lei": ["okto_documents/apostas_de_quota_fixa.txt",
            "okto_documents/autorizacoes_aposta.txt",
            "okto_documents/loterias.txt",
            "okto_documents/promocao_comercial.txt",
            "okto_documents/promocoes_comerciais.txt"
            ],
    "Documentacao": [ "okto_documents/afiliados_TeC.txt",
                      "okto_documents/politica_de_bonus.txt",
                      "okto_documents/politica_de_jogo_responsavel.txt",
                      "okto_documents/politica_de_privacidade.txt",
                      "okto_documents/politica_de_reclamacao.txt",
                      "okto_documents/regra_de_apostas.txt",
                      "okto_documents/seguranca.txt",
                      "okto_documents/termos_e_condicao.txt"
                      ]
}

for collection_name, txt_file in txt_files.items():
    print(f"Processando {collection_name}...")

    documents = load_txt_files([txt_file])

    split_docs = text_splitter.split_documents(documents)

    vectordb = FAISS.from_documents(
        documents=split_docs,
        embedding=embedding_model
    )

    with open(f"{faiss_index_path}_{collection_name}.pkl", "wb") as f:
        pickle.dump(vectordb, f)

    print(f"Índice FAISS para {collection_name} salvo em {faiss_index_path}_{collection_name}.pkl")

