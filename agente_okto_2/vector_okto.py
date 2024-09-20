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

# Utilizando um modelo de embeddings mais leve e rápido: sentence-transformers/all-MiniLM-L6-v2
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50, 
    separators=['\n\n', '\n', '.', ' ', '']
)

# Diretório para salvar o índice FAISS
faiss_index_path = "faiss_index"

txt_files = {
    "afiliados_TeC": "afiliados_TeC.txt",
    "apostas_de_quota_fixa": "apostas_de_quota_fixa.txt",
    "autorizacoes_aposta": "autorizacoes_aposta.txt",
    "loterias": "loterias.txt",
    "politica_de_bonus": "politica_de_bonus.txt",
    "politica_de_jogo_responsavel": "politica_de_jogo_responsavel.txt",
    "politica_de_privacidade": "politica_de_privacidade.txt",
    "politica_de_reclamacao": "politica_de_reclamacao.txt",
    "promocao_comercial": "promocao_comercial.txt",
    "promocoes_comerciais": "promocoes_comerciais.txt",
    "regra_de_apostas": "regra_de_apostas.txt",
    "seguranca": "seguranca.txt",
    "termos_e_condicao": "termos_e_condicao.txt",
}

# Processando cada arquivo TXT e criando as coleções separadas
for collection_name, txt_file in txt_files.items():
    print(f"Processando {collection_name}...")

    # Carregar os arquivos TXT e extrair o texto
    documents = load_txt_files([txt_file])

    # Dividir o conteúdo dos textos em partes menores
    split_docs = text_splitter.split_documents(documents)

    # Criar ou conectar ao índice FAISS para essa coleção específica
    vectordb = FAISS.from_documents(
        documents=split_docs,
        embedding=embedding_model
    )

    # Salvar o índice FAISS para uso posterior
    with open(f"{faiss_index_path}_{collection_name}.pkl", "wb") as f:
        pickle.dump(vectordb, f)

    print(f"Índice FAISS para {collection_name} salvo em {faiss_index_path}_{collection_name}.pkl")

