# Importa o ChatOpenAI da biblioteca langchain_openai, que faz a integração com o modelo GPT da OpenAI.
from langchain_openai import ChatOpenAI
# Habilita o modo de debug para fornecer mais informações sobre a execução do código.
from langchain.globals import set_debug
from langchain.tools import BaseTool

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Biblioteca para manipulação de variáveis de ambiente.
import os
# Carrega a biblioteca dotenv para carregar variáveis de ambiente de um arquivo .env.
from dotenv import load_dotenv
# Carrega as variáveis de ambiente do arquivo .env.
load_dotenv()
# Ativa o modo de debug para rastrear os processos da execução.
set_debug(True)


# class LeitorDePDF(BaseTool):
#     name = "LeitorDePDF"
#     description = "Lê e extrai o texto de um arquivo PDF."

#     def _run(self, caminho_pdf: str) -> str:
#         try:
#             reader = PdfReader(caminho_pdf)
#             texto = ""
#             for page in reader.pages:
#                 texto += page.extract_text()
#             return texto
#         except Exception as e:
#             return f"Erro ao ler o PDF: {e}"

llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.5,
        api_key=os.getenv("OPENAI_API_KEY")  # Obtém a chave da API do arquivo .env
)

loaders = [
        TextLoader("betano_documents/apostas_de_quota_fixa/apostas_de_quota_fixa.txt", encoding="utf-8"),
        TextLoader("betano_documents/autorizacao_de_aposta_de_quota_fixa/autorizacoes_aposta.txt", encoding="utf-8"),
        TextLoader("betano_documents/loterias/loterias.txt", encoding="utf-8"),
        TextLoader("betano_documents/promocao_comercial/promocao_comercial.txt", encoding="utf-8"),
        TextLoader("betano_documents/promocoes_comerciais/promocoes_comerciais.txt", encoding="utf-8"),
]

# Carregar documentos de cada loader
documents = []
for loader in loaders:
    documents.extend(loader.load())

quebrador = CharacterTextSplitter(separator="\n\n", chunk_size=2000, chunk_overlap=200)
texts = quebrador.split_documents(documents)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)

qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

pergunta = "Como devo proceder caso tenha um item comprado roubado?"
resultado = qa_chain.invoke({'query': pergunta})

print(resultado)