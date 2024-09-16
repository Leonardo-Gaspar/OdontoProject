from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
import os

def criar_agente(documentos):
    # Configuração do modelo e embeddings
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.5,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Quebra os documentos em pedaços menores
    quebrador = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    textos = quebrador.split_documents(documentos)

    # Criação do vetor de embeddings
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(textos, embeddings)

    # Criação da cadeia de perguntas e respostas
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

    return qa_chain

def responder_pergunta(qa_chain, pergunta):
    resultado = qa_chain.invoke({'query': pergunta})
    return resultado
