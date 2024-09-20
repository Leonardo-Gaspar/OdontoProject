from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from reader_txt import TxtReader
from langchain import hub
from langchain.agents import create_react_agent, Tool
import os
import json


class AgenteOpenAIFunctions:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4",
                              temperature=0.5,
                              api_key=os.getenv("OPENAI_API_KEY"))

        self.txt_reader = TxtReader()
        
        self.tools = [
            Tool(
                name=self.txt_reader.name,
                func=self.txt_reader._run,
                description=self.txt_reader.description,
                return_direct=False
            )
        ]

        prompt = hub.pull("hwchase17/react")
        self.agente = create_react_agent(self.llm, self.tools, prompt)

    def _run(self, input_text: str) -> str:
        resposta = self.agente.invoke({"input": input_text})
        return json.dumps(resposta)


    @staticmethod
    def criar_qa_chain(documentos):
        # Configuração do modelo e embeddings
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.5,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        quebrador = CharacterTextSplitter(chunk_size=500,
                                          chunk_overlap=100)

        textos = quebrador.split_documents(documentos)

        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(textos, embeddings)


        qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever(search_kwargs={"k": 5}))

        return qa_chain

    @staticmethod
    def responder_pergunta(qa_chain, pergunta):
        resultado = qa_chain.invoke({'query': pergunta})
        return resultado
