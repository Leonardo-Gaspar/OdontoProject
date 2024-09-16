from langchain_community.document_loaders import TextLoader 
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
from pydantic import BaseModel, Field

def carregar_documentos():
    loaders = [
        TextLoader("betano_documents/apostas_de_quota_fixa/apostas_de_quota_fixa.txt", encoding="utf-8"),
        TextLoader("betano_documents/autorizacao_de_aposta_de_quota_fixa/autorizacoes_aposta.txt", encoding="utf-8"),
        TextLoader("betano_documents/loterias/loterias.txt", encoding="utf-8"),
        TextLoader("betano_documents/promocao_comercial/promocao_comercial.txt", encoding="utf-8"),
        TextLoader("betano_documents/promocoes_comerciais/promocoes_comerciais.txt", encoding="utf-8"),
    ]

    documentos = []
    for loader in loaders:
        documentos.extend(loader.load())
    
    return documentos

class ExtratorDeBetano(BaseModel):
    betano_dados: str = Field("Extrator de todos os dados obtidos através de artigos e leis.")

class TxtReader(BaseTool):
    name = "TxtReader"
    description = """Esta ferramenta é utilizada para fazer a leitura completa
    de todos os arquivos TXTs inseridos na pasta 'betano_documents'"""  
    
    def _run(self, input: str) -> str:
        llm = ChatOpenAI(model="gpt-4", 
                         temperature= 0.5,
                         api_key=os.getenv("OPENAI_API_KEY"))
        
        parser = JsonOutputParser(pydantic_object=ExtratorDeBetano)  
        template = PromptTemplate(template="""Você deve analisar a entrada a seguir e extrair o nome informado em minúsculo.
                                        Entrada:
                                        -----------------
                                        {input}
                                        -----------------
                        Formato de saída:
                        {formato_saida}""",
                        input_variables=["input"],
                        partial_variables={"formato_saida": parser.get_format_instructions()})
        cadeia = template | llm | parser
        resposta = cadeia.invoke({"input": input})
        return resposta
        
