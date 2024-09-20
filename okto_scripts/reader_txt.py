from langchain_community.document_loaders import TextLoader 
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
from pydantic import BaseModel, Field

def carregar_documentos():
    loaders = [
        TextLoader("okto_documents/apostas_de_quota_fixa.txt", encoding="utf-8"),
        TextLoader("okto_documents/autorizacoes_aposta.txt", encoding="utf-8"),
        TextLoader("okto_documents/loterias.txt", encoding="utf-8"),
        TextLoader("okto_documents/promocao_comercial.txt", encoding="utf-8"),
        TextLoader("okto_documents/promocoes_comerciais.txt", encoding="utf-8"),
        TextLoader("okto_documents/afiliados_TeC.txt", encoding="utf-8"),
        TextLoader("okto_documents/politica_de_bonus.txt", encoding="utf-8"),
        TextLoader("okto_documents/politica_de_jogo_responsavel.txt", encoding="utf-8"),
        TextLoader("okto_documents/politica_de_privacidade.txt", encoding="utf-8"),
        TextLoader("okto_documents/politica_de_reclamacao.txt", encoding="utf-8"),
        TextLoader("okto_documents/regra_de_apostas.txt", encoding="utf-8"),
        TextLoader("okto_documents/seguranca.txt", encoding="utf-8"),
        TextLoader("okto_documents/termos_e_condicao.txt", encoding="utf-8"),
    ]

    documentos = []
    for loader in loaders:
        documentos.extend(loader.load())
    
    return documentos

class ExtratorDeBetano(BaseModel):
    betano_dados: str = Field("Extrator de todos os dados obtidos através de artigos, leis e documentação Betano.")

class TxtReader(BaseTool):
    name = "TxtReader"  
    description = """Esta ferramenta é utilizada para fazer a leitura completa
    de todos os arquivos TXTs inseridos para alimentar sua base de dados.
    Você deve fornecer um resposta mais próxima da realidade o possível, baseada nos documentos que foram inseridos."""  
    
    def _run(self, input: str) -> str:
        llm = ChatOpenAI(model="gpt-4", 
                        temperature=0.5,
                        api_key=os.getenv("OPENAI_API_KEY"))
        
        parser = JsonOutputParser(pydantic_object=ExtratorDeBetano)
        
        # Ajustando o prompt para garantir uma saída JSON
        template = PromptTemplate(
            template="""
            Você deve analisar a entrada a seguir e extrair as informações relevantes com base nos documentos disponíveis, os quais foram retirados da classe TxtReader.
            Nesses Txts possuem leis e documentos diretos  relacionados a empresa Betano.
            Faça uma análise  completa e extraia todos os dados possíveis, com  base nos documentos disponíveis.
            Não é para inventar  dados, apenas extrair os dados que estão disponíveis.

            Entrada:
            -----------------
            {input}
            -----------------
            Formato de saída:
            {formato_saida}
            """,
            input_variables=["input"],
            partial_variables={"formato_saida": parser.get_format_instructions()}
        )
        
        cadeia = template | llm | parser
        
        try:
            resposta = cadeia.invoke({"input": input})
        except Exception as e:
            return f"Erro ao processar a entrada: {e}"
        
        return resposta