from langchain_community.document_loaders import TextLoader 
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
from pydantic import BaseModel, Field
from enum import Enum

def carregar_documentos():
    loaders = [
        TextLoader("okto_documents/apostas_de_quota_fixa.txt", encoding="utf-8"),
        TextLoader("okto_documents/autorizacoes_aposta.txt", encoding="utf-8"),
        TextLoader("okto_documents/loterias.txt", encoding="utf-8"),
        TextLoader("okto_documents/promocao_comercial.txt", encoding="utf-8"),
        TextLoader("okto_documents/promocoes_comerciais.txt",encoding="utf-8")      
    ]

    documentos = []
    for loader in loaders:
        try:
            documentos.extend(loader.load())
        except FileNotFoundError as fnf_error:
            print(f"Arquivo não encontrado: {fnf_error}")
        except Exception as e:
            print(f"Erro ao carregar documentos: {e}")
    
    return documentos

class CategoriaDocumento(BaseModel):
    LEI = "Lei", "Artigo", "Legislação"
    DOCUMENTACAO = "Documentação", "Política"

class ExtratorDeOkto(BaseModel):
    okto_dados: str = Field("Extrator de todos os dados obtidos através de artigos, leis e documentação Okto.")

class TxtReader(BaseTool):
    name = "TxtReader"
    description = """Ferramenta para leitura e extração de informações dos documentos TXTs relacionados à Okto.
    A resposta deve ser baseada diretamente nos documentos disponíveis."""
    
    documentos_cache = [] 
    
    def __init__(self):
        super().__init__()

    def carregar_documentos_cache(self):
        if self.documentos_cache is None:
            try:
                self.documentos_cache = carregar_documentos()
            except Exception as e:
                print(f"Erro ao carregar documentos: {str(e)}")
                raise e
            return self.documentos_cache
            
    def _run(self, input: str) -> str:
        self.carregar_documentos_cache()
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Chave da API OpenAI não foi configurada.")
            llm = ChatOpenAI(model="gpt-4", temperature=0.5, api_key=api_key)
                            
            parser = JsonOutputParser(pydantic_object=ExtratorDeOkto)
            
            template = PromptTemplate(
                template="""
                    Você deve analisar a entrada a seguir e extrair as informações relevantes com base nos documentos disponíveis, os quais foram retirados da classe TxtReader.
                    
                    Instruções:
                    
                    - Baseie-se exclusivamente nas informações dos documentos. 
                    - Procure por palavras chaves e devolva as informações relevantes, sem inventar dados.
                    - Se a informação não for encontrada, diga "Informação não encontrada", sem inventar dados.
                    - Explique brevemente como chegou à conclusão, sem inventar dados e sem citar 'Lei XYZ'."
                    - Seja conciso, claro e objetivo, sem inventar dados.
                    - Respire fundo, faça com calma e minimize erros, sem inventar dados.

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
            resposta = cadeia.invoke({"input": input})
        except Exception as e:
            return f"Erro inesperado: {str(e)}"
        
        return resposta