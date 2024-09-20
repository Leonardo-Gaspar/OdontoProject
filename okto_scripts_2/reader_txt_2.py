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
        try:
            documentos.extend(loader.load())
        except FileNotFoundError as fnf_error:
            print(f"Arquivo não encontrado: {fnf_error}")
        except Exception as e:
            print(f"Erro ao carregar documentos: {e}")
    
    return documentos

class CategoriaDocumento(str, Enum):
    LEI = "Lei"
    POLITICA = "Política"
    REGRAS = "Regras"
    OUTRO = "Outro"

class ExtratorDeBetano(BaseModel):
    titulo: str
    conteudo: str
    categoria: CategoriaDocumento

class TxtReader(BaseTool):
    name = "TxtReader"
    description = """Ferramenta para leitura e extração de informações dos documentos TXTs relacionados à Betano.
    A resposta deve ser baseada diretamente nos documentos disponíveis."""
    
    documentos_cache: list = None  # Definindo o documentos_cache na classe
    
    def __init__(self):
        super().__init__()

    def carregar_documentos_cache(self):
        if self.documentos_cache is None:
            try:
                self.documentos_cache = carregar_documentos()
            except Exception as e:
                print(f"Erro ao carregar documentos: {str(e)}")
                raise e
            
    def _run(self, input: str) -> str:
        self.carregar_documentos_cache()
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Chave da API OpenAI não foi configurada.")
            llm = ChatOpenAI(model="gpt-4", temperature=0.5, api_key=api_key)
                            
            parser = JsonOutputParser(pydantic_object=ExtratorDeBetano)
            
            template = PromptTemplate(
                template="""
                    Analise a entrada fornecida e extraia informações dos documentos sobre a Betano, como leis, políticas, termos e condições, e regras de apostas.

                    Instruções:
                    - Baseie-se exclusivamente nas informações dos documentos.
                    - Se a informação não for encontrada, diga "Informação não encontrada", sem adivinhar.
                    - Explique brevemente como chegou à conclusão, citando os documentos.
                    - Seja conciso e objetivo.

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
            
            # Execute o prompt com o modelo e o parser
            prompt = template.format(input=input)
            resposta_modelo = llm.generate(prompt)
            resposta = parser.parse(resposta_modelo)
            
        except Exception as e:
            return f"Erro inesperado: {str(e)}"
        
        return resposta