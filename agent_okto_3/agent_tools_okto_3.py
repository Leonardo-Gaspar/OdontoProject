from pydantic import BaseModel, Field
from langchain.agents import tool
import pickle
import os

class consulta_tipo_documentacao_args(BaseModel):
    query: str = Field(description='pergunta do usuario')
    questao_usuario: str = Field(description='Questão relacionada à faq, lei, decreto, portaria')
    
def carregar_faiss(collection_name):
    faiss_file = f'agent_okto_3/faiss_index_{collection_name}.pkl'
    if not os.path.exists(faiss_file):
        raise FileNotFoundError(f"O arquivo {faiss_file} não foi encontrado.")
    with open(faiss_file, "rb") as f:
        vectordb = pickle.load(f)
    return vectordb

@tool(args_schema=consulta_tipo_documentacao_args)
def retorna_condicoes(query: str, questao_usuario: str):
    '''
    Retorna regras e condições comerciais e contratuais para resposta ao usuário,
    conforme o tipo de condução informada pelo usuário, restringindo a resposta apenas
    ao que existe na base vetorizada.
    '''

    questao_usuario_lower = questao_usuario.lower()
    if 'lei' in questao_usuario_lower:
        collection_name = 'lei'
    elif 'decreto' in questao_usuario_lower:
        collection_name = 'decreto'
    elif 'portaria' in questao_usuario_lower:
        collection_name = 'portaria'
    else:
        collection_name = 'faq'

    vectordb = carregar_faiss(collection_name)

    results = vectordb.similarity_search(query.lower(), k=10)

    if not results:
        return f"Nenhuma informação relevante encontrada na categoria '{collection_name}' para sua consulta."

    return results
