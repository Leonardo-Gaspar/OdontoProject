from langchain_community.embeddings.sentence_transformer import HuggingFaceEmbeddings
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.pydantic_v1 import BaseModel, Field
from langchain.agents import tool
import pandas as pd


class consulta_tipo_conducao_args(BaseModel):
    query: str = Field(description='pergunta do usuario')
    uso_veiculo: str = Field(description='forma de uso e contratacao informada pelo usuário para localizar a resposta motorista aplicativo ou motorista particular')

@tool(args_schema=consulta_tipo_conducao_args)
def retorna_condicoes(query: str, uso_veiculo: str):
    '''
    Retorna regras e condições, comerciais e contratuais para resposta ao usuário conforme o tipo de condução informada pelo usuário,
    restringindo a resposta apenas ao que existe na base vetorizada.
    '''
    
    # Definir a coleção com base no tipo de uso do veículo
    if 'aplicativo' in uso_veiculo.lower():
        collection_name = 'motorista_aplicativo'
    else:
        collection_name = 'motorista_particular'

    # Utilizando o modelo de embeddings da Hugging Face (paraphrase-multilingual-MiniLM-L12-v2)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    persist_directory = "vector_db"

    # Carregar o banco de dados vetorial para a coleção especificada
    vectordb = Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embedding_model
    )

    # Realizar a pesquisa no banco de dados vetorial
    results = vectordb.similarity_search(query.lower(), k=10)  # k=10 retorna os 10 resultados mais similares
    
    # Se nenhum resultado for encontrado, retorne uma mensagem apropriada
    if not results:
        return "Nenhuma informação relevante encontrada para sua consulta na base de dados."

    # Retornar as informações encontradas, garantindo que seja apenas o conteúdo presente na base vetorizada
    
    return results

class estoque_args(BaseModel):
    query: str = Field(description='pergunta do usuario')

@tool(args_schema=estoque_args)
def carros_estoque(query: str):
    '''
    Retorna os carros, categoria, opcionais, tipo, valor, preco e forma de pagamento de acordo com o uso informado pelo usuario
    '''
    df = pd.read_excel(
        "arquivos/estoque.xlsx"
    )
    chat = ChatOpenAI(model='gpt-3.5-turbo-0125')
    agent = create_pandas_dataframe_agent(
        chat,
        df,
        verbose=False,
        allow_dangerous_code=True,
        agent_type='tool-calling'
    )
    return df #agent.invoke({'input':query})
