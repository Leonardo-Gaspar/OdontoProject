from pydantic import BaseModel, Field
from langchain.agents import tool
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.agents.agent import AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
import os
from faiss_scripts.faiss import carregar_faiss
import streamlit as st
import time

# Definição da classe e ferramenta
class consulta_tipo_documentacao_args(BaseModel):
    query: str = Field(description='pergunta do usuario')
    questao_usuario: str = Field(description='Questão relacionada à faq, lei, decreto, portaria')

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
    elif os.path.exists(f'faiss_container/faiss_index_user_{questao_usuario}.pkl'):
        collection_name = f'faiss_container/faiss_index_user_{questao_usuario}'
    else:
        collection_name = 'faq'

    vectordb = carregar_faiss(collection_name)

    results = vectordb.similarity_search(query.lower(), k=10)
    return results

# Função para criar o agente executor
def create_agent_executor():
    load_dotenv()
    
    chat = ChatOpenAI(model="gpt-4",
                      temperature=0.5,
                      api_key=os.getenv("OPENAI_API_KEY"))
    
    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key='chat_history'
    )
    
    tools = [retorna_condicoes]  # Usa a função definida no mesmo arquivo
    tools_json = [convert_to_openai_function(tool) for tool in tools]
    
    with open('prompts/system_prompt.txt', 'r') as file:
        system_prompt = " ".join(line.rstrip() for line in file)
    
    prompt = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        MessagesPlaceholder(variable_name='chat_history'),
        ('user', '{input}'),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ])
    
    pass_through = RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_to_openai_function_messages(x['intermediate_steps'])
    )
    
    agent_chain = pass_through | prompt | chat.bind(functions=tools_json) | OpenAIFunctionsAgentOutputParser()
    
    agent_executor = AgentExecutor(
        agent=agent_chain,
        tools=tools,
        memory=memory,
        verbose=False
    )
    
    return agent_executor

# Função geradora de respostas com indicador de carregamento
# def model_res_generator(entrada: str):
#     try:
#         with st.spinner('Gerando resposta...'):
#             agent_executor = st.session_state['agent_executor']
#             resposta = agent_executor.invoke({'input': entrada})
#             time.sleep(0.1)
#             return resposta['output']
#     except Exception as e:
#         st.error(f"Ocorreu um erro: {str(e)}")
#         return "Desculpe, não consegui processar sua solicitação no momento."


# Função geradora de respostas com indicador de carregamento
def model_res_generator(entrada: str):
    try:
        print("Gerando resposta, por favor aguarde...")
        time.sleep(2)      
        agent_executor = create_agent_executor() 
        if hasattr(agent_executor, 'invoke'):
            resposta = agent_executor.invoke({'input': entrada})
        else:
            raise AttributeError("O agent_executor não possui o método 'invoke'")
        return resposta['output']
    except Exception as e:
        print(f"Ocorreu um erro: {str(e)}")
        return "Desculpe, não consegui processar sua solicitação no momento."