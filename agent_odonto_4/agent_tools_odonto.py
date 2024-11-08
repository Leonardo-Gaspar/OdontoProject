from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.agents.agent import AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
import os
import time
import openai
# Função para criar o agente executor com a ferramenta de análise de imagem
def create_agent_executor():
    load_dotenv()
    
    # Configura a mensagem inicial do sistema para o ChatCompletion
    with open('prompts/system_prompt.txt', 'r') as file:
        system_prompt = " ".join(line.rstrip() for line in file)
    
    # Inicializa o modelo ChatOpenAI com a chave da API
    chat = ChatOpenAI(
        model="gpt-4",
        temperature=0.5,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    memory = ConversationBufferMemory(return_messages=True, memory_key='chat_history')
    
    tools = []
    tools_json = [convert_to_openai_function(tool) for tool in tools]
    
    prompt = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        MessagesPlaceholder(variable_name='chat_history'),
        ('user', '{input}'),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ])
    
    pass_through = RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_to_openai_function_messages(x['intermediate_steps'])
    )
    
    agent_chain = pass_through | prompt | chat | OpenAIFunctionsAgentOutputParser()
    
    agent_executor = AgentExecutor(
        agent=agent_chain,
        tools=tools_json,
        memory=memory,
        verbose=False
    )
    
    return agent_executor

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