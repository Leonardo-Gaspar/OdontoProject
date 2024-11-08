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
import time
import openai

# Classe corrigida como subclasse de BaseModel do Pydantic
class AnalyzeImageArgs(BaseModel):
    image_file: bytes = Field(..., description='Arquivo de imagem para análise')

@tool(args_schema=AnalyzeImageArgs)
def analyze_image_for_human(image_file: bytes):
    """
    Envia uma imagem para a API da OpenAI para verificar se há uma pessoa presente.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    try:
        response = openai.Image.create(
            file=image_file,
            model="gpt-4-turbo",
            purpose="image_analysis",
            prompt="A imagem contém uma pessoa?"
        )
        result_text = response["choices"][0]["text"].strip()

        if "sim" in result_text.lower():
            return "A imagem contém uma pessoa."
        elif "não" in result_text.lower():
            return "A imagem não contém uma pessoa."
        else:
            return "Não foi possível determinar se a imagem contém uma pessoa."
    
    except Exception as e:
        return f"Erro ao analisar a imagem: {e}"

def create_agent_executor():
    load_dotenv()
    
    chat = ChatOpenAI(model="gpt-4", temperature=0.5, api_key=os.getenv("OPENAI_API_KEY"))
    
    memory = ConversationBufferMemory(return_messages=True, memory_key='chat_history')

    tools = [analyze_image_for_human]
    tools_json = [convert_to_openai_function(tool) for tool in tools]

    with open('prompts/system_prompt.txt', 'r') as file:
        system_prompt = " ".join(line.rstrip() for line in file)
    
    prompt = ChatPromptTemplate.from_messages([('system', system_prompt),
                                              MessagesPlaceholder(variable_name='chat_history'),
                                              ('user', '{input}'),
                                              MessagesPlaceholder(variable_name='agent_scratchpad')])
    
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
