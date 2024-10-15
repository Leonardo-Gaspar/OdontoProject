import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.agents.agent import AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from agent_tools_okto_3 import retorna_condicoes
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from dotenv import load_dotenv
import os

def create_agent_executor():
    load_dotenv()
    
    chat = ChatOpenAI(model="gpt-4",
                              temperature=0.5,
                              api_key=os.getenv("OPENAI_API_KEY"))
    
    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key='chat_history'
    )
    
    tools = [retorna_condicoes]
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

if 'agent_executor' not in st.session_state:
    st.session_state['agent_executor'] = create_agent_executor()

agent_executor = st.session_state['agent_executor']

####################### INTERFACE DE CHAT COM STREAMLIT ########################

st.set_page_config(page_title="Converse com o Assistente Jurídico")
st.title("💬 Assistente Jurídico Okto")

def model_res_generator(entrada: str):
    resposta = agent_executor.invoke(
        {'input': entrada}
    )
    return resposta['output']

def clear_chat_history():
    st.session_state['messages'] = [{"role": "assistant", "content": "Olá, tudo bem? Em que posso ajudar?"}]
    st.session_state['agent_executor'].memory.clear()

st.sidebar.button('Limpar histórico', on_click=clear_chat_history)
st.sidebar.button('Sair', on_click=st.stop)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Olá, tudo bem? Em que posso ajudar?"}]

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Fale com o assistente jurídico"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt, unsafe_allow_html=False)
    
    with st.chat_message("assistant"):
        message = model_res_generator(prompt)
        st.session_state["messages"].append({"role": "assistant", "content": message})
        st.markdown(message, unsafe_allow_html=False)