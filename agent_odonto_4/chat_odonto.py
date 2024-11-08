import streamlit as st
from agent_tools_odonto import create_agent_executor, model_res_generator
from documentos_scripts.valida_documento import validar_tamanho_arquivo, validar_tipo_documento
from documentos_scripts.carrega_documento import upload_arquivo_com_progresso

def main():
    st.set_page_config(page_title="Assistente OdontoPrev")
    st.title("🦷 Assistente OdontoPrev")

if __name__ == "__main__":
    main()

####################### INTERFACE DE CHAT COM STREAMLIT ########################

if 'agent_executor' not in st.session_state:
    st.session_state['agent_executor'] = create_agent_executor()
agent_executor = st.session_state['agent_executor']

# Função para limpar o histórico do chat
def clear_chat_history():
    st.session_state['messages'] = [{"role": "assistant", "content": "Olá! Sou o assistente da OdontoPrev. Como posso ajudar você com questões odontológicas?"}]
    st.session_state['agent_executor'].memory.clear()
    st.experimental_rerun()

# Botões para limpar o chat ou sair
st.sidebar.button('Limpar histórico', on_click=clear_chat_history)
st.sidebar.button('Sair', on_click=st.stop)

# Upload de documentos odontológicos com validação
uploaded_file = st.sidebar.file_uploader("Envie o documento odontológico para análise", type=["txt", "pdf", "docx"])
if uploaded_file is not None:
    if not validar_tamanho_arquivo(uploaded_file):
        st.sidebar.error("O arquivo enviado excede o tamanho permitido.")
    elif not validar_tipo_documento(uploaded_file):
        st.sidebar.error("Tipo de arquivo inválido. Envie apenas .txt, .pdf ou .docx.")
    else:
        file_path = upload_arquivo_com_progresso(uploaded_file)
        st.sidebar.success("Arquivo enviado com sucesso! A análise está em andamento.")

# Histórico de mensagens do chat
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Olá! Sou o assistente da OdontoPrev. Em que posso ajudar você com relação aos seus cuidados odontológicos?"}]

# Exibindo as mensagens do chat
for message in st.session_state["messages"]:
    if message["role"] == "user":
        st.chat_message(message["role"]).markdown(f"**Você**: {message['content']}")
    else:
        st.chat_message(message["role"]).markdown(f"**Assistente OdontoPrev**: {message['content']}")

# Entrada de texto do chat
if prompt := st.chat_input("Fale com o assistente OdontoPrev"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt, unsafe_allow_html=False)

    # Gerar a resposta do assistente
    with st.chat_message("assistant"):
        message = model_res_generator(prompt)
        st.session_state["messages"].append({"role": "assistant", "content": message})
        st.markdown(message, unsafe_allow_html=False)
