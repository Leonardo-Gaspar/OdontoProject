import streamlit as st
import os
from documentos_scripts.processa_documento import process_user_document

UPLOAD_FOLDER = 'uploaded_documents'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
#Função para  carregar o arquivo
def upload_arquivo_com_progresso(uploaded_file):
    with st.spinner('Processando documento...'):
        file_name = uploaded_file.name
        file_path = os.path.join(UPLOAD_FOLDER, file_name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f'Arquivo {file_name} salvo com sucesso!')
        process_user_document(file_path)  
        return file_path  