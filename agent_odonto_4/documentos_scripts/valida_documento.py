import streamlit as st

# Função para validar o tipo de documento
def validar_tipo_documento(file):
    try:
        if file.type not in ['text/plain', 'application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
            st.error("Tipo de arquivo não suportado. Por favor, envie um arquivo .txt, .pdf ou .docx.")
            return False
    except AttributeError as e:
        st.error(f"Erro ao acessar o tipo de arquivo: {e}")
        return False
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado ao validar o tipo do arquivo: {e}")
        return False
    return True

# Função para validar o tamanho do documento
def validar_tamanho_arquivo(file):
    try:
        max_size = 10 * 1024 * 1024 
        if file.size > max_size:
            st.error(f"O arquivo enviado é muito grande. O tamanho máximo permitido é de 10MB.")
            return False
    except AttributeError as e:
        st.error(f"Erro ao acessar o tamanho do arquivo: {e}")
        return False
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado ao validar o tamanho do arquivo: {e}")
        return False
    return True