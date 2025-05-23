import streamlit as st
from roboflow import Roboflow
import tempfile
import os
import shutil
from datetime import datetime

# ========== CONFIGURAÇÕES INICIAIS ==========

# Diretório para salvar imagens processadas
SAVE_DIR = 'saved_images'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Função para salvar imagens originais e anotadas
def save_images(original_path, annotated_path):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')
    original_filename = f'original_{timestamp}.jpg'
    annotated_filename = f'annotated_{timestamp}.jpg'
    original_save_path = os.path.join(SAVE_DIR, original_filename)
    annotated_save_path = os.path.join(SAVE_DIR, annotated_filename)
    shutil.copy(original_path, original_save_path)
    shutil.copy(annotated_path, annotated_save_path)
    return original_save_path, annotated_save_path

# Função para listar pares de imagens salvas
def list_saved_images(save_dir):
    if not os.path.exists(save_dir):
        return []
    files = os.listdir(save_dir)
    jpg_files = [f for f in files if f.endswith('.jpg')]
    originals = sorted([f for f in jpg_files if f.startswith('original_')])
    annotated = sorted([f for f in jpg_files if f.startswith('annotated_')])
    return list(zip(originals, annotated))

# Função para exibir as imagens salvas na página de galeria
def show_saved_images_page(save_dir):
    st.title('📁 Imagens Salvas')
    image_pairs = list_saved_images(save_dir)
    if not image_pairs:
        st.write('Nenhuma imagem salva encontrada.')
        return
    for original, annotated in reversed(image_pairs):  # Mostra as mais recentes primeiro
        st.write(f'**Imagem original:**')
        st.image(os.path.join(save_dir, original), use_container_width=True)
        st.write(f'**Imagem anotada:**')
        st.image(os.path.join(save_dir, annotated), use_container_width=True)
        st.markdown('---')

# ========== INICIALIZAÇÃO DO MODELO ==========
rf = Roboflow(api_key="X1uAA9JaWIBjxliimDY3")
project = rf.workspace("jongboo-u5zow").project("cavity-73rfa")
model = project.version(1).model

# ========== INTERFACE STREAMLIT ==========

st.set_page_config(page_title="Chat Odontológico Roboflow", page_icon="🦷")

# Menu lateral para navegação entre páginas
st.sidebar.title("Navegação")
page = st.sidebar.radio("Escolha a página:", ["Chat Odontológico", "Imagens Salvas"])

if page == "Chat Odontológico":
    st.title("🦷 Chat Odontológico com Roboflow")
    st.write("Envie uma imagem odontológica para avaliação do modelo.")

    uploaded_file = st.file_uploader("Envie sua imagem (JPEG, PNG):", type=["jpeg", "jpg", "png"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        prediction = model.predict(tmp_path, confidence=40).json()

        st.subheader("Resultados da Detecção")
        if prediction["predictions"]:
            for obj in prediction["predictions"]:
                classe = obj["class"]
                confidence = obj["confidence"]
                st.write(f"Detectado: {classe} (Confiança: {confidence:.2f})")
        else:
            st.write("Nenhuma detecção encontrada.")

        # Salvar a imagem anotada
        annotated_path = tmp_path.replace(".jpg", "_annotated.jpg")
        model.predict(tmp_path, confidence=40).save(annotated_path)
        if os.path.exists(annotated_path):
            st.image(annotated_path, caption="Imagem com detecções", use_container_width=True)
            # Salvar ambas as imagens para consulta futura
            save_images(tmp_path, annotated_path)
        else:
            st.warning("Não foi possível gerar a imagem com as detecções.")
    else:
        st.write("Por favor, envie uma imagem para avaliação.")

elif page == "Imagens Salvas":
    show_saved_images_page(SAVE_DIR)
