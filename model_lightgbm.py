import pandas as pd
import streamlit as st
from pycaret.regression import *
import os

# Função para carregar o arquivo com base na extensão
def carregar_arquivo(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.ftr'):
            return pd.read_feather(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        else:
            st.error("Formato de arquivo não suportado. Utilize .ftr ou .csv.")
            return None
    else:
        st.error("Nenhum arquivo foi enviado.")
        return None

# Streamlit: título do aplicativo
st.title("Análise de Crédito com PyCaret")

# Carregar o arquivo via uploader
uploaded_file = st.file_uploader("Escolha o arquivo .ftr ou .csv", type=["ftr", "csv"])

# Se o arquivo for carregado, continue o processamento
if uploaded_file is not None:
    df = carregar_arquivo(uploaded_file)
    
    if df is not None:
        st.success("Arquivo carregado com sucesso!")
        st.write("Visualização das primeiras linhas do arquivo:")
        st.dataframe(df.head())

        # Configuração do ambiente PyCaret
        with st.spinner("Configurando o ambiente PyCaret..."):
            reg1 = setup(data=df, target='renda', session_id=123)

        # Treinamento do modelo LightGBM
        with st.spinner("Treinando o modelo LightGBM..."):
            modelo_lightgbm = create_model('lightgbm')

        # Pipeline de transformação
        pipeline = get_config('pipeline')  
        st.write("Pipeline de transformação utilizado:")
        st.write(pipeline)

        # Salvar o modelo e o pipeline
        with st.spinner("Salvando o modelo e o pipeline..."):
            save_path = 'modelo_lightgbm.pkl'
            save_model(modelo_lightgbm, save_path)
            st.success("Modelo e pipeline salvos com sucesso!")

        # Botão para download do modelo treinado
        with open(save_path, 'rb') as f:
            st.download_button(
                label="Baixar modelo treinado",
                data=f,
                file_name='modelo_lightgbm.pkl',
                mime='application/octet-stream'
            )
            
else:
    st.info("Por favor, envie o arquivo .ftr ou .csv.")
