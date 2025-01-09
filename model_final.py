import pandas as pd
import streamlit as st
from pycaret.regression import setup, load_model, predict_model

# Caminho do modelo salvo
model_path = r"C:\Users\User\Documents\VHR\EBAC\Cientista de Dados\Profissão Cientista de Dados\Módulo 38_Streamlit VI e Pycaret\Projeto Final\model_final"

# Função para carregar o modelo
@st.cache_resource
def carregar_modelo(caminho):
    try:
        modelo = load_model(caminho)
        return modelo
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

# Streamlit App
st.title("Previsão de Renda com PyCaret")

# Upload do arquivo CSV para previsão
uploaded_file = st.file_uploader("Escolha o arquivo .csv para previsão", type="csv")

if uploaded_file is not None:
    # Carregar os dados
    input_data = pd.read_csv(uploaded_file)
    st.write("Dados carregados:")
    st.dataframe(input_data.head())

    # Verificar e converter a coluna `data_ref` para datetime
    if 'data_ref' in input_data.columns:
        try:
            input_data['data_ref'] = pd.to_datetime(input_data['data_ref'])
            st.success("Coluna 'data_ref' convertida para datetime com sucesso!")
        except Exception as e:
            st.error(f"Erro ao converter 'data_ref' para datetime: {e}")
            st.stop()
    else:
        st.warning("A coluna 'data_ref' não foi encontrada nos dados.")

    # Inicializar o ambiente PyCaret
    with st.spinner("Inicializando o ambiente PyCaret..."):
        try:
            setup(
                data=input_data,
                target='renda',
                session_id=123,
                log_experiment=False,
                html=False
            )
        except Exception as e:
            st.error(f"Erro ao inicializar o ambiente PyCaret: {e}")
            st.stop()

    # Carregar o modelo treinado
    modelo = carregar_modelo(model_path)

    if modelo:
        # Fazer previsões
        with st.spinner("Realizando previsões..."):
            try:
                previsoes = predict_model(modelo, data=input_data)

                # Exibir os resultados completos
                st.write("Resultado completo retornado por `predict_model`:")
                st.dataframe(previsoes)

                # Botão para download do resultado completo
                csv_completo = previsoes.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Baixar Resultados Completos em CSV",
                    data=csv_completo,
                    file_name='resultados_completos.csv',
                    mime='text/csv'
                )
            except Exception as e:
                st.error(f"Erro ao realizar previsões: {e}")
else:
    st.info("Por favor, envie um arquivo CSV para realizar as previsões.")
