"""
To run this app, open the terminal and type:
    streamlit run APP.py

This application allows you to predict crude oil properties using trained Deep Learning models.
Ensure that the models, scalers, and JSON column files are located in the correct directory
before running the app.

Author: Willian Teixeira
Repository: https://github.com/TeixeiraWILLIAN/DeepLearning
Python Version: 3.10+
Dependencies: Streamlit, os, json, joblib, numpy, tensorflow, pandas e pathlib
"""

import streamlit as st
import os
import json
import joblib
import numpy as np
import tensorflow as tf
import pandas as pd
from pathlib import Path

# CONFIGURATIONS AND FUNCTIONS FROM ITERATIVE_PREDICTION.py

# SETTINGS
PROPRIEDADES = ['Density', 'Pour_Point', 'Wax',
                'Asphaltene', 'Viscosity_20C', 'Viscosity_50C']
UNIDADES = {
    'Density': 'kg/L',
    'Pour_Point': '°C',
    'Wax': 'wght %',
    'Asphaltene': 'wght %',
    'Viscosity_20C': 'mPa.s',
    'Viscosity_50C': 'mPa.s'
}

RANGES_TIPICOS = {
    'Density': (0.68, 0.98),
    'Pour_Point': (-82, 30),
    'Wax': (0, 45),
    'Asphaltene': (0, 15),
    'Viscosity_20C': (0, 5500),
    'Viscosity_50C': (0.4, 1160)
}

# Model performance (R²)
# These values should be updated after the final tuning
DESEMPENHO_MODELOS = {
    'Density': 0.93,
    'Viscosity_20C': 0.97,
    'Pour_Point': 0.30,
    'Asphaltene': 0.63,
    'Wax': 0.31,
    'Viscosity_50C': 0.91
}

# Path to models
MODELS_FOLDER = 'Results_model'

# Dictionary to store model loading status
LOAD_STATUS = {}

# Cached loading functions for optimization


@st.cache_resource
def load_tf_model(propriedade):
    """Loads the TensorFlow model."""
    modelo_path = os.path.join(
        MODELS_FOLDER, propriedade, f'modelo_{propriedade}.keras')
    if not os.path.exists(modelo_path):
        LOAD_STATUS[propriedade] = f"Modelo .keras não encontrado em: {modelo_path}"
        return None
    try:
        # Streamlit handles model caching automatically
        return tf.keras.models.load_model(modelo_path)
    except Exception as e:
        LOAD_STATUS[propriedade] = f"Erro ao carregar modelo .keras: {e}"
        return None


@st.cache_data
def load_joblib_data(propriedade, suffix):
    """Loads scalers and JSON columns with cached data."""

    # Load column JSON file
    if suffix == 'colunas_entrada':
        file_path = os.path.join(
            MODELS_FOLDER, propriedade, 'colunas_entrada.json')
        if not os.path.exists(file_path):
            LOAD_STATUS[
                propriedade] = f"Arquivo de colunas .json não encontrado em: {file_path}"
            return None
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            LOAD_STATUS[propriedade] = f"Erro ao carregar colunas .json: {e}"
            return None

    # Load scaler .pkl files
    file_path = os.path.join(MODELS_FOLDER, propriedade,
                             f'{propriedade}_{suffix}.pkl')
    if not os.path.exists(file_path):
        LOAD_STATUS[propriedade] = f"Scaler .pkl não encontrado em: {file_path}"
        return None
    try:
        return joblib.load(file_path)
    except Exception as e:
        LOAD_STATUS[propriedade] = f"Erro ao carregar scaler .pkl: {e}"
        return None


def carregar_modelo_completo(propriedade):
    """Main function to load all model artifacts."""
    LOAD_STATUS.clear()
    modelo = load_tf_model(propriedade)
    scaler_x = load_joblib_data(propriedade, 'normalizador_x')
    scaler_y = load_joblib_data(propriedade, 'normalizador_y')
    colunas_entrada = load_joblib_data(propriedade, 'colunas_entrada')

    if modelo and scaler_x and scaler_y and colunas_entrada:
        return modelo, scaler_x, scaler_y, colunas_entrada
    else:
        return None, None, None, None


def fazer_predicao(modelo, scaler_x, scaler_y, valores_entrada, colunas_esperadas):
    """Performs the prediction."""
    try:
        # Arrange input data in the correct order
        X = np.array([[valores_entrada[col] for col in colunas_esperadas]])

        # Normalize
        X_norm = scaler_x.transform(X)

        # Predict
        y_pred_norm = modelo.predict(X_norm, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_norm)

        return y_pred[0][0]
    except Exception as e:
        st.error(f"Erro ao fazer predição: {str(e)}")
        return None

# STREAMLIT INTERFACE


# Page configuration
st.set_page_config(
    page_title="Predição de Propriedades do Petróleo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and header
st.title("Predição de Propriedades do Petróleo Bruto")
st.markdown("---")

# SIDEBAR: Property selection
with st.sidebar:
    st.header("Configuração da Predição")

    # Target property selection
    propriedade_alvo = st.selectbox(
        "Selecione a Propriedade a Predizer:",
        options=PROPRIEDADES,
        index=PROPRIEDADES.index(
            'Viscosity_50C') if 'Viscosity_50C' in PROPRIEDADES else 0
    )

    # Load model
    modelo, scaler_x, scaler_y, colunas_entrada = carregar_modelo_completo(
        propriedade_alvo)

    # Display R² and confidence
    if modelo is not None:
        r2 = DESEMPENHO_MODELOS.get(propriedade_alvo, 0.0)

        if r2 > 0.90:
            status = "Alta Confiabilidade"
            cor = "green"
        elif r2 > 0.50:
            status = "Moderada Confiabilidade"
            cor = "orange"
        else:
            status = "Baixa Confiabilidade"
            cor = "red"

        st.markdown("---")
        st.subheader("Desempenho do Modelo")
        st.metric(label="R² do Modelo", value=f"{r2:.4f}")
        st.markdown(f"Status: :{cor}[**{status}**]")

# MAIN CONTENT: Data Input and Prediction
if modelo is not None:

    # Expected input columns (all except target)
    colunas_entrada_esperadas = [
        col for col in colunas_entrada if col != propriedade_alvo]

    st.header(f"Insira os Valores para Predizer **{propriedade_alvo}**")
    st.markdown(
        f"A propriedade alvo será predita em **{UNIDADES[propriedade_alvo]}**.")
    st.markdown("---")

    valores_entrada = {}

    # Create input columns for layout
    num_cols = 2
    cols = st.columns(num_cols)

    for i, prop in enumerate(colunas_entrada_esperadas):
        col_idx = i % num_cols

        min_val, max_val = RANGES_TIPICOS.get(prop, (0.0, 100.0))
        unidade = UNIDADES.get(prop, '-')

        # Use number_input for precision
        with cols[col_idx]:
            valor = st.number_input(
                label=f"{prop} ({unidade})",
                min_value=float(min_val),
                max_value=float(max_val),
                value=(min_val + max_val) / 2,
                step=(max_val - min_val) / 100,  # Smaller step for precision
                format="%.4f",
                help=f"Range típico: {min_val} a {max_val} {unidade}"
            )
            valores_entrada[prop] = valor

    st.markdown("---")

    # Prediction button
    if st.button("Fazer Predição", type="primary", use_container_width=True):

        with st.spinner("Processando predição..."):
            predicao = fazer_predicao(
                modelo, scaler_x, scaler_y, valores_entrada, colunas_entrada_esperadas
            )

        if predicao is not None:

            # Display result
            st.success("Predição Concluída!")

            col_res, col_r2 = st.columns([2, 1])

            with col_res:
                st.metric(
                    label=f"Valor Predito para {propriedade_alvo}",
                    value=f"{predicao:.4f} {UNIDADES[propriedade_alvo]}",
                    delta=None
                )

            with col_r2:
                # Display R² and reliability again
                st.metric(label="R² do Modelo", value=f"{r2:.4f}")
                st.info(f"Confiança: **{status}**")

            st.markdown("---")
            st.subheader("Valores de Entrada Utilizados")

            # Show input values as a DataFrame for clarity
            df_entrada = pd.DataFrame(
                {
                    "Propriedade": valores_entrada.keys(),
                    "Valor": [f"{v:.4f}" for v in valores_entrada.values()],
                    "Unidade": [UNIDADES[p] for p in valores_entrada.keys()]
                }
            ).set_index("Propriedade")

            st.dataframe(df_entrada, use_container_width=True)

else:
    # Show detailed error if model loading fails
    st.error(
        "Não foi possível carregar os modelos. Verifique a estrutura de arquivos.")
    st.info(
        f"O Streamlit está procurando os modelos na pasta: `{MODELS_FOLDER}`")

    if LOAD_STATUS:
        st.subheader("Detalhes do Erro de Carregamento:")
        # Display individual error messages
        for prop, error_msg in LOAD_STATUS.items():
            st.warning(f"Propriedade {prop}: {error_msg}")

# Footer
st.markdown("---")
st.markdown(
    "Projeto de Deep Learning para Predição de Propriedades do Petróleo Bruto.")
