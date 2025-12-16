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

# ==========================================================
# 📦 IMPORTS
# ==========================================================
import streamlit as st
import os
import json
import joblib
import numpy as np
import tensorflow as tf
import pandas as pd
from pathlib import Path

# ============================================
# ⚙️ DEFAULT SETTINGS (by property)
# ============================================
PROPRIEDADES = ['Density', 'Pour_Point', 'Wax',
                'Asphaltene', 'Viscosity_20C', 'Viscosity_50C']

DISPLAY_NAMES = {p: p.replace("_", " ").replace(
    "20C", "20°C").replace("50C", "50°C") for p in PROPRIEDADES}

UNIDADES = {
    'Density': 'kg/L',
    'Pour_Point': '°C',
    'Wax': 'wght %',
    'Asphaltene': 'wght %',
    'Viscosity_20C': 'mPa.s',
    'Viscosity_50C': 'mPa.s'
}

RANGES_TIPICOS = {
    'Density': (0.75, 0.99),
    'Pour_Point': (-90, 30),
    'Wax': (0, 50),
    'Asphaltene': (0, 15),
    'Viscosity_20C': (0, 5400),
    'Viscosity_50C': (0, 1200)
}

# Model performance (R²)
# These values must be updated after the final tuning
DESEMPENHO_MODELOS = {
    'Density': 0.94,
    'Viscosity_20C': 0.98,
    'Pour_Point': 0.31,
    'Asphaltene': 0.63,
    'Wax': 0.31,
    'Viscosity_50C': 0.91
}

BASE_DIR = Path(__file__).resolve().parent
MODELS_FOLDER = BASE_DIR / "Results_model"


# Dictionary to store the loading status
LOAD_STATUS = {}

# Cached loading functions for optimization


@st.cache_resource
def load_tf_model(propriedade):
    """Loads the TensorFlow model."""
    # Uses Path to safely build the file path
    modelo_path = Path(MODELS_FOLDER) / propriedade / \
        f'modelo_{propriedade}.keras'

    if not modelo_path.exists():
        LOAD_STATUS[propriedade] = f"Modelo .keras não encontrado em: {modelo_path}"
        return None
    try:
        # Streamlit handles the caching of the TensorFlow model
        return tf.keras.models.load_model(str(modelo_path))
    except Exception as e:
        LOAD_STATUS[propriedade] = f"Erro ao carregar modelo .keras: {e}"
        return None


@st.cache_data
def load_joblib_data(propriedade, suffix):
    """Loads the scalers and column data using cache."""

    # Logic to load the JSON file containing the input columns
    if suffix == 'colunas_entrada':
        file_path = Path(MODELS_FOLDER) / propriedade / 'colunas_entrada.json'
        if not file_path.exists():
            LOAD_STATUS[
                propriedade] = f"Arquivo de colunas .json não encontrado em: {file_path}"
            return None
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            LOAD_STATUS[propriedade] = f"Erro ao carregar colunas .json: {e}"
            return None

    # Logic to load the .pkl scalers
    file_path = Path(MODELS_FOLDER) / propriedade / \
        f'{propriedade}_{suffix}.pkl'
    if not file_path.exists():
        LOAD_STATUS[propriedade] = f"Scaler .pkl não encontrado em: {file_path}"
        return None
    try:
        return joblib.load(str(file_path))
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
        # Prepares the data in the correct order.
        X = np.array([[valores_entrada[col] for col in colunas_esperadas]])

        # Normalizes
        X_norm = scaler_x.transform(X)

        # Prediz
        y_pred_norm = modelo.predict(X_norm, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_norm)

        return y_pred[0][0]
    except Exception as e:
        st.error(f"Error during estimation: {str(e)}")
        return None

# ===================
# INTERFACE STREAMLIT
# ===================

# Page configuration
st.set_page_config(
    page_title="ECOPANN",
    layout="wide",
    initial_sidebar_state="expanded"
)

# title
st.markdown(
    """
    <div style='line-height: 1.2;'>
        <h1 style='margin-bottom: 0px;'>
            <strong><span style='color:#F36B5B;'>ECOPANN</span></strong>
        </h1>
        <p style='font-size:26px; font-weight:normal; margin-top:0px;'>
            Estimation of Crude Oil Properties Using Artificial Neural Networks
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

# SIDEBAR: Property Selection
with st.sidebar:
    st.header("Estimation Settings")

    # Target Property Selection
    propriedade_alvo = st.selectbox(
        "Select the property you want to estimate:",
        options=PROPRIEDADES,
        format_func=lambda x: DISPLAY_NAMES[x],
        index=PROPRIEDADES.index(
            'Viscosity_50C') if 'Viscosity_50C' in PROPRIEDADES else 0
    )

    # Load model
    modelo, scaler_x, scaler_y, colunas_entrada = carregar_modelo_completo(
        propriedade_alvo)

    # Display R² and Confidence
    if modelo is not None:
        r2 = DESEMPENHO_MODELOS.get(propriedade_alvo, 0.0)

        if r2 > 0.90:
            status = "High Reliability"
            cor = "green"
        elif r2 > 0.50:
            status = "Moderate Reliability"
            cor = "orange"
        else:
            status = "Low Reliability"
            cor = "red"

        st.markdown("---")
        st.subheader("Model Performance")
        st.metric(label="Model R²", value=f"{r2:.4f}")
        st.markdown(f"Status: :{cor}[**{status}**]")


# MAIN CONTENT: Data Input and Prediction
if modelo is not None:

    # Expected input columns (all except the target)
    colunas_entrada_esperadas = [
        col for col in colunas_entrada if col != propriedade_alvo]

    st.markdown(
        f"""
        <h2>
            Enter the values to estimate for 
            <span style='color:#FA8072;'><strong>{DISPLAY_NAMES[propriedade_alvo]}</strong></span>
        </h2>
        """,
        unsafe_allow_html=True
    )

    valores_entrada = {}

    # Create columns to organize the inputs
    num_cols = 2
    cols = st.columns(num_cols)

    for i, prop in enumerate(colunas_entrada_esperadas):
        col_idx = i % num_cols

        min_val, max_val = RANGES_TIPICOS.get(prop, (0.0, 100.0))
        unidade = UNIDADES.get(prop, '-')

        # Use number_input for greater precision
        with cols[col_idx]:
            valor = st.number_input(
                label=f"{DISPLAY_NAMES[prop]} ({unidade})",
                min_value=float(min_val),
                max_value=float(max_val),
                value=(min_val + max_val) / 2,
                # Smaller step size for precision
                step=(max_val - min_val) / 100,
                format="%.4f",
                help=f"Typical Range: {min_val} a {max_val} {unidade}"
            )
            valores_entrada[prop] = valor

    # Prediction Button
    if st.button("Run Estimation", type="primary", use_container_width=True):

        with st.spinner("Processing estimation..."):
            predicao = fazer_predicao(
                modelo, scaler_x, scaler_y, valores_entrada, colunas_entrada_esperadas
            )

        if predicao is not None:

            # Display Result
            st.success("Estimation Completed!")

            col_res, col_r2 = st.columns([2, 1])

            with col_res:
                st.metric(
                    label=f"Estimated Value for {DISPLAY_NAMES[propriedade_alvo]}",
                    value=f"{predicao:.4f} {UNIDADES[propriedade_alvo]}",
                    delta=None
                )

            with col_r2:
                # Re-display R² and status for emphasis
                st.metric(label="Model R²", value=f"{r2:.4f}")
                st.info(f"Confidence Level: **{status}**")

            st.markdown("---")

            st.subheader("Input Data and Predicted Value")

            # ORIGINAL table with input values
            df_entrada = pd.DataFrame(
                {
                    "Property": list(valores_entrada.keys()),
                    "Value": [f"{v:.4f}" for v in valores_entrada.values()],
                    "Unit": [UNIDADES[p] for p in valores_entrada.keys()]
                }
            )

            # Create additional row with the estimation
            df_estimado = pd.DataFrame(
                {
                    "Property": [f"{DISPLAY_NAMES[propriedade_alvo]} (Estimated)"],
                    "Value": [f"{predicao:.4f}"],
                    "Unit": [UNIDADES[propriedade_alvo]]
                }
            )

            # Concatenate without modifying the original structure
            df_final = pd.concat([df_entrada, df_estimado], ignore_index=True)

            # Display final table
            st.dataframe(df_final, use_container_width=True)


else:
    # Display detailed error if the model fails to load
    st.error(
        "Could not load the models. Please check the file structure.")
    st.info(
        f"Streamlit is looking for the models in the folder: `{MODELS_FOLDER}`")

    if LOAD_STATUS:
        st.subheader("Loading Error Details:")
        # We use st.warning for each detailed error
        for prop, error_msg in LOAD_STATUS.items():
            st.warning(f"Property {prop}: {error_msg}")

# Footer
st.markdown("---")
st.markdown(
    "This Deep Learning project, applied to the estimation of crude oil properties, "
    "represents part of the research activities I have been conducting at the Geoenergia Lab. "
    "I express my gratitude to Prof. Dr. Luiz Adolfo Hegele Junior and MSc. Vinicius Czarnobay for their "
    "technical support, scientific contributions, and continuous guidance throughout the development of this "
    "study.")


BASE = Path(__file__).parent
logo1 = BASE / "figure" / "logo_geoenergia.png"
logo2 = BASE / "figure" / "logo_petrobras.png"
logo3 = BASE / "figure" / "logo_sintef.png"
logo4 = BASE / "figure" / "udescc.png"

with st.sidebar:
    st.image(str(logo1), width=3125)
    st.image(str(logo2), width=3125)
    st.image(str(logo3), width=3125)
    st.image(str(logo4), width=3125)
    