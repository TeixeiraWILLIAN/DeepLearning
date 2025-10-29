"""
=======================================================================
PETROLEUM PROPERTY PREDICTION - FINAL INFERENCE AND RESULT EXPORT
=======================================================================
This script performs the final inference stage for the trained neural
network models used to predict crude oil properties. It loads the
pre-trained models, applies them to new input data, and exports the
predictions in both Excel and XML formats.

Main Features:
- Supports single-property or multi-property prediction ("all" mode).
- Loads trained models, input/output scalers, and input column structures.
- Automatically performs data normalization, prediction, and denormalization.
- Generates detailed XML files:
  * One XML per sample.
  * One aggregated XML file containing all predictions.
- Saves complete prediction datasets in Excel format for each property.

Pipeline Overview:
1. Select the property to predict or run all available properties.
2. Load corresponding artifacts (model, scalers, input columns).
3. Prepare and normalize the input data.
4. Run inference and generate predictions.
5. Export all results to organized output folders.

Developed for petroleum engineering applications involving
machine learning-based modeling of crude oil physical-chemical
properties.

Author: TeixeiraWILLIAN
Repository: [GitHub Project Link]
Python Version: 3.10+
Dependencies: TensorFlow, NumPy, Pandas, Joblib, JSON, OpenPyXL, Logging, OS
=======================================================================
"""

import os
import logging
import pandas as pd
import joblib
import tensorflow as tf

# CONFIGURAÇÃO PRINCIPAL
# Altere aqui para a propriedade que deseja prever.
# Opções: 'Density', 'Pour_Point', 'Wax', 'Asphaltene', 'Viscosity_20C', 'Viscosity_50C'.
# Use 'all' para executar a predição para todas as propriedades.
PROPRIEDADE_A_PREVER = 'Density'  # <<< ALTERE AQUI

# CONFIGURAÇÕES GERAIS (geralmente não precisam ser alteradas)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

PROPERTIES_CONFIG = {
    'Density': 'Density',
    'Pour_Point': 'Pour_Point',
    'Wax': 'Wax',
    'Asphaltene': 'Asphaltene',
    'Viscosity_20C': 'Viscosity_20C',
    'Viscosity_50C': 'Viscosity_50C'
}

BASE_RESULTS_FOLDER = 'Results_model'
INPUT_DATA_FILE = 'propriedades_oleo_SINTEF_final.xlsx'
OUTPUT_PREDICTIONS_FILE = f'{PROPRIEDADE_A_PREVER}_completed.xlsx'


def load_artifacts(property_name: str) -> tuple:
    folder_name = PROPERTIES_CONFIG.get(property_name)
    if not folder_name:
        logging.error(
            f"Configuração para a propriedade '{property_name}' não encontrada.")
        return None, None, None, None

    property_folder = os.path.join(BASE_RESULTS_FOLDER, folder_name)
    logging.info(f"Carregando artefatos da pasta: {property_folder}")

    model_path = os.path.join(property_folder, f'modelo_{property_name}.keras')
    scaler_x_path = os.path.join(
        property_folder, f'{property_name}_normalizador_x.pkl')
    scaler_y_path = os.path.join(
        property_folder, f'{property_name}_normalizador_y.pkl')
    columns_path = os.path.join(property_folder, 'colunas_entrada.json')

    for path in [model_path, scaler_x_path, scaler_y_path, columns_path]:
        if not os.path.exists(path):
            logging.error(f"Arquivo necessário não encontrado: {path}")
            return None, None, None, None

    try:
        model = tf.keras.models.load_model(model_path)
        scaler_x = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)
        input_columns = pd.read_json(columns_path, typ='series').tolist()
        logging.info(
            f"Artefatos para '{property_name}' carregados com sucesso.")
        return model, scaler_x, scaler_y, input_columns
    except Exception as e:
        logging.error(
            f"Falha ao carregar artefatos para '{property_name}': {e}")
        return None, None, None, None


def load_and_prepare_data(input_file: str, input_columns: list) -> pd.DataFrame:
    if not os.path.exists(input_file):
        logging.error(f"Arquivo de entrada não encontrado: {input_file}")
        return None

    try:
        df = pd.read_excel(input_file).rename(columns={
            'Density (kg/L)': 'Density', 'Pour Point (°C)': 'Pour_Point',
            'Wax (wght %)': 'Wax', 'Asphaltene (wght %)': 'Asphaltene',
            'Viscosity (mPa.s) - 20°C': 'Viscosity_20C', 'Viscosity (mPa.s) - 50°C': 'Viscosity_50C'
        })

        if not all(col in df.columns for col in input_columns):
            missing_cols = [
                col for col in input_columns if col not in df.columns]
            logging.error(
                f"Colunas de entrada ausentes no arquivo de dados: {missing_cols}")
            return None

        return df[input_columns]
    except Exception as e:
        logging.error(
            f"Falha ao carregar ou preparar os dados de '{input_file}': {e}")
        return None


def run_prediction(model, scaler_x, scaler_y, data: pd.DataFrame, prop_name: str) -> pd.Series:
    logging.info("Iniciando predição...")
    data_normalized = scaler_x.transform(data.values)
    predictions_normalized = model.predict(data_normalized)
    predictions_denormalized = scaler_y.inverse_transform(
        predictions_normalized)
    logging.info("Predição concluída e resultados desnormalizados.")
    return pd.Series(predictions_denormalized.flatten(), name=f'{prop_name}_predicted')


# EXECUÇÃO DO SCRIPT
if PROPRIEDADE_A_PREVER.lower() == 'all':
    properties_to_run = list(PROPERTIES_CONFIG.keys())
    logging.info("Executando predição para TODAS as propriedades disponíveis.")
else:
    if PROPRIEDADE_A_PREVER not in PROPERTIES_CONFIG:
        logging.error(
            f"Propriedade '{PROPRIEDADE_A_PREVER}' é inválida. Verifique a variável no topo do script.")
        properties_to_run = []
    else:
        properties_to_run = [PROPRIEDADE_A_PREVER]
        logging.info(
            f"Executando predição para a propriedade: {PROPRIEDADE_A_PREVER}")

all_predictions = {}

if properties_to_run:
    for prop in properties_to_run:
        logging.info(f"--- Processando: {prop} ---")

        model, scaler_x, scaler_y, input_columns = load_artifacts(prop)
        if not all([model, scaler_x, scaler_y, input_columns]):
            logging.warning(
                f"Não foi possível carregar os artefatos para '{prop}'. Pulando.")
            continue

        input_data = load_and_prepare_data(INPUT_DATA_FILE, input_columns)
        if input_data is None:
            logging.warning(
                f"Não foi possível carregar os dados de entrada para '{prop}'. Pulando.")
            continue

        predictions = run_prediction(
            model, scaler_x, scaler_y, input_data, prop)

        results_df = input_data.copy()
        results_df[predictions.name] = predictions.values
        all_predictions[prop] = results_df

        # Criar subpasta prediction/PROP
        prediction_dir = os.path.join('Prediction', prop)
        os.makedirs(prediction_dir, exist_ok=True)

        # Salvar arquivos XML individuais por linha
        for idx, row in results_df.iterrows():
            xml_content = "<Prediction>\n"
            for col in row.index:
                xml_content += f"    <{col}>{row[col]}</{col}>\n"
            xml_content += "</Prediction>"

            xml_filename = f"{prop}_{idx+1}.xml"
            xml_path = os.path.join(prediction_dir, xml_filename)

            with open(xml_path, 'w', encoding='utf-8') as f:
                f.write(xml_content)

        # Salvar XML agrupado com todas as predições em um único arquivo
        xml_agrupado = "<Predictions>\n"
        for idx, row in results_df.iterrows():
            xml_agrupado += "  <Prediction>\n"
            for col in row.index:
                xml_agrupado += f"    <{col}>{row[col]}</{col}>\n"
            xml_agrupado += "  </Prediction>\n"
        xml_agrupado += "</Predictions>"

        xml_agrupado_path = os.path.join(
            prediction_dir, f"{prop}_completo.xml")
        with open(xml_agrupado_path, 'w', encoding='utf-8') as f:
            f.write(xml_agrupado)

        logging.info(
            f"Arquivos XML individuais e agrupado salvos na pasta: {prediction_dir}")

    # Salvar arquivo .xlsx individual por propriedade, dentro da pasta prediction/PROP
    for prop_name, df in all_predictions.items():
        xlsx_path = os.path.join(
            'Prediction', prop_name, f'{prop_name}_completed.xlsx')
        df.to_excel(xlsx_path, index=False)
        logging.info(f"Arquivo {xlsx_path} salvo com sucesso.")