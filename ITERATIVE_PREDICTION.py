"""
===============================================================
PETROLEUM PROPERTY PREDICTION SCRIPT
===============================================================
This Python script allows users to manually input petroleum
sample data and predict a selected crude oil property using
a pre-trained neural network model.

Main Features:
- Interactive CLI for property selection and data input.
- Loads TensorFlow (.keras) models and corresponding scalers.
- Displays model reliability based on R² performance.
- Handles normalization and inverse transformation of data.
- Provides prediction results with confidence indicators.

Structure:
1. Configuration section defining properties, units, and ranges.
2. Utility functions for UI handling (screen cleaning, headers, etc.).
3. Core functions for model loading, input parsing, and prediction.
4. Main execution loop allowing continuous predictions.

Developed for petroleum engineering applications focused on
machine learning-based property estimation.

Author: Willian Teixeira
Repository: https://github.com/TeixeiraWILLIAN/DeepLearning
Python Version: 3.10+
Dependencies: TensorFlow, NumPy, joblib, json, os
===============================================================
"""

import os
import json
import joblib
import numpy as np
import tensorflow as tf
from pathlib import Path

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
    'Density': (0.75, 1.05),
    'Pour_Point': (-40, 30),
    'Wax': (0, 20),
    'Asphaltene': (0, 15),
    'Viscosity_20C': (50, 1000),
    'Viscosity_50C': (10, 200)
}

DESEMPENHO_MODELOS = {
    'Density': 0.93,
    'Viscosity_20C': 0.97,
    'Pour_Point': 0.30,
    'Asphaltene': 0.27,
    'Wax': 0.31,
    'Viscosity_50C': 0.91
}

MODELS_FOLDER = 'Results_model'


# FUNÇÕES

def limpar_tela():
    """Clears the console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def exibir_header():
    """Displays the application header"""
    print("=" * 70)
    # ANSI Colors
    print(
        "\033[95m" + "PREDIÇÃO DE PROPRIEDADES DO PETRÓLEO BRUTO".center(70) + "\033[0m")
    print("=" * 70)
    print()


def exibir_menu_propriedades():
    """Displays properties selection menu"""
    print("Selecione a propriedade que deseja PREDIZER:\n")

    for i, prop in enumerate(PROPRIEDADES, 1):
        r2 = DESEMPENHO_MODELOS[prop]

        # Define color and status
        if r2 > 0.90:
            cor = "\033[92m"  # Green
            status = "Alta Confiabilidade"
        elif r2 > 0.50:
            cor = "\033[93m"  # Yellow
            status = "Moderada Confiabilidade"
        else:
            cor = "\033[91m"  # Red
            status = "Baixa Confiabilidade"

        reset = "\033[0m"
        print(f"  {i}. {prop:20} (R² = {r2:7.2f}) {cor}{status}{reset}")

    print()


def obter_propriedade_alvo():
    """Gets the target property from the user"""
    while True:
        exibir_menu_propriedades()
        try:
            escolha = int(
                input("Digite o número da propriedade (1-6): ").strip())
            if 1 <= escolha <= 6:
                return PROPRIEDADES[escolha - 1]
            else:
                print("Opção inválida! Digite um número entre 1 e 6.\n")
        except ValueError:
            print("Entrada inválida!\n")


def carregar_modelo(propriedade):
    """Loads the trained model to a property"""
    pasta = os.path.join(MODELS_FOLDER, propriedade)

    try:
        modelo_path = os.path.join(pasta, f'modelo_{propriedade}.keras')
        scaler_x_path = os.path.join(
            pasta, f'{propriedade}_normalizador_x.pkl')
        scaler_y_path = os.path.join(
            pasta, f'{propriedade}_normalizador_y.pkl')
        colunas_path = os.path.join(pasta, 'colunas_entrada.json')

        # Check if all files exist
        if not all(os.path.exists(p) for p in [modelo_path, scaler_x_path, scaler_y_path, colunas_path]):
            print(
                f"Erro: Arquivos do modelo para '{propriedade}' não encontrados em {pasta}")
            return None, None, None, None

        # Upload files
        modelo = tf.keras.models.load_model(modelo_path)
        scaler_x = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)

        with open(colunas_path, 'r') as f:
            colunas_entrada = json.load(f)

        return modelo, scaler_x, scaler_y, colunas_entrada

    except Exception as e:
        print(f"Erro ao carregar modelo: {str(e)}")
        return None, None, None, None


def obter_valores_entrada(colunas_esperadas):
    """Retrieves user input values"""
    valores = {}

    print("\n" + "=" * 70)
    print(
        "\033[95m" + "INSIRA OS VALORES DAS PROPRIEDADES DE ENTRADA".center(70) + "\033[0m")
    print("=" * 70)
    print()

    for prop in colunas_esperadas:
        min_val, max_val = RANGES_TIPICOS[prop]
        unidade = UNIDADES[prop]

        while True:
            try:
                prompt = f"{prop} ({unidade}) [range típico: {min_val} - {max_val}]: "
                valor = float(input(prompt))
                valores[prop] = valor
                break
            except ValueError:
                print("Entrada inválida! Digite um número.\n")

    return valores


def fazer_predicao(modelo, scaler_x, scaler_y, valores_entrada, colunas_esperadas):
    """Make the prediction"""
    try:
        # Prepare data in the correct order.
        X = np.array([[valores_entrada[col] for col in colunas_esperadas]])

        # Normalize
        X_norm = scaler_x.transform(X)

        # Predict
        y_pred_norm = modelo.predict(X_norm, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_norm)

        return y_pred[0][0]
    except Exception as e:
        print(f"Erro ao fazer predição: {str(e)}")
        return None


def exibir_resultado(propriedade_alvo, predicao, valores_entrada, r2):
    """Displays the prediction result"""
    print("\n" + "=" * 70)
    print("\033[95m" + "RESULTADO DA PREDIÇÃO".center(70) + "\033[0m")
    print("=" * 70)
    print()

    # Main result
    unidade = UNIDADES[propriedade_alvo]
    print(f"Propriedade a Predizer: \033[91m {propriedade_alvo} \033[0m")
    print(f"Valor Predito: \033[91m{predicao:.4f} {unidade} \033[0m")
    print()

    # Model performance
    if r2 > 0.90:
        cor = "\033[92m"  # Verde
        status = "Altamente Confiável"
        confianca = "Alta"
    elif r2 > 0.50:
        cor = "\033[93m"  # Amarelo
        status = "Moderadamente Confiável"
        confianca = "Moderada"
    else:
        cor = "\033[91m"  # Vermelho
        status = "Baixa Confiabilidade"
        confianca = "Baixa"

    reset = "\033[0m"  # Resetar cor

    print(f"Desempenho do Modelo (R²): {r2:.4f}")
    print(f"Status: {cor}{status}{reset}")
    print(f"Confiança Estimada: {cor}{confianca}{reset}\n")

    # Input values ​​used
    print("Valores de Entrada Utilizados:")
    print("-" * 70)
    for prop, valor in valores_entrada.items():
        unidade_prop = UNIDADES[prop]
        print(f"  {prop:20} = {valor:10.4f} {unidade_prop}")
    print()


def perguntar_continuar():
    """Asks the user if they want to make another prediction"""
    while True:
        resposta = input(
            "Deseja fazer outra predição? (s/n): ").strip().lower()
        if resposta in ['s', 'sim', 'y', 'yes']:
            return True
        elif resposta in ['n', 'nao', 'não', 'no']:
            return False
        else:
            print("Digite 's' para sim ou 'n' para não.\n")


# PROGRAMA PRINCIPAL

def main():
    """Main function"""
    limpar_tela()
    exibir_header()

    # Check if the templates folder exists
    if not os.path.exists(MODELS_FOLDER):
        print(f"Erro: Pasta '{MODELS_FOLDER}' não encontrada!")
        print(
            f"   Certifique-se de que os modelos estão em '{MODELS_FOLDER}/'")
        input("\nPressione ENTER para sair...")
        return

    while True:
        # Get target property
        propriedade_alvo = obter_propriedade_alvo()

        print(f"\nVocê selecionou: \033[91m {propriedade_alvo} \033[0m")
        print("\033[93m" + "Carregando modelo..." + "\033[0m")

        # Load template
        modelo, scaler_x, scaler_y, colunas_entrada = carregar_modelo(
            propriedade_alvo)

        if modelo is None:
            input("\nPressione ENTER para tentar novamente...")
            limpar_tela()
            exibir_header()
            continue

        # Get input columns (properties that are not being predicted)
        colunas_entrada_esperadas = [
            col for col in colunas_entrada if col != propriedade_alvo]

        print("\033[32m" + "Modelo carregado com sucesso!\n" + "\033[0m")

        # Get input values
        valores_entrada = obter_valores_entrada(colunas_entrada_esperadas)

        # Make prediction
        print("\033[93m" + "\nProcessando predição..." + "\033[0m")
        predicao = fazer_predicao(
            modelo, scaler_x, scaler_y, valores_entrada, colunas_entrada_esperadas)

        if predicao is not None:
            # Get the R² of the model
            r2 = DESEMPENHO_MODELOS[propriedade_alvo]

            # Display result
            exibir_resultado(propriedade_alvo, predicao, valores_entrada, r2)

        # Ask if they wish to continue.
        if not perguntar_continuar():
            print("\nObrigado por usar a aplicação!")
            break

        limpar_tela()
        exibir_header()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAplicação interrompida pelo usuário.")
    except Exception as e:
        print(f"\nErro inesperado: {str(e)}")
        input("\nPressione ENTER para sair...")
