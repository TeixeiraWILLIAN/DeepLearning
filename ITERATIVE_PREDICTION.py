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

Author: TeixeiraWILLAIN
Repository: [GitHub Project Link]
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

# ============================================
# CONFIGURAÇÕES
# ============================================
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
    'Wax': -0.20,
    'Viscosity_50C': 0
}

MODELS_FOLDER = 'Results_model'

# ============================================
# FUNÇÕES
# ============================================


def limpar_tela():
    """Limpa a tela do console"""
    os.system('cls' if os.name == 'nt' else 'clear')


def exibir_header():
    """Exibe o cabeçalho da aplicação"""
    print("=" * 70)
    # Cores ANSI
    print(
        "\033[95m" + "PREDIÇÃO DE PROPRIEDADES DO PETRÓLEO BRUTO".center(70) + "\033[0m")
    print("=" * 70)
    print()


def exibir_menu_propriedades():
    """Exibe menu de seleção de propriedades"""
    print("Selecione a propriedade que deseja PREDIZER:\n")
    
    for i, prop in enumerate(PROPRIEDADES, 1):
        r2 = DESEMPENHO_MODELOS[prop]
        
        # Define cor e status
        if r2 > 0.90:
            cor = "\033[92m"  # Verde
            status = "Alta Confiabilidade"
        elif r2 > 0.50:
            cor = "\033[93m"  # Amarelo
            status = "Moderada Confiabilidade"
        else:
            cor = "\033[91m"  # Vermelho
            status = "Baixa Confiabilidade"
        
        reset = "\033[0m"
        print(f"  {i}. {prop:20} (R² = {r2:7.2f}) {cor}{status}{reset}")
    
    print()
    

def obter_propriedade_alvo():
    """Obtém a propriedade alvo do usuário"""
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
    """Carrega o modelo treinado para uma propriedade"""
    pasta = os.path.join(MODELS_FOLDER, propriedade)

    try:
        modelo_path = os.path.join(pasta, f'modelo_{propriedade}.keras')
        scaler_x_path = os.path.join(
            pasta, f'{propriedade}_normalizador_x.pkl')
        scaler_y_path = os.path.join(
            pasta, f'{propriedade}_normalizador_y.pkl')
        colunas_path = os.path.join(pasta, 'colunas_entrada.json')

        # Verificar se todos os arquivos existem
        if not all(os.path.exists(p) for p in [modelo_path, scaler_x_path, scaler_y_path, colunas_path]):
            print(
                f"Erro: Arquivos do modelo para '{propriedade}' não encontrados em {pasta}")
            return None, None, None, None

        # Carregar arquivos
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
    """Obtém os valores de entrada do usuário"""
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
    """Faz a predição"""
    try:
        # Preparar dados na ordem correta
        X = np.array([[valores_entrada[col] for col in colunas_esperadas]])

        # Normalizar
        X_norm = scaler_x.transform(X)

        # Predizer
        y_pred_norm = modelo.predict(X_norm, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_norm)

        return y_pred[0][0]
    except Exception as e:
        print(f"Erro ao fazer predição: {str(e)}")
        return None


def exibir_resultado(propriedade_alvo, predicao, valores_entrada, r2):
    """Exibe o resultado da predição"""
    print("\n" + "=" * 70)
    print("\033[95m" + "RESULTADO DA PREDIÇÃO".center(70) + "\033[0m")
    print("=" * 70)
    print()

    # Resultado principal
    unidade = UNIDADES[propriedade_alvo]
    print(f"Propriedade a Predizer: \033[91m {propriedade_alvo} \033[0m")
    print(f"Valor Predito: \033[91m{predicao:.4f} {unidade} \033[0m")
    print()

    # Desempenho do modelo
    if r2 > 0.90:
        status = "Altamente Confiável"
        confianca = "Alta"
    elif r2 > 0.50:
        status = "Moderadamente Confiável"
        confianca = "Moderada"
    else:
        status = "Baixa Confiabilidade"
        confianca = "Baixa"

    print(f"Desempenho do Modelo (R²): {r2:.4f}")
    print(f"Status: {status}")
    print(f"Confiança Estimada: {confianca}")
    print()

    # Valores de entrada utilizados
    print("Valores de Entrada Utilizados:")
    print("-" * 70)
    for prop, valor in valores_entrada.items():
        unidade_prop = UNIDADES[prop]
        print(f"  {prop:20} = {valor:10.4f} {unidade_prop}")
    print()


def perguntar_continuar():
    """Pergunta se o usuário deseja fazer outra predição"""
    while True:
        resposta = input(
            "Deseja fazer outra predição? (s/n): ").strip().lower()
        if resposta in ['s', 'sim', 'y', 'yes']:
            return True
        elif resposta in ['n', 'nao', 'não', 'no']:
            return False
        else:
            print("Digite 's' para sim ou 'n' para não.\n")

# ============================================
# PROGRAMA PRINCIPAL
# ============================================


def main():
    """Função principal"""
    limpar_tela()
    exibir_header()

    # Verificar se a pasta de modelos existe
    if not os.path.exists(MODELS_FOLDER):
        print(f"Erro: Pasta '{MODELS_FOLDER}' não encontrada!")
        print(
            f"   Certifique-se de que os modelos estão em '{MODELS_FOLDER}/'")
        input("\nPressione ENTER para sair...")
        return

    while True:
        # Obter propriedade alvo
        propriedade_alvo = obter_propriedade_alvo()

        print(f"\nVocê selecionou: \033[91m {propriedade_alvo} \033[0m")
        print("\033[93m" + "Carregando modelo..." + "\033[0m")

        # Carregar modelo
        modelo, scaler_x, scaler_y, colunas_entrada = carregar_modelo(
            propriedade_alvo)

        if modelo is None:
            input("\nPressione ENTER para tentar novamente...")
            limpar_tela()
            exibir_header()
            continue

        # Obter colunas de entrada (propriedades que não estão sendo preditas)
        colunas_entrada_esperadas = [
            col for col in colunas_entrada if col != propriedade_alvo]

        print("\033[32m" + "Modelo carregado com sucesso!\n" + "\033[0m")

        # Obter valores de entrada
        valores_entrada = obter_valores_entrada(colunas_entrada_esperadas)

        # Fazer predição
        print("\033[93m" + "\nProcessando predição..." + "\033[0m")
        predicao = fazer_predicao(
            modelo, scaler_x, scaler_y, valores_entrada, colunas_entrada_esperadas)

        if predicao is not None:
            # Obter R² do modelo
            r2 = DESEMPENHO_MODELOS[propriedade_alvo]

            # Exibir resultado
            exibir_resultado(propriedade_alvo, predicao, valores_entrada, r2)

        # Perguntar se deseja continuar
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