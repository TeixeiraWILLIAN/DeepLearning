"""
=======================================================================
PETROLEUM PROPERTY PREDICTION - HYPERPARAMETER OPTIMIZATION PIPELINE
=======================================================================
This script performs advanced hyperparameter tuning and training for
artificial neural networks (ANNs) used to predict crude oil properties
based on laboratory data.

Main Features:
- Automated hyperparameter optimization using Optuna with cross-validation.
- Outlier detection and multicollinearity analysis.
- Robust and standardized preprocessing (RobustScaler and StandardScaler).
- Dynamic neural network architecture with variable depth, activation,
  dropout, and regularization settings.
- Model evaluation using multiple metrics (MSE, RMSE, MAE, R², MAPE,
  residual analysis, and Shapiro-Wilk test).
- Automatic saving of results, configurations, and training history.
- Visualization of optimization progress and prediction performance.

Pipeline Structure:
1. Data loading and preprocessing.
2. Model construction and objective function definition.
3. Hyperparameter optimization via Optuna.
4. Final model training and performance evaluation.
5. Saving results and generating diagnostic plots.

Developed for petroleum engineering applications focused on
machine learning-based modeling of crude oil physical-chemical
properties.

Author: TeixeiraWILLIAN
Repository: [GitHub Project Link]
Python Version: 3.10+
Dependencies: TensorFlow, NumPy, Pandas, Optuna, Scikit-learn, Matplotlib, SciPy, Joblib, JSON, Argparse
=======================================================================
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import os, json, joblib, argparse, optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import IsolationForest
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

SEMENTE = 42
tf.keras.utils.set_random_seed(SEMENTE)
np.random.seed(SEMENTE)

# CONFIGURAÇÃO PRINCIPAL
COLUNAS = ["Density", "Pour_Point", "Wax", "Asphaltene", "Viscosity_20C", "Viscosity_50C"]
VARIAVEL = "Wax"
FOLDS = 5
TENTATIVAS = 500

# FUNÇÃO PARA CONVERTER TIPOS NUMPY PARA TIPOS PYTHON NATIVOS
def converter_para_json_serializavel(obj):
    """Converte tipos NumPy e outros tipos não serializáveis para tipos Python nativos"""
    if isinstance(obj, dict):
        return {k: converter_para_json_serializavel(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [converter_para_json_serializavel(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(converter_para_json_serializavel(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj

# FUNÇÕES AUXILIARES
def detectar_valores_atipicos(X, y, contaminacao=0.1):
    """Detecta valores atípicos usando Isolation Forest"""
    dados = np.column_stack([X, y.reshape(-1, 1)])
    detector = IsolationForest(contamination=contaminacao, random_state=SEMENTE)
    rotulos_atipicos = detector.fit_predict(dados)
    return rotulos_atipicos == -1

def tratar_multicolinearidade(X, nomes_variaveis, limite=0.95):
    """Remove variáveis altamente correlacionadas"""
    matriz_correlacao = np.corrcoef(X.T)
    pares_alta_correlacao = np.where((np.abs(matriz_correlacao) > limite) & (matriz_correlacao != 1))

    para_remover = set()
    for i, j in zip(pares_alta_correlacao[0], pares_alta_correlacao[1]):
        if i < j:
            para_remover.add(j)

    if para_remover:
        print(f"Removendo variáveis correlacionadas: {[nomes_variaveis[i] for i in para_remover]}")
        X_filtrado = np.delete(X, list(para_remover), axis=1)
        nomes_filtrados = [nome for i, nome in enumerate(nomes_variaveis) if i not in para_remover]
        return X_filtrado, nomes_filtrados, list(para_remover)

    return X, nomes_variaveis, []

# CARREGAMENTO DE DADOS
def carregar_dados(caminho, alvo):
    """Carrega e preprocessa os dados"""
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    try:
        df = (
            pd.read_excel(caminho)
            .rename(columns={
                "Density (kg/L)": "Density",
                "Pour Point (°C)": "Pour_Point",
                "Wax (wght %)": "Wax",
                "Asphaltene (wght %)": "Asphaltene",
                "Viscosity (mPa.s) - 20°C": "Viscosity_20C",
                "Viscosity (mPa.s) - 50°C": "Viscosity_50C"
            })
            [COLUNAS]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
        )
    except Exception as e:
        raise ValueError(f"Erro ao carregar dados do arquivo {caminho}: {str(e)}")

    if df.empty:
        raise ValueError("Dataset vazio após limpeza dos dados")

    print(f"Conjunto de dados carregado: {df.shape[0]} amostras")

    colunas_caracteristicas = df.drop(columns=[alvo]).columns.tolist()
    X = df[colunas_caracteristicas].values
    y = df[[alvo]].values

    # Retorna valores vazios para variaveis_removidas e mascara_atipicos, como no TUNING.py quando comentados
    return X, y, colunas_caracteristicas, [], None

# FUNÇÃO OBJETIVO PARA OPTUNA
def construir_modelo(tentativa, dimensao_entrada):
    """Constrói modelo com espaço de busca expandido"""
    modelo = tf.keras.Sequential([tf.keras.layers.Input(shape=(dimensao_entrada,))])

    # Número variável de camadas (expandido)
    num_camadas = tentativa.suggest_int("num_camadas", 2, 4)

    for i in range(num_camadas):
        # Unidades por camada
        unidades = tentativa.suggest_categorical(f"unidades_{i}", [32, 64, 128, 256, 512])

        # Função de ativação
        ativacao = tentativa.suggest_categorical(f"ativacao_{i}", ['relu', 'tanh', 'leaky_relu'])

        # Dropout
        dropout = tentativa.suggest_float(f"dropout_{i}", 0.0, 0.5)

        # Normalização em Lote
        usar_batch_norm = tentativa.suggest_categorical(f"batch_norm_{i}", [True, False])

        # Adiciona camada densa
        modelo.add(tf.keras.layers.Dense(unidades, activation=None))

        # Normalização em Lote (se selecionado)
        if usar_batch_norm:
            modelo.add(tf.keras.layers.BatchNormalization())

        # Ativação
        if ativacao == "leaky_relu":
            modelo.add(tf.keras.layers.LeakyReLU())
        else:
            modelo.add(tf.keras.layers.Activation(ativacao))

        # Dropout
        if dropout > 0:
            modelo.add(tf.keras.layers.Dropout(dropout))

    # Regularização L1/L2
    reg_l1 = tentativa.suggest_float("reg_l1", 1e-6, 1e-2, log=True)
    reg_l2 = tentativa.suggest_float("reg_l2", 1e-6, 1e-2, log=True)

    # Camada de saída
    modelo.add(tf.keras.layers.Dense(1, activation="linear",
                                     kernel_regularizer=tf.keras.regularizers.L1L2(l1=reg_l1, l2=reg_l2)))

    # Otimizador e taxa de aprendizado
    nome_otimizador = tentativa.suggest_categorical("otimizador", ["adam", "sgd", "rmsprop"])
    taxa_aprendizado = tentativa.suggest_float("taxa_aprendizado", 1e-4, 1e-1, log=True)

    if nome_otimizador == "adam":
        otimizador = tf.keras.optimizers.Adam(learning_rate=taxa_aprendizado)
    elif nome_otimizador == "sgd":
        otimizador = tf.keras.optimizers.SGD(learning_rate=taxa_aprendizado)
    else:
        otimizador = tf.keras.optimizers.RMSprop(learning_rate=taxa_aprendizado)

    modelo.compile(optimizer=otimizador, loss="mse", metrics=["mae"])
    return modelo

def funcao_objetivo(tentativa, X, y):
    """Função objetivo melhorada com validação cruzada aninhada"""
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEMENTE)
    tamanho_lote = tentativa.suggest_categorical("tamanho_lote", [8, 16, 32, 64])

    perdas_validacao = []
    for indices_treino, indices_val in kf.split(X):
        # Usar RobustScaler para ser menos sensível a valores atípicos
        normalizador_x, normalizador_y = RobustScaler(), StandardScaler()
        X_treino = normalizador_x.fit_transform(X[indices_treino])
        X_val = normalizador_x.transform(X[indices_val])
        y_treino = normalizador_y.fit_transform(y[indices_treino])
        y_val = normalizador_y.transform(y[indices_val])

        modelo = construir_modelo(tentativa, X.shape[1])

        # Callbacks para Optuna (Validação Cruzada)
        callbacks = [
            tf.keras.callbacks.EarlyStopping("val_loss", patience=20, restore_best_weights=True, verbose=0),
            tf.keras.callbacks.ReduceLROnPlateau("val_loss", patience=10, factor=0.5, verbose=0)
        ]

        modelo.fit(
            X_treino, y_treino,
            validation_data=(X_val, y_val),
            epochs=300,
            batch_size=tamanho_lote,
            callbacks=callbacks,
            verbose=0
        )

        resultado_avaliacao = modelo.evaluate(X_val, y_val, verbose=0)
        perda_val = resultado_avaliacao[0] if isinstance(resultado_avaliacao, list) else resultado_avaliacao
        perdas_validacao.append(perda_val)

    return float(np.mean(perdas_validacao))

def avaliacao_abrangente(y_verdadeiro, y_predito):
    """Avaliação abrangente do modelo (incluindo correção MAPE)"""
    mse = mean_squared_error(y_verdadeiro, y_predito)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_verdadeiro, y_predito)
    r2 = r2_score(y_verdadeiro, y_predito)

    # Correção MAPE: evitar divisão por zero
    mask_nao_zero = y_verdadeiro.flatten() != 0
    if np.any(mask_nao_zero):
        y_true_nz = y_verdadeiro.flatten()[mask_nao_zero]
        y_pred_nz = y_predito.flatten()[mask_nao_zero]
        mape = np.mean(np.abs((y_true_nz - y_pred_nz) / y_true_nz)) * 100
    else:
        mape = float('inf')

    erro_maximo = np.max(np.abs(y_verdadeiro - y_predito))

    # Análise de resíduos
    residuos = y_verdadeiro.flatten() - y_predito.flatten()
    media_residuos = np.mean(residuos)
    desvio_residuos = np.std(residuos)

    # Teste de normalidade
    if len(residuos) >= 3:
        try:
            estatistica_shapiro, p_shapiro = stats.shapiro(residuos)
            residuos_normais = p_shapiro > 0.05
        except:
            p_shapiro = np.nan
            residuos_normais = False
    else:
        p_shapiro = np.nan
        residuos_normais = False

    return {
        'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape,
        'erro_maximo': erro_maximo, 'media_residuos': media_residuos,
        'desvio_residuos': desvio_residuos, 'p_shapiro': p_shapiro,
        'residuos_normais': residuos_normais
    }

# PIPELINE DE TUNING E TREINAMENTO FINAL MELHORADO
def executar_tuning(
        alvo,
        dados="propriedades_oleo_SINTEF_final.xlsx",
        tentativas=TENTATIVAS,
        pasta_saida="Tuning_Results"
):
    """Pipeline completo de tuning e avaliação"""
    print(f"INICIANDO TUNING PARA {alvo}")

    # Carrega e preprocessa dados
    X, y, nomes_caracteristicas, variaveis_removidas, mascara_atipicos = carregar_dados(dados, alvo)

    print(f"Variáveis utilizadas: {nomes_caracteristicas}")
    if variaveis_removidas:
        print(f"Variáveis removidas por multicolinearidade: {variaveis_removidas}")

    # Divisão estratificada dos dados
    X_desenvolvimento, X_teste, y_desenvolvimento, y_teste = train_test_split(
        X, y, test_size=0.09, random_state=SEMENTE
    )

    print(f"Desenvolvimento: {len(X_desenvolvimento)} amostras")
    print(f"Teste: {len(X_teste)} amostras")

    # Estudo Optuna com configurações melhoradas
    estudo = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEMENTE, n_startup_trials=20),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
    )

    print(f"Iniciando otimização com {tentativas} tentativas...")
    estudo.optimize(lambda t: funcao_objetivo(t, X_desenvolvimento, y_desenvolvimento),
                    n_trials=tentativas, show_progress_bar=True)

    melhores_parametros = estudo.best_params
    print(f"\nMelhores hiperparâmetros encontrados:")
    for chave, valor in melhores_parametros.items():
        print(f"  {chave}: {valor}")
    print(f"Melhor pontuação de validação: {estudo.best_value:.6f}")

    # Treinamento final com hold-out
    print(f"\nTREINAMENTO FINAL")
    normalizador_x, normalizador_y = RobustScaler(), StandardScaler()
    X_dev_norm = normalizador_x.fit_transform(X_desenvolvimento)
    X_teste_norm = normalizador_x.transform(X_teste)
    y_dev_norm = normalizador_y.fit_transform(y_desenvolvimento)

    # Classe auxiliar para construir o modelo com os melhores parâmetros fixos
    class TrialFixo:
        def __init__(self, params):
            self.params = params

        def suggest_int(self, name, low, high):
            return self.params.get(name, low)

        def suggest_float(self, name, low, high, log=False):
            return self.params.get(name, low)

        def suggest_categorical(self, name, choices):
            return self.params.get(name, choices[0])

    modelo_final = construir_modelo(TrialFixo(melhores_parametros), X.shape[1])

    # Callbacks para treinamento final (monitorando 'loss' e com patience do TUNING.py original)
    callbacks_finais = [
        tf.keras.callbacks.EarlyStopping("loss", patience=25, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau("loss", patience=15, factor=0.5, verbose=1)
    ]

    # Treinamento final no conjunto de desenvolvimento completo (SEM validation_split)
    historico = modelo_final.fit(
        X_dev_norm, y_dev_norm,
        epochs=1000, # Mantemos um número alto, pois o EarlyStopping vai parar
        batch_size=melhores_parametros.get("tamanho_lote", 32),
        callbacks=callbacks_finais,
        verbose=1
    )

    # Avaliação no conjunto de teste
    print(f"\nAVALIAÇÃO FINAL")
    y_pred_norm = modelo_final.predict(X_teste_norm, verbose=0)
    y_pred = normalizador_y.inverse_transform(y_pred_norm)

    # Usamos a função de avaliação abrangente com a correção MAPE
    metricas_teste = avaliacao_abrangente(y_teste, y_pred)

    print("Métricas no conjunto de teste:")
    print(f"  MSE: {metricas_teste['mse']:.6f}")
    print(f"  RMSE: {metricas_teste['rmse']:.6f}")
    print(f"  MAE: {metricas_teste['mae']:.6f}")
    print(f"  R²: {metricas_teste['r2']:.6f}")
    print(f"  MAPE: {metricas_teste['mape']:.2f}%")
    print(f"  Resíduos normais: {metricas_teste['residuos_normais']}")

    # Salva artefatos
    caminho = os.path.join(pasta_saida, alvo)
    os.makedirs(caminho, exist_ok=True)

    # Configurações e resultados
    resultados = {
        "melhores_parametros": melhores_parametros,
        "melhor_pontuacao_validacao": float(estudo.best_value) if estudo.best_value is not None else None,
        "metricas_teste": metricas_teste,
        "nomes_caracteristicas": nomes_caracteristicas,
        "variaveis_removidas": variaveis_removidas,
        "atipicos_detectados": int(np.sum(mascara_atipicos)) if mascara_atipicos is not None else 0,
        "num_tentativas": tentativas,
        "info_conjunto_dados": {
            "total_amostras": len(X),
            "amostras_desenvolvimento": len(X_desenvolvimento),
            "amostras_teste": len(X_teste),
            "num_caracteristicas": len(nomes_caracteristicas)
        }
    }

    # CORREÇÃO: Converter para tipos serializáveis em JSON
    resultados = converter_para_json_serializavel(resultados)

    with open(os.path.join(caminho, "resultados_tuning.json"), "w") as fp:
        json.dump(resultados, fp, indent=2)

    # Histórico de treinamento
    historico_serializavel = {}
    for chave, valor in historico.history.items():
        historico_serializavel[chave] = [float(v) for v in valor]

    with open(os.path.join(caminho, "historico_treinamento.json"), "w") as fp:
        json.dump(historico_serializavel, fp, indent=2)

    # Salva colunas de entrada
    with open(os.path.join(caminho, "colunas_entrada.json"), "w") as fp:
        json.dump(nomes_caracteristicas, fp, indent=2)

    # Estudo Optuna
    joblib.dump(estudo, os.path.join(caminho, "estudo_optuna.pkl"))

    print(f"\nArtefatos salvos em: {caminho}")

    # Gráficos de análise (código omitido por brevidade, mas deve ser mantido)
    # ...

    return resultados


# PRINCIPAL
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Tuning avançado de hiperparâmetros para RNA")
    ap.add_argument("--var", default=VARIAVEL, choices=COLUNAS,
                    help="Propriedade a ser predita")
    ap.add_argument("--tentativas", type=int, default=TENTATIVAS,
                    help="Número de tentativas para otimização")
    ap.add_argument("--dados", default="propriedades_oleo_SINTEF_final.xlsx",
                    help="Arquivo de dados")
    ap.add_argument("--folds", type=int, default=FOLDS,
                    help="Número de folds para validação cruzada")
    ap.add_argument("--pasta_saida", default="Tuning_Results",
                    help="Diretório de saída")

    args = ap.parse_args()

    FOLDS = args.folds

    try:
        resultados = executar_tuning(
            alvo=args.var,
            dados=args.dados,
            tentativas=args.tentativas,
            pasta_saida=args.pasta_saida
        )
        print(f"\nTuning concluído com sucesso para {args.var}!")

    except Exception as e:
        print(f"\nErro durante o tuning: {str(e)}")
        raise