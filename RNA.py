"""
=======================================================================
PETROLEUM PROPERTY PREDICTION - TRAINING AND FINAL MODEL EXECUTION
=======================================================================
This script loads predefined hyperparameter configurations and performs
the final training, evaluation, and saving of neural network models
for crude oil property prediction.

Main Features:
- Loads optimized hyperparameters for each property (Density, Wax, etc.).
- Performs K-Fold cross-validation to estimate model stability.
- Executes final model training with early stopping and learning rate scheduling.
- Computes a comprehensive set of performance metrics (MSE, RMSE, MAE, R², MAPE,
  residual distribution, and Shapiro Wilk test).
- Saves all artifacts including:
  * Trained model (.keras)
  * Scalers (joblib)
  * Metrics and configuration files (.json)
  * Diagnostic plots (loss curve and prediction scatter)
- Modular functions for data loading, model construction, and evaluation.

Pipeline Overview:
1. Load dataset and select target property.
2. Retrieve pre-optimized hyperparameters.
3. Perform cross-validation to determine optimal epoch count.
4. Train the final model on development data.
5. Evaluate on test set and save all results.

Developed for petroleum engineering applications focusing on
machine learning-based modeling of crude oil physical-chemical
properties.

Author: Willian Teixeira
Repository: https://github.com/TeixeiraWILLIAN/DeepLearning
Python Version: 3.10+
Dependencies: TensorFlow, NumPy, Pandas, Scikit-learn, Matplotlib, SciPy, Joblib, JSON, Argparse
=======================================================================
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import json
import joblib
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

SEMENTE = 42
tf.keras.utils.set_random_seed(SEMENTE)
np.random.seed(SEMENTE)

# DEFAULT SETTINGS (by property)
# ============================================
configuracoes_predefinidas = {
    'Density': dict(
        otimizador='adam',
        taxa_aprendizado=0.0370945205376818,
        tamanho_lote=8,
        camadas=[(512, 'tanh', 0.00775117625775313),
                 (128, 'relu', 0.033273865106612885)],
        reg_l1=6.3560690619094156e-06,
        reg_l2=1.2399557004054873e-05
    ),

    'Asphaltene': dict(
        otimizador='adam',
        taxa_aprendizado=0.002821109346785365,
        tamanho_lote=16,
        camadas=[(128, 'leakyrelu', 0.28221415209509226),
                 (256, 'relu', 0.1374385906580356)],
        reg_l1=4.8497904001971764e-05,
        reg_l2=0.00012448401445882366
    ),

    'Pour_Point': dict(
        otimizador='adam',
        taxa_aprendizado=0.0009368323927352315,
        tamanho_lote=32,
        camadas=[(512, 'relu', 0.03332736826621852),
                 (256, 'tanh', 0.3823054650868597)],
        reg_l1=1.1732750655683321e-06,
        reg_l2=6.187908737771573e-05
    ),

    'Wax': dict(
        otimizador='rmsprop',
        taxa_aprendizado=0.015218557038587922,
        tamanho_lote=32,
        camadas=[(128, 'tanh', 0.31632860109975486),
                 (512, 'leakyrelu', 0.08158012055887812)],
        reg_l1=5.223282240031189e-05,
        reg_l2=0.00011916931922503524
    ),

    'Viscosity_20C': dict(
        otimizador='rmsprop',
        taxa_aprendizado=0.006816805963431236,
        tamanho_lote=32,
        camadas=[(512, 'tanh', 0.23097950960028463),
                 (512, 'relu', 0.42916996858725887)],
        reg_l1=2.236474408206667e-06,
        reg_l2=1.8100436625872475e-06
    ),

    'Viscosity_50C': dict(
        otimizador='rmsprop',
        taxa_aprendizado=0.04316747384224007,
        tamanho_lote=32,
        camadas=[(64, 'leakyrelu', 0.1170434889383446),
                 (32, 'relu', 0.3912906794861852)],
        reg_l1=7.57472071885502e-06,
        reg_l2=1.9660467736409024e-05
    )
}

# MAIN CONFIGURATION
COLUNAS = ["Density", "Pour_Point", "Wax",
           "Asphaltene", "Viscosity_20C", "Viscosity_50C"]
VARIAVEL = "Pour_Point"

# Utilities


def converter_para_json_serializavel(obj):
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
    elif pd.isna(obj):
        return None
    else:
        return obj


def detectar_valores_atipicos(X, y, contaminacao=0.1):
    # Not used in this script, but kept for signature compatibility if needed.
    return np.zeros(len(X), dtype=bool)


def tratar_multicolinearidade(X, nomes_variaveis, limite=0.95):
    # Not used in this script, but kept for signature compatibility if needed.
    return X, nomes_variaveis, []


def carregar_dados(caminho, alvo):
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
        raise ValueError(
            f"Erro ao carregar dados do arquivo {caminho}: {str(e)}")

    if df.empty:
        raise ValueError("Dataset vazio após limpeza dos dados")

    print(f"Conjunto de dados carregado: {df.shape[0]} amostras")

    colunas_caracteristicas = df.drop(columns=[alvo]).columns.tolist()
    X = df[colunas_caracteristicas].values
    y = df[[alvo]].values

    # Returns empty values ​​for removed_variables and atypical_masks, as in TUNING.py when commented out.
    return X, y, colunas_caracteristicas, [], None

# Gets the configuration for the target variable


def obter_configuracao(alvo):
    """
    Retrieves the target variable configuration in the new format (layers=[(...), ...]).
    """
    if alvo in configuracoes_predefinidas:
        return configuracoes_predefinidas[alvo]
    else:
        raise ValueError(
            f"Configuração predefinida não encontrada para a variável alvo: {alvo}")

# Model builder with fixed parameters


def construir_modelo_otimizado(params, dimensao_entrada):
    """
    Builds the neural network model based on the provided parameters.
    Expects the format: params["layers"] = [(units, ativ, dropout), ...]
    """
    modelo = tf.keras.Sequential(
        [tf.keras.layers.Input(shape=(dimensao_entrada,))])

    for camada in params["camadas"]:
        # Get the first 3 elements (units, activation, dropout)
        unidades, ativacao, dropout = camada[:3]
        # Checks if batch_norm is present
        usar_batch_norm = camada[3] if len(camada) > 3 else False

        modelo.add(tf.keras.layers.Dense(unidades, activation=None))
        if usar_batch_norm:
            modelo.add(tf.keras.layers.BatchNormalization())

        if ativacao.lower() == "leakyrelu" or ativacao.lower() == "leaky_relu":
            modelo.add(tf.keras.layers.LeakyReLU())
        else:
            modelo.add(tf.keras.layers.Activation(ativacao))

        if dropout and dropout > 0:
            modelo.add(tf.keras.layers.Dropout(dropout))

    reg_l1 = params.get("reg_l1", 0.0)
    reg_l2 = params.get("reg_l2", 0.0)

    modelo.add(
        tf.keras.layers.Dense(
            1,
            activation="linear",
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=reg_l1, l2=reg_l2)
        )
    )

    nome_otimizador = params.get("otimizador", "adam").lower()
    taxa_aprendizado = params.get("taxa_aprendizado", 1e-3)

    if nome_otimizador == "adam":
        otimizador = tf.keras.optimizers.Adam(learning_rate=taxa_aprendizado)
    elif nome_otimizador == "sgd":
        otimizador = tf.keras.optimizers.SGD(learning_rate=taxa_aprendizado)
    else:
        otimizador = tf.keras.optimizers.RMSprop(
            learning_rate=taxa_aprendizado)

    modelo.compile(optimizer=otimizador, loss="mse", metrics=["mae"])
    return modelo


def avaliacao_abrangente(y_verdadeiro, y_predito):
    mse = mean_squared_error(y_verdadeiro, y_predito)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_verdadeiro, y_predito)
    r2 = r2_score(y_verdadeiro, y_predito)

    mask_nao_zero = y_verdadeiro.flatten() != 0
    if np.any(mask_nao_zero):
        y_true_nz = y_verdadeiro.flatten()[mask_nao_zero]
        y_pred_nz = y_predito.flatten()[mask_nao_zero]
        mape = np.mean(np.abs((y_true_nz - y_pred_nz) / y_true_nz)) * 100
    else:
        mape = float("inf")

    erro_maximo = np.max(np.abs(y_verdadeiro - y_predito))

    residuos = y_verdadeiro.flatten() - y_predito.flatten()
    media_residuos = np.mean(residuos)
    desvio_residuos = np.std(residuos)

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
        "mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape,
        "erro_maximo": erro_maximo, "media_residuos": media_residuos,
        "desvio_residuos": desvio_residuos, "p_shapiro": p_shapiro,
        "residuos_normais": residuos_normais
    }


def executar_modelo_otimizado(
        alvo,
        params,
        dados="oil_prop_database.xlsx",
        pasta_saida="Results_model"
):
    print(f"INICIANDO TREINAMENTO PARA {alvo} COM HIPERPARÂMETROS OTIMIZADOS")
    # load_data returns X, y, characteristic_names, removed_variables, atypical_masks
    X, y, nomes_caracteristicas, _, _ = carregar_dados(dados, alvo)
    print(f"Variáveis utilizadas: {nomes_caracteristicas}")

    X_desenvolvimento, X_teste, y_desenvolvimento, y_teste = train_test_split(
        X, y, test_size=0.09, random_state=SEMENTE
    )
    print(f"Desenvolvimento: {len(X_desenvolvimento)} amostras")
    print(f"Teste: {len(X_teste)} amostras")
    # print(f"Valores do conjunto teste usado: {y_teste}")

    # K-FOLD CROSS-VALIDATION
    print(f"\nVALIDAÇÃO CRUZADA K-FOLD (k=5)")
    kf = KFold(n_splits=5, shuffle=True, random_state=SEMENTE)
    epocas_todas = []

    for fold, (treino, validacao) in enumerate(kf.split(X_desenvolvimento), 1):
        print(f"Processando fold {fold}/5")

        normalizador_X, normalizador_y = RobustScaler(), StandardScaler()
        X_treino_norm = normalizador_X.fit_transform(X_desenvolvimento[treino])
        X_val_norm = normalizador_X.transform(X_desenvolvimento[validacao])
        y_treino_norm = normalizador_y.fit_transform(y_desenvolvimento[treino])
        y_val_norm = normalizador_y.transform(y_desenvolvimento[validacao])

        modelo_fold = construir_modelo_otimizado(
            params, X_desenvolvimento.shape[1])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                'val_loss', patience=20, restore_best_weights=True, verbose=0),
            tf.keras.callbacks.ReduceLROnPlateau(
                'val_loss', patience=10, factor=0.5, verbose=0)
        ]

        historico_fold = modelo_fold.fit(
            X_treino_norm, y_treino_norm,
            validation_data=(X_val_norm, y_val_norm),
            epochs=1000,
            batch_size=params.get('tamanho_lote', 32),
            callbacks=callbacks,
            verbose=0
        )
        epocas_todas.append(len(historico_fold.history['loss']))

    # FINAL TRAINING IN THE DEVELOPMENT SET
    print(f"\nTREINAMENTO FINAL")
    epocas_media = int(np.mean(epocas_todas))
    print(f"Épocas médias dos folds: {epocas_media}")

    normalizador_x, normalizador_y = RobustScaler(), StandardScaler()
    X_dev_norm = normalizador_x.fit_transform(X_desenvolvimento)
    y_dev_norm = normalizador_y.fit_transform(y_desenvolvimento)

    modelo_final = construir_modelo_otimizado(
        params, X_desenvolvimento.shape[1])

    callbacks_finais = [
        tf.keras.callbacks.EarlyStopping(
            'loss', patience=15, restore_best_weights=True, verbose=0),
        tf.keras.callbacks.ReduceLROnPlateau(
            'loss', patience=8, factor=0.5, verbose=0)
    ]

    historico = modelo_final.fit(
        X_dev_norm, y_dev_norm,
        epochs=epocas_media + 50,
        batch_size=params.get('tamanho_lote', 32),
        callbacks=callbacks_finais,
        verbose=1
    )

    print(f"\nAVALIAÇÃO FINAL")
    X_teste_norm = normalizador_x.transform(X_teste)
    y_pred_norm = modelo_final.predict(X_teste_norm, verbose=0)
    y_pred = normalizador_y.inverse_transform(y_pred_norm)

    metricas_teste = avaliacao_abrangente(y_teste, y_pred)

    print("Métricas no conjunto de teste:")
    print(f"  MSE: {metricas_teste['mse']:.6f}")
    print(f"  RMSE: {metricas_teste['rmse']:.6f}")
    print(f"  MAE: {metricas_teste['mae']:.6f}")
    print(f"  R²: {metricas_teste['r2']:.6f}")
    print(f"  MAPE: {metricas_teste['mape']:.2f}%")
    print(f"  Resíduos normais: {metricas_teste['residuos_normais']}")

    caminho = os.path.join(pasta_saida, alvo)
    os.makedirs(caminho, exist_ok=True)

    resultados = {
        "hiperparametros_utilizados": converter_para_json_serializavel(params),
        "metricas_teste": converter_para_json_serializavel(metricas_teste),
        "nomes_caracteristicas": nomes_caracteristicas,
        "info_conjunto_dados": {
            "total_amostras": len(X),
            "amostras_treinamento": len(X_desenvolvimento),
            "amostras_teste": len(X_teste),
            "num_caracteristicas": len(nomes_caracteristicas)
        }
    }

    with open(os.path.join(caminho, f"resultados_{alvo}_otimizado.json"), "w") as fp:
        json.dump(resultados, fp, indent=2)

    historico_serializavel = {
        chave: [float(v) for v in valor] for chave, valor in historico.history.items()}
    with open(os.path.join(caminho, f"{alvo}_historico_treinamento.json"), "w") as fp:
        json.dump(historico_serializavel, fp, indent=2)

    modelo_final.save(os.path.join(caminho, f"modelo_{alvo}.keras"))
    joblib.dump(normalizador_x, os.path.join(
        caminho, f"{alvo}_normalizador_x.pkl"))
    joblib.dump(normalizador_y, os.path.join(
        caminho, f"{alvo}_normalizador_y.pkl"))

    print(f"\nArtefatos salvos em: {caminho}")

    # Final loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(historico.history['loss'])
    plt.yscale('log')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.title(f'{alvo} – Treinamento final')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(caminho, 'perda_final.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Predictions vs Observations
    plt.figure(figsize=(10, 8))
    plt.scatter(y_teste, y_pred, alpha=0.7, color='blue')
    valor_min = min(np.min(y_teste), np.min(y_pred))
    valor_max = max(np.max(y_teste), np.max(y_pred))
    plt.plot([valor_min, valor_max], [valor_min, valor_max], 'r--', lw=2)
    plt.xlabel('Valores Observados')
    plt.ylabel('Valores Preditos')
    plt.title(f'{alvo} - Predições vs Observações (Teste)')
    plt.grid(True, alpha=0.3)
    plt.text(0.05, 0.95, f'R² = {metricas_teste["r2"]:.4f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(caminho, 'predicoes_vs_observacoes.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Save input columns
    with open(os.path.join(caminho, "colunas_entrada.json"), "w") as f:
        json.dump(nomes_caracteristicas, f, indent=2)

    # Save test metrics
    with open(os.path.join(caminho, "metricas_teste.json"), "w") as f:
        json.dump(converter_para_json_serializavel(
            metricas_teste), f, indent=2)

    # Save hyperparameters with additional information
    dados_configuracao = {
        'configuracao_predefinida': params,
        'colunas_entrada_original': nomes_caracteristicas,
        'colunas_entrada_filtradas': nomes_caracteristicas,
        'variaveis_removidas': [],
        'metricas_teste': metricas_teste,
        'resumo_metricas_cv': {
            'mse_media': float(metricas_teste['mse']),
            'rmse_media': float(metricas_teste['rmse']),
            'r2_media': float(metricas_teste['r2'])
        }
    }
    with open(os.path.join(caminho, 'hiperparametros.json'), 'w') as f:
        json.dump(converter_para_json_serializavel(
            dados_configuracao), f, indent=2)

    return metricas_teste


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Executa o modelo de rede neural com hiperparâmetros específicos por propriedade.")
    parser.add_argument("--target", type=str, default=VARIAVEL,
                        help="Variável alvo: 'Density', 'Pour_Point', 'Wax', 'Asphaltene', 'Viscosity_20C', 'Viscosity_50C'.")
    parser.add_argument("--data_path", type=str, default="oil_prop_database.xlsx",
                        help="Caminho para o arquivo de dados de entrada.")
    parser.add_argument("--output_dir", type=str, default="Results_model",
                        help="Diretório para salvar os resultados e artefatos do modelo.")

    args = parser.parse_args()

    # Automatically selects the target property setting
    params_escolhidos = obter_configuracao(args.target)
    executar_modelo_otimizado(
        args.target, params_escolhidos, args.data_path, args.output_dir)
