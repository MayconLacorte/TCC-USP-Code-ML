# %%

# -*- coding: utf-8 -*-
"""
@author: Maycon Lacorte - USP
"""

# %% Construção do df_final (ENEM 2022 + IPEA MM3 de 2022 + FUNDEB MM3 de 2022)

import pandas as pd
import numpy as np
import os

# ============================================================
# 0. FUNÇÃO AUXILIAR PARA PADRONIZAR CÓDIGO IBGE
# ============================================================

def padronizar_cod(x):
    try:
        return str(int(float(x))).zfill(7)
    except:
        return None


# ============================================================
# 1. IPEA — CARREGAR E TRATAR
# ============================================================

def carregar_ipea(caminho_arquivo, nome_variavel):
    df = pd.read_csv(
        caminho_arquivo,
        sep=",",
        encoding="latin1"
    )
    
    df.columns = ["periodo", "cod_municipio", nome_variavel]
    df["cod_municipio"] = df["cod_municipio"].astype(str).str.zfill(7)
    df["ano"] = pd.to_datetime(df["periodo"], errors="coerce").dt.year
    df = df.drop(columns=["periodo"])
    return df

dir_ipea = r"Y:\OneDrive\Documentos\Pós Graduação\USP\Data Science\TCC\Dados\IPEA"

arquivos_ipea = {
    "tx_homic_jovens":   "Taxa de Homicídios de Jovens_municípios.csv",
    "tx_homic_mulheres": "Taxa de Homicídios Mulheres_municípios.csv",
    "tx_homic_homens":   "Taxa de Homicídios Homens_municípios.csv",
    "tx_armas_fogo":     "Taxa de Óbitos por Armas de Fogo_municípios.csv"
}

dfs_ipea = []
for nome_var, nome_arquivo in arquivos_ipea.items():
    caminho = os.path.join(dir_ipea, nome_arquivo)
    df_temp = carregar_ipea(caminho, nome_var)
    dfs_ipea.append(df_temp)

df_ipea = dfs_ipea[0]
for df_temp in dfs_ipea[1:]:
    df_ipea = df_ipea.merge(df_temp, on=["cod_municipio", "ano"], how="outer")

df_ipea = df_ipea.sort_values(["cod_municipio", "ano"]).reset_index(drop=True)

taxas = ["tx_homic_jovens", "tx_homic_mulheres", "tx_homic_homens", "tx_armas_fogo"]

# MÉDIA MÓVEL 3 ANOS
df_ipea_mm = df_ipea.copy()

for col in taxas:
    df_ipea_mm[col + "_mm3"] = (
        df_ipea_mm
        .groupby("cod_municipio")[col]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

df_ipea_mm["ano_max"] = df_ipea_mm.groupby("cod_municipio")["ano"].transform("max")
df_ipea_last = df_ipea_mm[df_ipea_mm["ano"] == df_ipea_mm["ano_max"]].copy()

cols_ipea_final = ["cod_municipio"] + [c + "_mm3" for c in taxas]
df_ipea_last = df_ipea_last[cols_ipea_final].drop_duplicates("cod_municipio")


# ============================================================
# 2. FUNDEB — CARREGAR E TRATAR (MM3 2020–2022)
# ============================================================

dir_fundeb = r"Y:\OneDrive\Documentos\Pós Graduação\USP\Data Science\TCC\Dados\FUNDEB"
caminho_fundeb = os.path.join(dir_fundeb, "FUNDEB 2019-2025 - Transf muncipio.csv")

df_fundeb = pd.read_csv(
    caminho_fundeb,
    sep=";",
    encoding="latin1"
)

df_fundeb.columns = df_fundeb.columns.str.strip()

df_fundeb = df_fundeb.rename(columns={
    "Código IBGE": "cod_municipio",
    "Ano": "ano",
    "Valor Consolidado": "valor_fundeb"
})

df_fundeb["cod_municipio"] = df_fundeb["cod_municipio"].astype(str).str.zfill(7)

df_fundeb["valor_fundeb"] = (
    df_fundeb["valor_fundeb"]
    .astype(str)
    .str.replace("R$", "", regex=False)
    .str.replace(" ", "", regex=False)
    .str.replace(".", "", regex=False)
    .str.replace(",", ".", regex=False)
    .astype(float)
)

# USAR APENAS 2020, 2021, 2022
df_fundeb = df_fundeb[df_fundeb["ano"].isin([2020, 2021, 2022])]

df_fundeb_mm = (
    df_fundeb.groupby("cod_municipio")["valor_fundeb"]
    .mean()
    .reset_index()
    .rename(columns={"valor_fundeb": "fundeb_mm3"})
)

# ============================================================
# 3. ENEM 2022 — CARREGAR E TRATAR (COM IDH SINTÉTICO)
# ============================================================

def carregar_tratar_enem_2022(caminho_arquivo):
    
    colunas_enem = [
        "NU_ANO",
        "TP_ESCOLA",
        "CO_MUNICIPIO_ESC",
        "NO_MUNICIPIO_ESC",
        "CO_UF_ESC",
        "SG_UF_ESC",
        "CO_MUNICIPIO_PROVA",
        "NO_MUNICIPIO_PROVA",
        "CO_UF_PROVA",
        "SG_UF_PROVA",
        "NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC",
        "NU_NOTA_MT", "NU_NOTA_REDACAO",
        "Q001", "Q002", "Q005", "Q006" # Incluídas Q001, Q002, Q005
    ]

    df = pd.read_csv(
        caminho_arquivo,
        sep=";",
        encoding="latin1",
        usecols=colunas_enem,
        low_memory=False
    )

    df = df[df["NU_ANO"] == 2022]

    # ============================================================
    # FILTRO METODOLÓGICO — TP_ESCOLA + RENDA (Q006)
    # ============================================================

    renda_baixa = ["A", "B", "C", "D", "E"]

    df = df[
        (df["TP_ESCOLA"] == 2) |                                 # Pública
        ((df["TP_ESCOLA"] == 1) & (df["Q006"].isin(renda_baixa))) # Não respondeu + renda baixa
    ]

    # ============================================================
    # CALCULAR PERFORMANCE INDIVIDUAL
    # ============================================================

    df["performance"] = df[
        ["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC",
         "NU_NOTA_MT", "NU_NOTA_REDACAO"]
    ].mean(axis=1)

    df = df[df["performance"].notna()]

    # ============================================================
    # CÁLCULO DO IDH_ENEM_RENDA (Q005 + Q006)
    # ============================================================
    # Mapeamento do Ponto Médio da Renda Familiar (SM 2022 = R$ 1.212,00)
    mapa_renda = {
        'A': 0.0, 'B': 606.0, 'C': 1515.0, 'D': 2121.0, 'E': 2727.0,
        'F': 3333.0, 'G': 4242.0, 'H': 5454.0, 'I': 6666.0, 'J': 7878.0,
        'K': 9090.0, 'L': 10302.0, 'M': 11514.0, 'N': 13332.0, 'O': 16362.0,
        'P': 21210.0, 'Q': 24240.0
    }
    
    df["renda_bruta"] = df["Q006"].map(mapa_renda)
    df["Q005"] = pd.to_numeric(df["Q005"], errors="coerce")
    
    # Renda Per Capita
    df["renda_pc"] = df["renda_bruta"] / df["Q005"]
    
    # Normalização Logarítmica (Padrão PNUD: max teto R$ 25.000)
    df["IDH_ENEM_Renda"] = np.log1p(df["renda_pc"]) / np.log1p(25000)
    df["IDH_ENEM_Renda"] = df["IDH_ENEM_Renda"].clip(0, 1)

    # ============================================================
    # CÁLCULO DO IDH_ENEM_EDU (Q001 + Q002)
    # ============================================================
    # Mapeamento para Anos de Estudo
    mapa_escolaridade = {
        'A': 0.0, 'B': 2.0, 'C': 5.0, 'D': 9.0,
        'E': 12.0, 'F': 16.0, 'G': 18.0, 'H': np.nan
    }
    
    df["anos_pai"] = df["Q001"].map(mapa_escolaridade)
    df["anos_mae"] = df["Q002"].map(mapa_escolaridade)
    
    # Média dos Pais (Se um for 'H'/desconhecido, ignora via axis=1)
    df["anos_pais_med"] = df[["anos_pai", "anos_mae"]].mean(axis=1)
    
    # Normalização do Indicador de Educação (Anos Médios / 18 anos)
    df["IDH_ENEM_Edu"] = df["anos_pais_med"] / 18.0
    df["IDH_ENEM_Edu"] = df["IDH_ENEM_Edu"].clip(0, 1)

    # ============================================================
    # PADRONIZAR CÓDIGOS IBGE
    # ============================================================

    df["CO_MUNICIPIO_ESC"] = df["CO_MUNICIPIO_ESC"].apply(padronizar_cod)
    df["CO_MUNICIPIO_PROVA"] = df["CO_MUNICIPIO_PROVA"].apply(padronizar_cod)

    # ============================================================
    # USAR MUNICÍPIO DA PROVA COMO PROXY QUANDO ESCOLA ESTIVER AUSENTE
    # ============================================================

    df["cod_municipio"] = df["CO_MUNICIPIO_ESC"].fillna(df["CO_MUNICIPIO_PROVA"])
    df["cod_municipio"] = df["cod_municipio"].astype(str).str.zfill(7)

    # Retornar dados individuais com os novos indicadores
    return df[["cod_municipio", "performance", "IDH_ENEM_Renda", "IDH_ENEM_Edu"]]


enem_2022_path = r"Y:\OneDrive\Documentos\Pós Graduação\USP\Data Science\TCC\Dados\ENEM\microdados_enem_2022\DADOS\MICRODADOS_ENEM_2022.csv"
df_enem = carregar_tratar_enem_2022(enem_2022_path)


# ============================================================
# 4. INTEGRAR ENEM 2022 + IPEA + FUNDEB → df_final
# ============================================================

df_final = (
    df_enem
    .merge(df_ipea_last, on="cod_municipio", how="left")
    .merge(df_fundeb_mm, on="cod_municipio", how="left")
)

print("df_final shape:", df_final.shape)
print(df_final.head())


# %% Modelo XGBOOST

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import numpy as np

# ============================================================
# 1. PREPARAR BASE PARA O MODELO
# ============================================================



# 1) Criar transformação log1p para fundeb
df_final["fundeb_mm3_log"] = np.log1p(df_final["fundeb_mm3"])

# 2) Escolher as features (INCLUINDO IDH_ENEM_Renda e IDH_ENEM_Edu)
features = [
    "tx_homic_jovens_mm3",
    "tx_homic_mulheres_mm3",
    "tx_homic_homens_mm3",
    "tx_armas_fogo_mm3",
    "fundeb_mm3_log",
    "IDH_ENEM_Renda",
    "IDH_ENEM_Edu"
]

# 1. Separação dos atributos e do alvo
X = df_final[features]
y = df_final["performance"]

# ============================================================
# 2. TREINO / TESTE (80% / 20%)
# ============================================================


# 2. Divisão entre treino e teste ANTES de qualquer imputação
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
)


# ============================================================
# 3. Evitando Data leakage
# ============================================================

# 3. Imputação baseada EXCLUSIVAMENTE nas estatísticas do grupo de treino
for col in features:
    if X_train[col].isna().any():
        # Calcula a mediana APENAS no treino
        median_val = X_train[col].median()
        
        # Preenche treino e teste com a mediana calculada no treino
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

# ============================================================
# 2. TREINO / TESTE (80% / 20%)
# ============================================================

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.20, random_state=42
# )

# ============================================================
# 4. PADRONIZAÇÃO
# ============================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convertendo de volta para DataFrame para manter os nomes das colunas no XGBoost/SHAP
X_train_scaled = pd.DataFrame(X_train_scaled, columns=features)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=features)

# ============================================================
# 5. TREINAR MODELO XGBOOST
# ============================================================

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# ============================================================
# 6. AVALIAÇÃO DO MODELO
# ============================================================

y_pred = model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("========== MÉTRICAS DO MODELO ==========")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")

# ============================================================
# 7. IMPORTÂNCIA DAS VARIÁVEIS
# ============================================================

importances = pd.DataFrame({
    "variavel": X.columns,
    "importancia": model.feature_importances_
}).sort_values(by="importancia", ascending=False)

print("\n========== IMPORTÂNCIA DAS VARIÁVEIS ==========")
print(importances)


# %% Análise SHAP (Summary, Bar, Dependence e Waterfall)

import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# 1. AMOSTRAGEM PARA DESEMPENHO E MEMÓRIA
# ============================================================
# Seleciona 2.000 observações aleatórias de X_test para processamento rápido
N_SAMPLES = 2000

if len(X_test_scaled) > N_SAMPLES:
    X_test_sample = X_test.sample(n=N_SAMPLES, random_state=42).reset_index(drop=True)
    X_test_scaled_sample = X_test_scaled.loc[X_test_sample.index].reset_index(drop=True)
else:
    X_test_sample = X_test.reset_index(drop=True)
    X_test_scaled_sample = X_test_scaled.reset_index(drop=True)

# ============================================================
# 2. INICIALIZAR O EXPLAINER E CALCULAR OS SHAP VALUES
# ============================================================

explainer = shap.TreeExplainer(model)

# Objeto Explanation (nativo para shap.plots.*)
shap_explanation = explainer(X_test_scaled_sample)

# Array numpy para métodos legados (shap.summary_plot, shap.dependence_plot)
shap_values_array = shap_explanation.values

# ============================================================
# 3. SHAP SUMMARY PLOT (BEESWARM PLOT)
# ============================================================
print("===== SHAP SUMMARY PLOT (BEESWARM) =====")
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values_array, 
    X_test_scaled_sample, 
    feature_names=features, 
    show=False
)
plt.title("SHAP Summary Plot — Impacto e Direção das Variáveis na Nota", fontsize=12, pad=15)
plt.tight_layout()
plt.show()

# ============================================================
# 4. SHAP BAR PLOT (IMPORTÂNCIA ABSOLUTA MÉDIA)
# ============================================================
print("===== SHAP BAR PLOT =====")
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values_array, 
    X_test_scaled_sample, 
    feature_names=features, 
    plot_type="bar", 
    show=False
)
plt.title("SHAP Bar Plot — Impacto Médio Absoluto (|SHAP|) no Desempenho", fontsize=12, pad=15)
plt.tight_layout()
plt.show()

# ============================================================
# 5. SHAP DEPENDENCE PLOTS (RELAÇÕES NÃO LINEARES)
# ============================================================
print("===== SHAP DEPENDENCE PLOTS =====")

# Nota: Passamos X_test_sample (dados não padronizados) no parâmetro 'features'
# para que o eixo X exiba os valores reais das variáveis (0 a 1), facilitando a leitura no TCC.

# 5.1 IDH_ENEM_Renda
plt.figure(figsize=(8, 5))
shap.dependence_plot(
    "IDH_ENEM_Renda", 
    shap_values_array, 
    X_test_sample,
    interaction_index=None,
    show=False
)
plt.title("SHAP Dependence Plot — IDH ENEM Renda vs Impacto na Nota", fontsize=11)
plt.ylabel("Valor SHAP (Impacto na Nota do ENEM)")
plt.xlabel("IDH ENEM Renda (Escala 0 a 1)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# 5.2 IDH_ENEM_Edu
plt.figure(figsize=(8, 5))
shap.dependence_plot(
    "IDH_ENEM_Edu", 
    shap_values_array, 
    X_test_sample,
    interaction_index=None,
    show=False
)
plt.title("SHAP Dependence Plot — IDH ENEM Educação vs Impacto na Nota", fontsize=11)
plt.ylabel("Valor SHAP (Impacto na Nota do ENEM)")
plt.xlabel("IDH ENEM Educação (Escala 0 a 1)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# 5.3 FUNDEB (com interação visual com a Renda)
plt.figure(figsize=(8, 5))
shap.dependence_plot(
    "fundeb_mm3_log", 
    shap_values_array, 
    X_test_sample, 
    interaction_index="IDH_ENEM_Renda",
    show=False
)
plt.title("SHAP Dependence Plot — FUNDEB (log) x Renda do Aluno", fontsize=11)
plt.ylabel("Valor SHAP (Impacto na Nota do ENEM)")
plt.xlabel("FUNDEB MM3 (log)")
plt.tight_layout()
plt.show()

# ============================================================
# 6. SHAP WATERFALL PLOT (EXPLICABILIDADE INDIVIDUAL)
# ============================================================
print("===== SHAP WATERFALL PLOT =====")

# Analisa o 1º aluno da amostra (índice 0)
plt.figure(figsize=(8, 6))
shap.plots.waterfall(shap_explanation[0], show=False)
plt.title("SHAP Waterfall Plot — Decomposição da Nota de um Aluno Específico", fontsize=11, pad=15)
plt.tight_layout()
plt.show()

# %% Modelo XGBoost Classificador — Categorização do Desempenho

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score
)
from xgboost import XGBClassifier

# ============================================================
# 1. CRIAR VARIÁVEL TARGET CATEGÓRICA (3 NÍVEIS)
# ============================================================

# Rotulos textuais e numericos (0: Baixo, 1: Intermediario, 2: Alto)
labels_texto = ["Baixo Desempenho", "Desempenho Intermediário", "Alto Desempenho"]
labels_num = [0, 1, 2]

# Divisao por tercis (qcut) para garantir balanceamento perfeito entre as classes
df_final["performance_cat"] = pd.qcut(
    df_final["performance"], 
    q=3, 
    labels=labels_num
)

# Exibir os limites de corte de cada categoria
limites = pd.qcut(df_final["performance"], q=3).value_counts().sort_index()
print("========== DISTRIBUIÇÃO DAS CATEGORIAS (TERCIS) ==========")
print(df_final["performance_cat"].value_counts().sort_index())
print("\nIntervalos de Nota por Categoria:")
print(pd.qcut(df_final["performance"], q=3).cat.categories)

# ============================================================
# 2. DEFINIR FEATURES E TARGET (X e y)
# ============================================================

X_cls = df_final[features]
y_cls = df_final["performance_cat"].astype(int)

# ============================================================
# 3. DIVISÃO TREINO E TESTE STRATIFIED (80% / 20%)
# ============================================================

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls, 
    y_cls, 
    test_size=0.20, 
    random_state=42, 
    stratify=y_cls
)

# Padronizacao dos dados
scaler_cls = StandardScaler()
X_train_cls_scaled = pd.DataFrame(
    scaler_cls.fit_transform(X_train_cls), 
    columns=features
)
X_test_cls_scaled = pd.DataFrame(
    scaler_cls.transform(X_test_cls), 
    columns=features
)

# ============================================================
# 4. TREINAR O MODELO XGBCLASSIFIER
# ============================================================

model_cls = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)

model_cls.fit(X_train_cls_scaled, y_train_cls)

# ============================================================
# 5. AVALIAÇÃO DO MODELO E MATRIZ DE CONFUSÃO
# ============================================================

y_pred_cls = model_cls.predict(X_test_cls_scaled)

acc = accuracy_score(y_test_cls, y_pred_cls)
f1_macro = f1_score(y_test_cls, y_pred_cls, average="macro")

print("\n========== MÉTRICAS DO MODELO CLASSIFICADOR ==========")
print(f"Acurácia Geral: {acc:.4f}")
print(f"F1-Score (Macro): {f1_macro:.4f}")
print("\nRelatório de Classificação Detalhado:")
print(classification_report(y_test_cls, y_pred_cls, target_names=labels_texto))

# Plotagem da Matriz de Confusão
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test_cls, y_pred_cls)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_texto)
disp.plot(cmap="Blues", values_format="d", ax=plt.gca())
plt.title("Matriz de Confusão — Classificação do Desempenho no ENEM", fontsize=12, pad=15)
plt.xticks(rotation=15)
plt.grid(False)
plt.tight_layout()
plt.show()

# ============================================================
# 6. IMPORTÂNCIA DAS VARIÁVEIS (XGBCLASSIFIER)
# ============================================================

importances_cls = pd.DataFrame({
    "variavel": features,
    "importancia": model_cls.feature_importances_
}).sort_values(by="importancia", ascending=False)

print("\n========== IMPORTÂNCIA DAS VARIÁVEIS (CLASSIFICAÇÃO) ==========")
print(importances_cls)


# %% Análise SHAP Multiclasse (Classificador)

import shap

# ============================================================
# 1. AMOSTRAGEM PARA EXPLICABILIDADE
# ============================================================

N_SAMPLES_CLS = 2000

if len(X_test_cls_scaled) > N_SAMPLES_CLS:
    X_test_sample_cls = X_test_cls.sample(n=N_SAMPLES_CLS, random_state=42).reset_index(drop=True)
    X_test_cls_scaled_sample = X_test_cls_scaled.loc[X_test_sample_cls.index].reset_index(drop=True)
else:
    X_test_sample_cls = X_test_cls.reset_index(drop=True)
    X_test_cls_scaled_sample = X_test_cls_scaled.reset_index(drop=True)

# ============================================================
# 2. CÁLCULO DOS VALORES SHAP MULTICLASSE
# ============================================================

explainer_cls = shap.TreeExplainer(model_cls)
shap_explanation_cls = explainer_cls(X_test_cls_scaled_sample)

# Em modelos multiclasse, shap_explanation_cls tem dimensao: (amostras, features, num_classes)
# Array numpy para metodos legados
shap_values_cls = shap_explanation_cls.values

# ============================================================
# 3. SHAP BAR PLOT (COMPARATIVO ENTRE AS 3 CLASSES)
# ============================================================

print("===== SHAP BAR PLOT (MULTICLASSE) =====")
plt.figure(figsize=(10, 6))

# Se shap_values_cls for 3D, passamos a lista por classe ou o array completo
if len(shap_values_cls.shape) == 3:
    # Converte array 3D para lista de arrays (1 por classe) compatível com o summary_plot multiclasse
    shap_list = [shap_values_cls[:, :, i] for i in range(3)]
    shap.summary_plot(
        shap_list, 
        X_test_cls_scaled_sample, 
        feature_names=features, 
        class_names=labels_texto,
        plot_type="bar",
        show=False
    )
else:
    shap.summary_plot(
        shap_values_cls, 
        X_test_cls_scaled_sample, 
        feature_names=features, 
        plot_type="bar",
        show=False
    )

plt.title("SHAP Bar Plot — Importância das Variáveis por Categoria de Desempenho", fontsize=12, pad=15)
plt.tight_layout()
plt.show()

# ============================================================
# 4. SHAP SUMMARY PLOT (FOCO NOS EXTREMOS: BAIXO VS ALTO DESEMPENHO)
# ============================================================

# Classe 0: Baixo Desempenho
print("===== SHAP SUMMARY PLOT — CLASSE: BAIXO DESEMPENHO =====")
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values_cls[:, :, 0], 
    X_test_cls_scaled_sample, 
    feature_names=features, 
    show=False
)
plt.title("SHAP Beeswarm Plot — Fatores para Classificação em BAIXO Desempenho", fontsize=12, pad=15)
plt.tight_layout()
plt.show()

# Classe 2: Alto Desempenho
print("===== SHAP SUMMARY PLOT — CLASSE: ALTO DESEMPENHO =====")
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values_cls[:, :, 2], 
    X_test_cls_scaled_sample, 
    feature_names=features, 
    show=False
)
plt.title("SHAP Beeswarm Plot — Fatores para Classificação em ALTO Desempenho", fontsize=12, pad=15)
plt.tight_layout()
plt.show()

# %% Integração dos Indicadores do IDHM Estadual ao df_final

import pandas as pd
import numpy as np

# ============================================================
# 1. MAPEAR SIGLA DA UF EM DF_FINAL (A PARTIR DO CÓDIGO IBGE)
# ============================================================

df_final["UF_cod"] = df_final["cod_municipio"].str[:2]

mapa_uf = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA",
    "16": "AP", "17": "TO", "21": "MA", "22": "PI", "23": "CE",
    "24": "RN", "25": "PB", "26": "PE", "27": "AL", "28": "SE",
    "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP",
    "41": "PR", "42": "SC", "43": "RS", "50": "MS", "51": "MT",
    "52": "GO", "53": "DF"
}

df_final["UF"] = df_final["UF_cod"].map(mapa_uf)

# ============================================================
# 2. CARREGAR E TRATAR DADOS DO IDHM ESTADUAL
# ============================================================

caminho_idh = r"Y:\OneDrive\Documentos\Pós Graduação\USP\Data Science\TCC\Dados\IDH\idhporestado.xlsx"
df_idh = pd.read_excel(caminho_idh)

df_idh = df_idh.rename(columns={"Territorialidades": "UF"})
df_idh = df_idh[df_idh["UF"] != "Brasil"]

colunas_idh = [
    "IDHM Educação 2020", "IDHM Educação 2021", "IDHM Educação 2022",
    "IDHM Longevidade 2020", "IDHM Longevidade 2021", "IDHM Longevidade 2022",
    "IDHM Renda 2020", "IDHM Renda 2021", "IDHM Renda 2022",
    "IDHM 2020", "IDHM 2021", "IDHM 2022"
]

for col in colunas_idh:
    df_idh[col] = pd.to_numeric(df_idh[col], errors="coerce")

# Cálculo das Médias 2020-2022
df_idh["idhm_educacao_media"] = df_idh[["IDHM Educação 2020", "IDHM Educação 2021", "IDHM Educação 2022"]].mean(axis=1)
df_idh["idhm_longevidade_media"] = df_idh[["IDHM Longevidade 2020", "IDHM Longevidade 2021", "IDHM Longevidade 2022"]].mean(axis=1)
df_idh["idhm_renda_media"] = df_idh[["IDHM Renda 2020", "IDHM Renda 2021", "IDHM Renda 2022"]].mean(axis=1)
df_idh["idhm_total_media"] = df_idh[["IDHM 2020", "IDHM 2021", "IDHM 2022"]].mean(axis=1)

# Mapear nome do estado para sigla
mapa_nome_sigla = {
    "Acre": "AC", "Alagoas": "AL", "Amapá": "AP", "Amazonas": "AM",
    "Bahia": "BA", "Ceará": "CE", "Distrito Federal": "DF", "Espírito Santo": "ES",
    "Goiás": "GO", "Maranhão": "MA", "Mato Grosso": "MT", "Mato Grosso do Sul": "MS",
    "Minas Gerais": "MG", "Pará": "PA", "Paraíba": "PB", "Paraná": "PR",
    "Pernambuco": "PE", "Piauí": "PI", "Rio de Janeiro": "RJ", "Rio Grande do Norte": "RN",
    "Rio Grande do Sul": "RS", "Rondônia": "RO", "Roraima": "RR", "Santa Catarina": "SC",
    "São Paulo": "SP", "Sergipe": "SE", "Tocantins": "TO"
}

df_idh["UF"] = df_idh["UF"].map(mapa_nome_sigla)

cols_idh_final = [
    "UF", 
    "idhm_educacao_media", 
    "idhm_longevidade_media", 
    "idhm_renda_media", 
    "idhm_total_media"
]

df_idh_media = df_idh[cols_idh_final].drop_duplicates("UF")

# ============================================================
# 3. MERGE COM DF_FINAL (NÍVEL DO ESTUDANTE)
# ============================================================

df_final = df_final.merge(df_idh_media, on="UF", how="left")

# Atualizar lista de features para os novos modelos
features_expandidas = [
    "tx_homic_jovens_mm3",
    "tx_homic_mulheres_mm3",
    "tx_homic_homens_mm3",
    "tx_armas_fogo_mm3",
    "fundeb_mm3_log",
    "IDH_ENEM_Renda",
    "IDH_ENEM_Edu",
    "idhm_educacao_media",
    "idhm_longevidade_media",
    "idhm_renda_media",
    "idhm_total_media"
]

# Imputação de mediana por segurança
for f in features_expandidas:
    if df_final[f].isna().any():
        df_final[f] = df_final[f].fillna(df_final[f].median())

print("Novas colunas integradas com sucesso!")
print("Shape do df_final:", df_final.shape)


# %% XGBoost Regressão com IDHM Estadual + Análise SHAP

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# ============================================================
# 1. TREINO / TESTE E PADRONIZAÇÃO
# ============================================================

X_reg = df_final[features_expandidas]
y_reg = df_final["performance"]

X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
    X_reg, y_reg, test_size=0.20, random_state=42
)

scaler_r = StandardScaler()
X_tr_r_scaled = pd.DataFrame(scaler_r.fit_transform(X_tr_r), columns=features_expandidas)
X_te_r_scaled = pd.DataFrame(scaler_r.transform(X_te_r), columns=features_expandidas)

# ============================================================
# 2. MODELAGEM REGRESSÃO
# ============================================================

model_reg_exp = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model_reg_exp.fit(X_tr_r_scaled, y_tr_r)

y_pred_r = model_reg_exp.predict(X_te_r_scaled)

print("========== MÉTRICAS DO MODELO REGRESSÃO (EXPANDIDO) ==========")
print(f"RMSE: {np.sqrt(mean_squared_error(y_te_r, y_pred_r)):.4f}")
print(f"MAE:  {mean_absolute_error(y_te_r, y_pred_r):.4f}")
print(f"R²:   {r2_score(y_te_r, y_pred_r):.4f}")


# 1. Definir o tamanho da figura
plt.figure(figsize=(8, 6))

# 2. Gráfico de dispersão
sns.scatterplot(x=y_te_r, y=y_pred_r, alpha=0.3, color='#1f77b4', edgecolor=None)

# 3. Linha de referência perfeita (y = x) - fontweight removido, linewidth adicionado
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Ideal (y = x)')

# 4. Rótulos e título
plt.title('XGBoost Regressão (Expandido) — Real vs Predito', fontsize=12)
plt.xlabel('Nota Real (Enem)', fontsize=10)
plt.ylabel('Nota Predita (Enem)', fontsize=10)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# 5. Salvar e exibir
plt.tight_layout()
plt.savefig('figura14_real_vs_predito.png', dpi=300)
plt.show()

# Importância de Variáveis
imp_r = pd.DataFrame({
    "variavel": features_expandidas,
    "importancia": model_reg_exp.feature_importances_
}).sort_values(by="importancia", ascending=False)
print("\n========== IMPORTÂNCIA DAS VARIÁVEIS (REGRESSÃO) ==========")
print(imp_r)

# ============================================================
# 3. ANÁLISE SHAP (REGRESSÃO)
# ============================================================

N_SAMPLES = 2000
X_sample_r = X_te_r_scaled.sample(n=N_SAMPLES, random_state=42).reset_index(drop=True)

explainer_r = shap.TreeExplainer(model_reg_exp)
shap_exp_r = explainer_r(X_sample_r)

# Summary Plot (Beeswarm)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_exp_r.values, X_sample_r, feature_names=features_expandidas, show=False)
plt.title("SHAP Beeswarm Plot — Regressão Expandida (com IDHM Estadual)", fontsize=12, pad=15)
plt.tight_layout()
plt.show()

# Bar Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_exp_r.values, X_sample_r, feature_names=features_expandidas, plot_type="bar", show=False)
plt.title("SHAP Bar Plot — Regressão Expandida", fontsize=12, pad=15)
plt.tight_layout()
plt.show()

# Waterfall Plot (1º Aluno)
plt.figure(figsize=(8, 6))
shap.plots.waterfall(shap_exp_r[0], show=False)
plt.title("SHAP Waterfall Plot — Exemplo Individual (Regressão Expandida)", fontsize=11, pad=15)
plt.tight_layout()
plt.show()



# %% XGBoost Classificação com IDHM Estadual + Análise SHAP

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from xgboost import XGBClassifier

# Garantir categorias em tercis
labels_texto = ["Baixo Desempenho", "Desempenho Intermediário", "Alto Desempenho"]
if "performance_cat" not in df_final.columns:
    df_final["performance_cat"] = pd.qcut(df_final["performance"], q=3, labels=[0, 1, 2])

X_cls = df_final[features_expandidas]
y_cls = df_final["performance_cat"].astype(int)

# ============================================================
# 1. DIVISÃO STRATIFIED E PADRONIZAÇÃO
# ============================================================

X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
    X_cls, y_cls, test_size=0.20, random_state=42, stratify=y_cls
)

scaler_c = StandardScaler()
X_tr_c_scaled = pd.DataFrame(scaler_c.fit_transform(X_tr_c), columns=features_expandidas)
X_te_c_scaled = pd.DataFrame(scaler_c.transform(X_te_c), columns=features_expandidas)

# ============================================================
# 2. MODELAGEM CLASSIFICAÇÃO
# ============================================================

model_cls_exp = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)

model_cls_exp.fit(X_tr_c_scaled, y_tr_c)

y_pred_c = model_cls_exp.predict(X_te_c_scaled)

print("========== MÉTRICAS DO MODELO CLASSIFICADOR (EXPANDIDO) ==========")
print(f"Acurácia Geral: {accuracy_score(y_te_c, y_pred_c):.4f}")
print(f"F1-Score (Macro): {f1_score(y_te_c, y_pred_c, average='macro'):.4f}")
print("\nRelatório de Classificação Detalhado:")
print(classification_report(y_te_c, y_pred_c, target_names=labels_texto))

# Matriz de Confusão
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_te_c, y_pred_c)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_texto)
disp.plot(cmap="Blues", values_format="d", ax=plt.gca())
plt.title("Matriz de Confusão — Classificação Expandida", fontsize=12, pad=15)
plt.xticks(rotation=15)
plt.grid(False)
plt.tight_layout()
plt.show()

# Importância de Variáveis
imp_c = pd.DataFrame({
    "variavel": features_expandidas,
    "importancia": model_cls_exp.feature_importances_
}).sort_values(by="importancia", ascending=False)
print("\n========== IMPORTÂNCIA DAS VARIÁVEIS (CLASSIFICAÇÃO) ==========")
print(imp_c)

# ============================================================
# 3. ANÁLISE SHAP MULTICLASSE (CLASSIFICAÇÃO)
# ============================================================

X_sample_c = X_te_c_scaled.sample(n=N_SAMPLES, random_state=42).reset_index(drop=True)

explainer_c = shap.TreeExplainer(model_cls_exp)
shap_exp_c = explainer_c(X_sample_c)
shap_vals_c = shap_exp_c.values

# Bar Plot Multiclasse
plt.figure(figsize=(10, 6))
if len(shap_vals_c.shape) == 3:
    shap_list = [shap_vals_c[:, :, i] for i in range(3)]
    shap.summary_plot(shap_list, X_sample_c, feature_names=features_expandidas, class_names=labels_texto, plot_type="bar", show=False)
else:
    shap.summary_plot(shap_vals_c, X_sample_c, feature_names=features_expandidas, plot_type="bar", show=False)

plt.title("SHAP Bar Plot — Classificação Expandida por Categoria", fontsize=12, pad=15)
plt.tight_layout()
plt.show()

# Beeswarm — Baixo Desempenho (Classe 0)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_vals_c[:, :, 0], X_sample_c, feature_names=features_expandidas, show=False)
plt.title("SHAP Beeswarm Plot — Fatores para BAIXO Desempenho (Modelo Expandido)", fontsize=12, pad=15)
plt.tight_layout()
plt.show()

# Beeswarm — Alto Desempenho (Classe 2)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_vals_c[:, :, 2], X_sample_c, feature_names=features_expandidas, show=False)
plt.title("SHAP Beeswarm Plot — Fatores para ALTO Desempenho (Modelo Expandido)", fontsize=12, pad=15)
plt.tight_layout()
plt.show()