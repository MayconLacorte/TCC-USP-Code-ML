# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 04:05:41 2026

@author: Maycon
"""

# %% Construção do df_final (ENEM 2022 + IPEA MM3 + FUNDEB MM3)

import pandas as pd
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
    df = pd.read_csv(caminho_arquivo, sep=",", encoding="latin1")
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
    dfs_ipea.append(carregar_ipea(caminho, nome_var))

df_ipea = dfs_ipea[0]
for df_temp in dfs_ipea[1:]:
    df_ipea = df_ipea.merge(df_temp, on=["cod_municipio", "ano"], how="outer")

df_ipea = df_ipea.sort_values(["cod_municipio", "ano"]).reset_index(drop=True)

taxas = ["tx_homic_jovens", "tx_homic_mulheres", "tx_homic_homens", "tx_armas_fogo"]

df_ipea_mm = df_ipea.copy()
for col in taxas:
    df_ipea_mm[col + "_mm3"] = (
        df_ipea_mm.groupby("cod_municipio")[col]
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

df_fundeb = pd.read_csv(caminho_fundeb, sep=";", encoding="latin1")
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

df_fundeb = df_fundeb[df_fundeb["ano"].isin([2020, 2021, 2022])]

df_fundeb_mm = (
    df_fundeb.groupby("cod_municipio")["valor_fundeb"]
    .mean()
    .reset_index()
    .rename(columns={"valor_fundeb": "fundeb_mm3"})
)


# ============================================================
# 3. ENEM 2022 — CARREGAR E TRATAR
# ============================================================

def carregar_tratar_enem_2022(caminho_arquivo):
    
    colunas_enem = [
        "NU_ANO", "TP_ESCOLA", "CO_MUNICIPIO_ESC", "NO_MUNICIPIO_ESC",
        "CO_UF_ESC", "SG_UF_ESC", "CO_MUNICIPIO_PROVA", "NO_MUNICIPIO_PROVA",
        "CO_UF_PROVA", "SG_UF_PROVA",
        "NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO"
    ]

    df = pd.read_csv(caminho_arquivo, sep=";", encoding="latin1", usecols=colunas_enem, low_memory=False)

    df = df[df["NU_ANO"] == 2022]
    df = df[df["TP_ESCOLA"].notna() & (df["TP_ESCOLA"] != 3)]

    df["performance"] = df[
        ["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO"]
    ].mean(axis=1)

    df = df[df["performance"].notna()]

    df["CO_MUNICIPIO_ESC"] = df["CO_MUNICIPIO_ESC"].apply(padronizar_cod)
    df["CO_MUNICIPIO_PROVA"] = df["CO_MUNICIPIO_PROVA"].apply(padronizar_cod)

    campos_esc = ["CO_MUNICIPIO_ESC", "NO_MUNICIPIO_ESC", "CO_UF_ESC", "SG_UF_ESC"]
    campos_prova = ["CO_MUNICIPIO_PROVA", "NO_MUNICIPIO_PROVA", "CO_UF_PROVA", "SG_UF_PROVA"]

    for campo_esc, campo_prova in zip(campos_esc, campos_prova):
        df[campo_esc] = df[campo_esc].fillna(df[campo_prova])

    df_grouped = df.groupby("CO_MUNICIPIO_ESC")["performance"].mean().reset_index()
    df_grouped = df_grouped.rename(columns={"CO_MUNICIPIO_ESC": "cod_municipio"})
    df_grouped["cod_municipio"] = df_grouped["cod_municipio"].astype(str).str.zfill(7)

    return df_grouped


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


# ============================================================
# 5. CLASSIFICAR DESEMPENHO DO ENEM EM 5 CATEGORIAS
# ============================================================

df_final["categoria_enem"] = pd.qcut(
    df_final["performance"],
    q=5,
    labels=["Muito ruim", "Ruim", "Médio", "Bom", "Muito bom"]
)

print(df_final["categoria_enem"].value_counts())
print(df_final.head())


# %% MODELO XGBOOST — CLASSIFICAÇÃO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# 1. PREPARAR BASE PARA O MODELO
# ============================================================

df_model = df_final.dropna().copy()
df_model = df_model.drop(columns=["cod_municipio"])

X = df_model.drop(columns=["categoria_enem", "performance"])
y = df_model["categoria_enem"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ============================================================
# 2. TREINO / TESTE
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)

# ============================================================
# 3. TREINAR MODELO XGBOOST CLASSIFIER
# ============================================================

model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="multi:softprob",
    num_class=5
)

model.fit(X_train, y_train)

# ============================================================
# 4. AVALIAÇÃO DO MODELO
# ============================================================

y_pred = model.predict(X_test)

print("\n========== MÉTRICAS DO MODELO (CLASSIFICAÇÃO) ==========")
print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ============================================================
# 5. MATRIZ DE CONFUSÃO
# ============================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão — XGBoost Classificação")
plt.show()

# ============================================================
# 6. IMPORTÂNCIA DAS VARIÁVEIS
# ============================================================

importances = pd.DataFrame({
    "variavel": X.columns,
    "importancia": model.feature_importances_
}).sort_values(by="importancia", ascending=False)

print("\n========== IMPORTÂNCIA DAS VARIÁVEIS ==========")
print(importances)

plt.figure(figsize=(8, 6))
sns.barplot(data=importances, x="importancia", y="variavel")
plt.title("Importância das Variáveis — XGBoost (Classificação)")
plt.show()


# %% Análise SHAP (SHapley Additive exPlanations)

import shap
import matplotlib.pyplot as plt

# Criar explicador SHAP para classificação
explainer = shap.Explainer(model, X_train)

# Calcular valores SHAP
shap_values = explainer(X_test)

# ============================================================
# 1. SUMMARY PLOT (impacto global das variáveis)
# ============================================================

plt.title("SHAP Summary Plot — XGBoost Classificação")
shap.summary_plot(shap_values, X_test, feature_names=X.columns, class_names=label_encoder.classes_)

plt.show()

# ============================================================
# 2. BAR PLOT — Importância média absoluta
# ============================================================

plt.title("SHAP Feature Importance — Mean |SHAP|")
shap.summary_plot(shap_values, X_test, feature_names=X.columns, plot_type="bar")
plt.show()

# ============================================================
# 3. DEPENDENCE PLOT (impacto de uma variável específica)
# ============================================================

shap.dependence_plot(
    ind="fundeb_mm3",      
    shap_values=shap_values.values,
    features=X_test,
    feature_names=X.columns
)
