# -*- coding: utf-8 -*-
"""
TCC — Construção do df_final (ENEM 2022 + IPEA MM3 + FUNDEB MM3 + IDH médio 2020–2022)
Classificação estadual com XGBoost
@author: Maycon
"""

# ============================================================
# IMPORTAÇÕES
# ============================================================

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import shap

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
        "NU_ANO", "TP_ESCOLA", "CO_MUNICIPIO_ESC",
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

    df_grouped = df.groupby("CO_MUNICIPIO_ESC")["performance"].mean().reset_index()
    df_grouped = df_grouped.rename(columns={"CO_MUNICIPIO_ESC": "cod_municipio"})
    df_grouped["cod_municipio"] = df_grouped["cod_municipio"].astype(str).str.zfill(7)

    return df_grouped

enem_2022_path = r"Y:\OneDrive\Documentos\Pós Graduação\USP\Data Science\TCC\Dados\ENEM\microdados_enem_2022\DADOS\MICRODADOS_ENEM_2022.csv"
df_enem = carregar_tratar_enem_2022(enem_2022_path)

# ============================================================
# 4. INTEGRAR ENEM + IPEA + FUNDEB → df_final
# ============================================================

df_final = (
    df_enem
    .merge(df_ipea_last, on="cod_municipio", how="left")
    .merge(df_fundeb_mm, on="cod_municipio", how="left")
)

# ============================================================
# 5. AGRUPAMENTO POR ESTADO
# ============================================================

df_estado = df_final.copy()
df_estado["UF"] = df_estado["cod_municipio"].str[:2]

mapa_uf = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA",
    "16": "AP", "17": "TO", "21": "MA", "22": "PI", "23": "CE",
    "24": "RN", "25": "PB", "26": "PE", "27": "AL", "28": "SE",
    "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP",
    "41": "PR", "42": "SC", "43": "RS", "50": "MS", "51": "MT",
    "52": "GO", "53": "DF"
}

df_estado["UF"] = df_estado["UF"].map(mapa_uf)

df_estado = df_estado.groupby("UF").agg({
    "performance": "mean",
    "tx_homic_jovens_mm3": "mean",
    "tx_homic_mulheres_mm3": "mean",
    "tx_homic_homens_mm3": "mean",
    "tx_armas_fogo_mm3": "mean",
    "fundeb_mm3": "mean"
}).reset_index()

# ============================================================
# 6. CARREGAR IDH E CALCULAR MÉDIAS 2020–2022
# ============================================================

caminho_idh = r"Y:\OneDrive\Documentos\Pós Graduação\USP\Data Science\TCC\Dados\IDH\idhporestado.xlsx"
df_idh = pd.read_excel(caminho_idh)

df_idh = df_idh.rename(columns={"Territorialidades": "UF"})
df_idh = df_idh[df_idh["UF"] != "Brasil"]

# Converter colunas numéricas
colunas_idh = [
    "IDHM Educação 2020", "IDHM Educação 2021", "IDHM Educação 2022",
    "IDHM Longevidade 2020", "IDHM Longevidade 2021", "IDHM Longevidade 2022",
    "IDHM Renda 2020", "IDHM Renda 2021", "IDHM Renda 2022",
    "IDHM 2020", "IDHM 2021", "IDHM 2022"
]

for col in colunas_idh:
    df_idh[col] = pd.to_numeric(df_idh[col], errors="coerce")

# Criar médias
df_idh["idhm_educacao_media"] = df_idh[[
    "IDHM Educação 2020", "IDHM Educação 2021", "IDHM Educação 2022"
]].mean(axis=1)

df_idh["idhm_longevidade_media"] = df_idh[[
    "IDHM Longevidade 2020", "IDHM Longevidade 2021", "IDHM Longevidade 2022"
]].mean(axis=1)

df_idh["idhm_renda_media"] = df_idh[[
    "IDHM Renda 2020", "IDHM Renda 2021", "IDHM Renda 2022"
]].mean(axis=1)

df_idh["idhm_total_media"] = df_idh[[
    "IDHM 2020", "IDHM 2021", "IDHM 2022"
]].mean(axis=1)

# ============================================================
# 7. CONVERTER NOMES COMPLETOS → SIGLAS (CORREÇÃO DO MERGE)
# ============================================================

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

df_idh_media = df_idh[[
    "UF",
    "idhm_educacao_media",
    "idhm_longevidade_media",
    "idhm_renda_media",
    "idhm_total_media"
]]

# MERGE FINAL (AGORA FUNCIONA)
df_estado = df_estado.merge(df_idh_media, on="UF", how="left")

# ============================================================
# 8. CRIAR 3 CATEGORIAS DE DESEMPENHO
# ============================================================

df_estado["categoria_3"] = pd.qcut(
    df_estado["performance"],
    q=3,
    labels=["Baixo desempenho", "Desempenho intermediário", "Alto desempenho"]
)

# ============================================================
# 9. PREPARAR BASE PARA O MODELO
# ============================================================

X = df_estado.drop(columns=["UF", "performance", "categoria_3"])
y = df_estado["categoria_3"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ============================================================
# 10. VALIDAÇÃO CRUZADA (K=5)
# ============================================================

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
conf_matrix_total = np.zeros((3, 3))

for train_index, test_index in kf.split(X, y_encoded):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="multi:softprob",
        num_class=3
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracies.append(accuracy_score(y_test, y_pred))
    conf_matrix_total += confusion_matrix(y_test, y_pred, labels=[0,1,2])

# ============================================================
# 11. RESULTADOS
# ============================================================

print("\n========== RESULTADOS DA VALIDAÇÃO CRUZADA (3 CATEGORIAS + IDH MÉDIO) ==========")
print(f"Acurácia média: {np.mean(accuracies):.4f}")
print(f"Desvio padrão: {np.std(accuracies):.4f}")

conf_matrix_total = conf_matrix_total.astype(int)

plt.figure(figsize=(7, 6))
sns.heatmap(conf_matrix_total, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão — Validação Cruzada (3 Categorias + IDH Médio)")
plt.show()

# ============================================================
# 12. IMPORTÂNCIA DAS VARIÁVEIS
# ============================================================

importances = pd.DataFrame({
    "variavel": X.columns,
    "importancia": model.feature_importances_
}).sort_values(by="importancia", ascending=False)

print("\n========== IMPORTÂNCIA DAS VARIÁVEIS (COM IDH MÉDIO) ==========")
print(importances)

plt.figure(figsize=(8, 6))
sns.barplot(data=importances, x="importancia", y="variavel")
plt.title("Importância das Variáveis — XGBoost (3 Categorias + IDH Médio)")
plt.show()

# # ============================================================
# # 13. SHAP — INTERPRETAÇÃO GLOBAL (COM NOMES DAS CATEGORIAS)
# # ============================================================

explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Summary plot com nomes reais das classes
plt.title("SHAP Summary Plot — XGBoost (3 Categorias + IDH Médio)")
shap.summary_plot(
    shap_values,
    X,
    feature_names=X.columns,
    class_names=label_encoder.classes_
)

# Feature importance com nomes reais das classes
plt.title("SHAP Feature Importance — Mean |SHAP|")
shap.summary_plot(
    shap_values,
    X,
    feature_names=X.columns,
    plot_type="bar",
    class_names=label_encoder.classes_
)
