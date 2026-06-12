# -*- coding: utf-8 -*-
"""
@author: Maycon
"""

# %% Construção do df_final (ENEM 2022 + IPEA MM3 de 2022 + FUNDEB MM3 de 2022)

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
# EDA DO IPEA — IDENTIFICAR OS 10 MUNICÍPIOS MAIS VIOLENTOS
# ============================================================

import seaborn as sns
import matplotlib.pyplot as plt

# 1. Criar métrica geral de violência
df_ipea_last["violencia_media"] = df_ipea_last[
    ["tx_homic_jovens_mm3", "tx_homic_mulheres_mm3",
     "tx_homic_homens_mm3", "tx_armas_fogo_mm3"]
].mean(axis=1)

# 2. Mapear UF a partir do código IBGE
mapa_uf = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP", "17": "TO",
    "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB", "26": "PE", "27": "AL",
    "28": "SE", "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP",
    "41": "PR", "42": "SC", "43": "RS", "50": "MS", "51": "MT", "52": "GO", "53": "DF"
}

df_ipea_last["uf"] = (
    df_ipea_last["cod_municipio"]
    .astype(str)
    .str[:2]
    .map(mapa_uf)
)

# 3. Se você NÃO tem nome do município em df_ipea_last, remova "nome_municipio" da seleção.
#   Se tiver, mantenha. Aqui vou supor que NÃO tem, para evitar erro.

top10_violentos = df_ipea_last.sort_values(
    "violencia_media", ascending=False
).head(10)

print("\n===== 10 MUNICÍPIOS MAIS VIOLENTOS (IPEA MM3) =====")
print(
    top10_violentos[
        ["cod_municipio", "uf",
         "tx_homic_jovens_mm3", "tx_homic_mulheres_mm3",
         "tx_homic_homens_mm3", "tx_armas_fogo_mm3", "violencia_media"]
    ]
)

# 4. Gráfico: Top 10 municípios mais violentos
plt.figure(figsize=(10,6))
sns.barplot(
    data=top10_violentos,
    x="violencia_media",
    y="cod_municipio",
    hue="uf",
    dodge=False
)
plt.title("Top 10 Municípios Mais Violentos — IPEA (MM3)")
plt.xlabel("Violência Média (MM3)")
plt.ylabel("Código IBGE do Município")
plt.legend(title="UF")
plt.show()


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

df_fundeb = df_fundeb[df_fundeb["ano"].isin([2020, 2021, 2022])]

df_fundeb_mm = (
    df_fundeb.groupby("cod_municipio")["valor_fundeb"]
    .mean()
    .reset_index()
    .rename(columns={"valor_fundeb": "fundeb_mm3"})
)

# ============================================================
# ANÁLISE DO FUNDEB — TOP 10 MUNICÍPIOS COM MAIOR REPASSE (MM3)
# ============================================================
# ============================================================
# 1. INTEGRAÇÃO DO FUNDEB AO DATAFRAME FINAL
# ============================================================

# df_fundeb_mm já contém: cod_municipio + fundeb_mm3
# df_ipea_last já contém: cod_municipio + indicadores + uf

df_ipea_last = df_ipea_last.merge(
    df_fundeb_mm,
    on="cod_municipio",
    how="left"
)

# ============================================================
# 2. CARREGAR NOMES DOS MUNICÍPIOS (IBGE)
# ============================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ibge_url = "https://raw.githubusercontent.com/kelvins/Municipios-Brasileiros/main/csv/municipios.csv"

municipios_ibge = pd.read_csv(ibge_url, dtype=str)
municipios_ibge = municipios_ibge.rename(columns={
    "codigo_ibge": "cod_municipio",
    "nome": "nome_municipio",
    "uf": "uf_ibge"
})

municipios_ibge["cod_municipio"] = municipios_ibge["cod_municipio"].str.zfill(7)

# Integrar nomes ao dataframe final
df_ipea_last = df_ipea_last.merge(municipios_ibge, on="cod_municipio", how="left")

# ============================================================
# 3. TOP 10 MUNICÍPIOS COM MAIOR REPASSE FUNDEB (MM3)
# ============================================================

top10_fundeb = df_ipea_last.sort_values("fundeb_mm3", ascending=False).head(10)

print("\n===== TOP 10 MUNICÍPIOS COM MAIOR REPASSE FUNDEB (MM3) =====")
print(
    top10_fundeb[
        ["cod_municipio", "nome_municipio", "uf", "fundeb_mm3"]
    ]
)

# ============================================================
# 4. GRÁFICO — TOP 10 FUNDEB
# ============================================================

plt.figure(figsize=(12,7))
sns.barplot(
    data=top10_fundeb,
    x="fundeb_mm3",
    y="nome_municipio",
    hue="uf",
    dodge=False
)
plt.title("Top 10 Municípios com Maior Repasse FUNDEB — MM3")
plt.xlabel("Repasse Médio FUNDEB (MM3)")
plt.ylabel("Município")
plt.legend(title="UF")
plt.show()



# ============================================================
# 3. ENEM 2022 — CARREGAR E TRATAR (SEM AGRUPAR PARA EDA)
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
        "NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO"
    ]

    df = pd.read_csv(
        caminho_arquivo,
        sep=";",
        encoding="latin1",
        usecols=colunas_enem,
        low_memory=False
    )

    df = df[df["NU_ANO"] == 2022]

    # MANTER TODAS AS ESCOLAS EXCETO PRIVADAS
    df = df[df["TP_ESCOLA"] != 3]

    # CALCULAR PERFORMANCE
    df["performance"] = df[
        ["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO"]
    ].mean(axis=1)

    df = df[df["performance"].notna()]

    # PADRONIZAR CÓDIGOS IBGE
    df["CO_MUNICIPIO_ESC"] = df["CO_MUNICIPIO_ESC"].apply(padronizar_cod)
    df["CO_MUNICIPIO_PROVA"] = df["CO_MUNICIPIO_PROVA"].apply(padronizar_cod)

    # COMPLETAR DADOS FALTANTES DA ESCOLA COM OS DA PROVA
    campos_esc = [
        "CO_MUNICIPIO_ESC",
        "NO_MUNICIPIO_ESC",
        "CO_UF_ESC",
        "SG_UF_ESC"
    ]

    campos_prova = [
        "CO_MUNICIPIO_PROVA",
        "NO_MUNICIPIO_PROVA",
        "CO_UF_PROVA",
        "SG_UF_PROVA"
    ]

    for campo_esc, campo_prova in zip(campos_esc, campos_prova):
        df[campo_esc] = df[campo_esc].fillna(df[campo_prova])

    # RETORNAR DADOS INDIVIDUAIS PARA EDA
    return df


enem_2022_path = r"Y:\OneDrive\Documentos\Pós Graduação\USP\Data Science\TCC\Dados\ENEM\microdados_enem_2022\DADOS\MICRODADOS_ENEM_2022.csv"

df_enem_raw = carregar_tratar_enem_2022(enem_2022_path)


# ============================================================
# 4. AGRUPAR PARA O MODELO (SEM IMPACTAR A EDA)
# ============================================================

df_enem = (
    df_enem_raw.groupby("CO_MUNICIPIO_ESC")["performance"]
    .mean()
    .reset_index()
    .rename(columns={"CO_MUNICIPIO_ESC": "cod_municipio"})
)

df_enem["cod_municipio"] = df_enem["cod_municipio"].astype(str).str.zfill(7)


# ============================================================
# 5. INTEGRAR ENEM + IPEA + FUNDEB
# ============================================================

df_final = (
    df_enem
    .merge(df_ipea_last, on="cod_municipio", how="left")
    .merge(df_fundeb_mm, on="cod_municipio", how="left")
)

print("df_final shape:", df_final.shape)
print(df_final.head())


# ============================================================
# 6. ANÁLISE EXPLORATÓRIA DE DADOS (EDA) — ENEM 2022
# ============================================================

import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------
# Estatísticas descritivas (SEM NU_ANO)
# -------------------------
print("\n===== Estatísticas Descritivas — ENEM 2022 =====")
print(df_enem_raw.drop(columns=["NU_ANO"]).describe())

# -------------------------
# Distribuição das notas
# -------------------------
notas = ["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO", "performance"]

for col in notas:
    plt.figure(figsize=(7,4))
    sns.histplot(df_enem_raw[col], kde=True, bins=40)
    plt.title(f"Distribuição de {col}")
    plt.show()

# -------------------------
# Boxplot
# ------------------------

plt.figure(figsize=(6,4))
sns.boxplot(y=df_enem_raw["performance"])
plt.title("Boxplot da Performance — ENEM 2022")
plt.ylabel("Performance")
plt.show()

# -------------------------
# Correlação entre notas
# -------------------------
plt.figure(figsize=(8,6))
sns.heatmap(df_enem_raw[notas].corr(), annot=True, cmap="coolwarm")
plt.title("Correlação entre Notas do ENEM 2022")
plt.show()

# -------------------------
# Performance média por UF
# -------------------------
df_uf = df_enem_raw.groupby("SG_UF_ESC")["performance"].mean().sort_values()

plt.figure(figsize=(10,6))
sns.barplot(x=df_uf.index, y=df_uf.values)
plt.title("Performance Média por UF — ENEM 2022")
plt.xticks(rotation=45)
plt.show()

# -------------------------
# Relação entre áreas
# -------------------------
sns.scatterplot(data=df_enem_raw, x="NU_NOTA_MT", y="NU_NOTA_LC", alpha=0.3)
plt.title("Matemática vs Linguagens")
plt.show()

# -------------------------
# Outliers da performance
# -------------------------
plt.figure(figsize=(6,4))
sns.boxplot(y=df_enem_raw["performance"])
plt.title("Outliers da Performance — ENEM 2022")
plt.show()
