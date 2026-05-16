import pandas as pd

# leitura do dataset

df = pd.read_csv("exercicio_preprocessamento_dados.csv")

# tratando valores nulos

df["temperatura"] = df["temperatura"].fillna(
    df["temperatura"].mean()
)

df["vibracao"] = df["vibracao"].fillna(
    df["vibracao"].mean()
)

df["pressao"] = df["pressao"].fillna(
    df["pressao"].mean()
)

# tratamento de outliers - temperatura

Q1 = df["temperatura"].quantile(0.25)
Q3 = df["temperatura"].quantile(0.75)

IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

outliers = df[
    (df["temperatura"] < limite_inferior) |
    (df["temperatura"] > limite_superior)
]

mediana = df["temperatura"].median()

df.loc[
    df["temperatura"] > limite_superior,
    "temperatura"
] = mediana

df.loc[
    df["temperatura"] < limite_inferior,
    "temperatura"
] = mediana

# tratamento de outliers - vibração

Q1 = df["vibracao"].quantile(0.25)
Q3 = df["vibracao"].quantile(0.75)

IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

mediana = df["vibracao"].median()

df.loc[
    df["vibracao"] > limite_superior,
    "vibracao"
] = mediana

df.loc[
    df["vibracao"] < limite_inferior,
    "vibracao"
] = mediana

# tratamento de outliers - pressão

Q1 = df["pressao"].quantile(0.25)
Q3 = df["pressao"].quantile(0.75)

IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

mediana = df["pressao"].median()

df.loc[
    df["pressao"] > limite_superior,
    "pressao"
] = mediana

df.loc[
    df["pressao"] < limite_inferior,
    "pressao"
] = mediana

# tratamento de outliers - tempo de operação

Q1 = df["tempo_operacao"].quantile(0.25)
Q3 = df["tempo_operacao"].quantile(0.75)

IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

mediana = df["tempo_operacao"].median()

df.loc[
    df["tempo_operacao"] > limite_superior,
    "tempo_operacao"
] = mediana

df.loc[
    df["tempo_operacao"] < limite_inferior,
    "tempo_operacao"
] = mediana

# preenchimento final de valores nulos

df["temperatura"] = df["temperatura"].fillna(
    df["temperatura"].median()
)

df["vibracao"] = df["vibracao"].fillna(
    df["vibracao"].median()
)

df["pressao"] = df["pressao"].fillna(
    df["pressao"].median()
)

# salvando dataset tratado

df.to_csv("dataset_tratado.csv", index=False)
print("salvo")

