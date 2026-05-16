import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# leitura e preparação dos dados

df = pd.read_csv("dataset_rede_neural_exemplo.csv")

print(df.head())

x = df.drop("falha", axis=1)
y = df["falha"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# normalização dos dados

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# criação e treinamento da rede neural

model = MLPClassifier(
    hidden_layer_sizes=(5),
    activation="relu",
    max_iter=1000,
    random_state=42
    )

model.fit(x_train, y_train)

# previsões e avaliação do modelo

y_pred = model.predict(x_test)

acc= accuracy_score(y_test, y_pred)
print("acuracia da rede neural", acc)

matriz = confusion_matrix(y_test, y_pred)
print(matriz)

