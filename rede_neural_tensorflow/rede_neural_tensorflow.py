import pandas as pd 
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# leitura e preparação dos dados

df = pd.read_csv("dataset_rede_neural_exemplo.csv")

x = df.drop("falha", axis=1)
y = df["falha"]

print("formato de x:" , x.shape)
print("formato de y " , y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42 
)

print("x_train: ", x_train.shape)
print("x_test: ", x_test.shape)
print("y_train: ", y_train.shape)
print("y_train: ", y_train.shape)

# normalização dos dados

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# criação da rede neural

model = tf.keras.Sequential([
    tf.keras.layers.Dense (8, activation="relu", input_shape=(x_train.shape[1],)),
    tf.keras.layers.Dense (1, activation="sigmoid")
    
])

# configuração do modelo

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# treinamento da rede neural

print(model.summary())

history = model.fit(
    x_train,
    y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.2,
    verbose=1
)

# avaliação do modelo

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("loss no conjunto de teste", loss)
print("acuracia no conjunto de teste", accuracy)
   
