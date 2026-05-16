import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import zipfile

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# configurações 
dataset_path = "dataset"
classes = ["gatos", "caes"]
img_size = 64

# descompactar dataset 
zip_file_name = "dataset.zip"
extract_path = "."

if os.path.exists(zip_file_name):
    with zipfile.ZipFile(zip_file_name, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Arquivo '{zip_file_name}' descompactado com sucesso.")
else:
    print(f"Erro: Arquivo '{zip_file_name}' não encontrado.")

# carregar e pré-processar imagens 
X = []
y = []

for classe in classes:
    pasta_classe = os.path.join(dataset_path, classe)

    for arquivo in os.listdir(pasta_classe):
        caminho_img = os.path.join(pasta_classe, arquivo)

        img = cv2.imread(caminho_img)

        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        img = img / 255.0

        X.append(img)
        y.append(classes.index(classe))

# converter para arrays 
X = np.array(X, dtype=np.float32)
y = np.array(y)

print("Formato de X:", X.shape)
print("Formato de y:", y.shape)

# visualizar exemplo 
plt.imshow(X[0])
plt.title(f"Classe: {classes[y[0]]}")
plt.axis("off")
plt.show()

# dividir treino e teste 0
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Treino:", X_train.shape)
print("Teste:", X_test.shape)

# criar modelo CNN 
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(img_size, img_size, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# compilar modelo 
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# treinar modelo
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=8,
    validation_split=0.2
)

# avaliar modelo 
loss, acc = model.evaluate(X_test, y_test)
print("Loss:", loss)
print("Accuracy:", acc)

y_prob = model.predict(X_test)
y_pred = (y_prob > 0.5).astype(int).flatten()

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=classes))

# predição de nova imagem 
nova_imagem = cv2.imread("nova_imagem.jpg")
nova_imagem = cv2.cvtColor(nova_imagem, cv2.COLOR_BGR2RGB)
nova_imagem = cv2.resize(nova_imagem, (img_size, img_size))
nova_imagem = nova_imagem / 255.0
nova_imagem = np.expand_dims(nova_imagem, axis=0)

pred = model.predict(nova_imagem)

if pred[0][0] > 0.5:
    print("Classe prevista: gato")
else:
    print("Classe prevista: cão")