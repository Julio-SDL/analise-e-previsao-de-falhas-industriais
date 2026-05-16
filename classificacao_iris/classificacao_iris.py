from sklearn.datasets import load_iris
data = load_iris()

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

x = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = DecisionTreeClassifier()

# treinamento de modelo

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print ("primeiras previsoes do modelo: ")
print(y_pred[:10])

print("valores reais")
print(y_test[:10])

# avaliando desempenho

acc = accuracy_score(y_test, y_pred)
print(f"acuracia do modelo: {acc:.2f}")

# matriz de confusao

matriz = confusion_matrix(y_test, y_pred)
print("matriz de confusao")
print(matriz)

# relatorio de classificacao 

nomes_classes = data.target_names 
print("relatorio de classificacao:")
print(classification_report(y_test, y_pred, target_names=nomes_classes ))