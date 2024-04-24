import numpy as np
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score
# Classificador montado
from sklearn.dissimilarity import DissimilarityRNGClassifier

# data = load_iris()
data = load_digits()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

dummy = DissimilarityRNGClassifier(estimator=KNeighborsClassifier(), random_state=42)
dummy.fit(X_train, y_train)
score = model_selection.cross_val_score(dummy, X_test, y_test, cv=10)
print(f"Acc mean:{score.mean()}\nAcc std:{score.std()}\nAcc per fold:{score}")

ypred=model_selection.cross_val_predict(dummy, X_test, y_test, cv=10)
# Precisão
prec=precision_score(y_test, ypred, average='weighted')
print("Precision:", prec)

# Revocação (Recall)
recall=recall_score(y_test, ypred, average='weighted')
print("Revocação:", recall)

# f1 score
f1 = f1_score(y_test, ypred, average='macro')
print("f1 score:", f1)

# Matriz de confusão
cm=confusion_matrix(y_test, ypred)
print(cm)