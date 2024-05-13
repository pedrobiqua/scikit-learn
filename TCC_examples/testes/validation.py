###### LIB ARFF_CONVERT ######
import numpy as np
import sys
sys.path.insert(0, '/home/pedro/projects/dataset_arff_covert')
from arffconvert import load_SEAGenerator_test_mode # Carrega a base de dados que está no formato arff para numpy X e Y
from arffconvert import load_AgrawalGenerator_test_mode # Carrega a base de dados que está no formato arff para numpy X e Y
##############################

# SELEÇÃO DO MODELO
from sklearn import model_selection
# CLASSIFICADORES
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
# BASE DE DADOS
from sklearn.datasets import load_iris, load_digits
# SEPARAÇÃO DO CONJUNTO DE TRAINO E TESTE
from sklearn.model_selection import train_test_split
# METRICAS
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# Classificador usando ESPAÇO DE DISSIMILARIDADE COM SELEÇÃO DOS R ALEATÓRIOS
from sklearn.dissimilarity import DissimilarityRNGClassifier

X, y = load_SEAGenerator_test_mode()
# X, y = load_AgrawalGenerator_test_mode()

results_confusion = []
results_score     = []

for i in range(20, 50):
    # Utiliza o mesmo random_state tanto para o separamento quanto dentro do classificador
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, test_size=0.5, random_state=i, shuffle=False)
    print(X_train)
    print(X_test)
    
    # Ainda não está bom, talvez ou a geração ou a separação não esteja correto
    dissimilarity = DissimilarityRNGClassifier(estimator=KNeighborsClassifier(n_neighbors=1), random_state=i) # 1NN com Dissimilaridade
    # dissimilarity = KNeighborsClassifier()
    print(dissimilarity)
    dissimilarity.fit(X_train, y_train)

    score = model_selection.cross_val_score(dissimilarity, X_test, y_test, cv=10)
    print(f"Acc mean:{score.mean()}\nAcc std:{score.std()}\nAcc per fold:{score}")
    results_score.append(score)

    ypred=model_selection.cross_val_predict(dissimilarity, X_test, y_test, cv=10)
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
    results_confusion.append(cm)
    print(cm)


# Salvar depois os resultados em um excel