###### LIB ARFF_CONVERT ######
import numpy as np
import sys
sys.path.insert(0, '/home/pedro/projects/dataset_arff_covert')
from arffconvert import load_SEAGenerator_test_mode # Carrega a base de dados que está no formato arff para numpy X e Y
from arffconvert import load_SEAGenerator_test_f2_f4 # Carrega a base de dados que está no formato arff para numpy X e Y
from arffconvert import load_AgrawalGenerator_test_mode # Carrega a base de dados que está no formato arff para numpy X e Y
from arffconvert import load_AgrawalGenerator_test_mode_f2_f9 # Carrega a base de dados que está no formato arff para numpy X e Y
##############################

# SELEÇÃO DO MODELO
from sklearn import model_selection
# CLASSIFICADORES
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
# BASE DE DADOS
from sklearn.datasets import load_iris, load_digits
# SEPARAÇÃO DO CONJUNTO DE TRAINO E TESTE
from sklearn.model_selection import train_test_split
# METRICAS
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score

# Classificador usando ESPAÇO DE DISSIMILARIDADE COM SELEÇÃO DOS R ALEATÓRIOS
from sklearn.dissimilarity import DissimilarityRNGClassifier

# X, y = load_SEAGenerator_test_mode()
X, y = load_SEAGenerator_test_f2_f4()
# X, y = load_AgrawalGenerator_test_mode()
# X, y = load_AgrawalGenerator_test_mode_f2_f9()

def model_test_dissimilarity(random_state, estimator, X_train, y_train, X_test, y_test):
    # dissimilarity = DissimilarityRNGClassifier(estimator=estimator, random_state=random_state)
    dissimilarity = KNeighborsClassifier(n_neighbors=1)
    print(dissimilarity)
    dissimilarity.fit(X_train, y_train)

    score = model_selection.cross_val_score(dissimilarity, X_test, y_test, cv=10)
    print(f"Acc mean:{score.mean()}\nAcc std:{score.std()}\nAcc per fold:{score}")

    ypred=model_selection.cross_val_predict(dissimilarity, X_test, y_test, cv=10)

    # Matriz de confusão
    cm=confusion_matrix(y_test, ypred)
    print(cm)

    return score, cm

def model_test_1nn(X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=1)
    print(knn)
    knn.fit(X_train, y_train)

    score = model_selection.cross_val_score(knn, X_test, y_test, cv=10)
    print(f"Acc mean:{score.mean()}\nAcc std:{score.std()}\nAcc per fold:{score}")

    ypred=model_selection.cross_val_predict(knn, X_test, y_test, cv=10)

    # Matriz de confusão
    cm=confusion_matrix(y_test, ypred)
    print(cm)

    return score, cm

def model_test_3nn(X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=3)
    print(knn)
    knn.fit(X_train, y_train)

    score = model_selection.cross_val_score(knn, X_test, y_test, cv=10)
    print(f"Acc mean:{score.mean()}\nAcc std:{score.std()}\nAcc per fold:{score}")

    ypred=model_selection.cross_val_predict(knn, X_test, y_test, cv=10)

    # Matriz de confusão
    cm=confusion_matrix(y_test, ypred)
    print(cm)

    return score, cm

def model_test_naive_bayes(X_train, y_train, X_test, y_test):
    naive_bayes = GaussianNB()
    print(naive_bayes)
    naive_bayes.fit(X_train, y_train)

    score = model_selection.cross_val_score(naive_bayes, X_test, y_test, cv=10)
    print(f"Acc mean:{score.mean()}\nAcc std:{score.std()}\nAcc per fold:{score}")

    ypred=model_selection.cross_val_predict(naive_bayes, X_test, y_test, cv=10)

    # Matriz de confusão
    cm=confusion_matrix(y_test, ypred)
    print(cm)

    return score, cm

def model_test_decision_tree(random_state, X_train, y_train, X_test, y_test):
    decision_tree = DecisionTreeClassifier(random_state=random_state)
    print(decision_tree)
    decision_tree.fit(X_train, y_train)

    
    score = model_selection.cross_val_score(decision_tree, X_test, y_test, cv=10)
    print(f"Acc mean:{score.mean()}\nAcc std:{score.std()}\nAcc per fold:{score}")

    ypred=model_selection.cross_val_predict(decision_tree, X_test, y_test, cv=10)

    # ypred = decision_tree.predict(X_test)
    # precisao = accuracy_score(y_test, ypred)
    # print(precisao)

    # Matriz de confusão
    cm=confusion_matrix(y_test, ypred)
    print(cm)

    return score, cm

def experimento_1():
    try:
        results_confusion = []
        results_score     = []

        for i in range(20, 50): # Vamos testar o random_state de 20 a 50
            # 50% treino e 50% para teste
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, test_size=0.5, shuffle=False)

            print(X_test)
            
            # score, cm = model_test_dissimilarity(random_state=i, estimator=DecisionTreeClassifier(random_state=i), X_train=X_train, y_train=y_train,X_test=X_test, y_test=y_test)
            # results_score.append(score)
            # results_confusion.append(cm)

            # score, cm = model_test_dissimilarity(random_state=i, estimator=GaussianNB(), X_train=X_train, y_train=y_train,X_test=X_test, y_test=y_test)
            # results_score.append(score)
            # results_confusion.append(cm)

            # score, cm = model_test_dissimilarity(random_state=i, estimator=KNeighborsClassifier(n_neighbors=1), X_train=X_train, y_train=y_train,X_test=X_test, y_test=y_test)
            # results_score.append(score)
            # results_confusion.append(cm)

            # score, cm = model_test_dissimilarity(random_state=i, estimator=KNeighborsClassifier(n_neighbors=3), X_train=X_train, y_train=y_train,X_test=X_test, y_test=y_test)
            # results_score.append(score)
            # results_confusion.append(cm)

            score, cm = model_test_decision_tree(random_state=i, X_train=X_train, y_train=y_train,X_test=X_test, y_test=y_test) # Roda o knn para  mesmma base
            results_score.append(score)
            results_confusion.append(cm)

            score, cm = model_test_naive_bayes(X_train=X_train, y_train=y_train,X_test=X_test, y_test=y_test) # Roda o knn para  mesmma base
            results_score.append(score)
            results_confusion.append(cm)

            score, cm = model_test_1nn(X_train=X_train, y_train=y_train,X_test=X_test, y_test=y_test) # Roda o knn para  mesmma base
            results_score.append(score)
            results_confusion.append(cm)

            score, cm = model_test_3nn(X_train=X_train, y_train=y_train,X_test=X_test, y_test=y_test) # Roda o knn para  mesmma base
            results_score.append(score)
            results_confusion.append(cm)

    # Salvar depois os resultados em um excel para validar e analisar melhor


    except KeyboardInterrupt:
        print(f"Tecla Crtl + c precionada!")

def experimento_2():
    print("Não montado")

#Inicializa os experimento
experimento_1()
experimento_2()