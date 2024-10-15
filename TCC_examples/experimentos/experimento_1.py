import numpy as np
import sys

###### LIB ARFF_CONVERT DATASETS ######
#from dataset_arff.arffconvert import load_SEAGenerator_test_mode # Carrega a base de dados que está no formato arff para numpy X e Y 
from dataset_arff.arffconvert import load_SEAGenerator_test_f2_f4 
from dataset_arff.arffconvert import load_AssetNegotiationGenerator_f1_f5 
from dataset_arff.arffconvert import load_AgrawalGenerator_test_mode_f2_f9 
#######################################

# Author: Pedro Bianchini de Quadros      <quadros.pedro@pucpr.edu.br>
#         Gabriel Antonio Gomes de Farias <gomes.farias@pucpr.edu.br>
# Co-Author: Jean Paul Barddal
#  ESCOLA POLITÉCNICA - PUCPR - CIÊNCIA DA COMPUTAÇÃO

# SELEÇÃO DO MODELO
from sklearn import model_selection
# CLASSIFICADORES
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
# BASE DE DADOS TESTE DO SCIKIT-LEARN
from sklearn.datasets import load_iris, load_digits
# SEPARAÇÃO DO CONJUNTO DE TRAINO E TESTE
from sklearn.model_selection import train_test_split
# METRICAS
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score

# Classificador usando ESPAÇO DE DISSIMILARIDADE COM SELEÇÃO DOS R ALEATÓRIOS
from sklearn.dissimilarity import DissimilarityRNGClassifier
from sklearn.dissimilarity import DissimilarityIHDClassifier
from sklearn.dissimilarity import DissimilarityCentroidClassifier
from sklearn.dissimilarity import DissimilarityIHDClassifier
from sklearn.dissimilarity import DissimilarityCentroidClassifier

import pandas as pd
import re

# Vamos testar o experimento 30 vezes, usando seeds de 20 a 50
range_experimento = [20, 50]

# Datasets que iremos utilizar
function_datasets_list = [
    load_AssetNegotiationGenerator_f1_f5, 
    load_AgrawalGenerator_test_mode_f2_f9,
    load_SEAGenerator_test_f2_f4
]

# Função responsavel por rodar os algoritmos
def model_test(estimator, X_train, y_train, X_test, y_test):
    
    estimator.fit(X_train, y_train)

    ypred = estimator.predict(X_test)
    tx_acerto = accuracy_score(y_test, ypred)
    # print(tx_acerto)

    # Matriz de confusão
    cm=confusion_matrix(y_test, ypred)
    # print(cm)

    return tx_acerto, cm

# List to store the results
results = []

def run_experiment_1():

    strategy = "all_class" # "per_class"
    if sys.argv[1] == "all_class":
        strategy = "all_class"
    elif sys.argv[1] == "per_class":
        strategy = "per_class"

    estimators = [
        # KNeighborsClassifier(n_neighbors=1),
        # KNeighborsClassifier(n_neighbors=3),
        # DecisionTreeClassifier(),
        # GaussianNB(),

        # DissimilarityRNGClassifier(estimator=KNeighborsClassifier(n_neighbors=1), r_per_class=3),
        # DissimilarityRNGClassifier(estimator=KNeighborsClassifier(n_neighbors=3), r_per_class=3),
        # DissimilarityRNGClassifier(estimator=DecisionTreeClassifier(), r_per_class=3),
        # DissimilarityRNGClassifier(estimator=GaussianNB(), r_per_class=3),

        DissimilarityCentroidClassifier(estimator=KNeighborsClassifier(n_neighbors=1), n_clusters=3, strategy=strategy),
        DissimilarityCentroidClassifier(estimator=KNeighborsClassifier(n_neighbors=3), n_clusters=3, strategy=strategy),
        DissimilarityCentroidClassifier(estimator=DecisionTreeClassifier(), n_clusters=3, strategy=strategy),
        DissimilarityCentroidClassifier(estimator=GaussianNB(), n_clusters=3, strategy=strategy),

        DissimilarityIHDClassifier(estimator=KNeighborsClassifier(n_neighbors=1), coef_k=1, din_k=True, k=1, r_size=10, strategy=strategy),
        DissimilarityIHDClassifier(estimator=KNeighborsClassifier(n_neighbors=3), coef_k=1, din_k=True, k=1, r_size=10, strategy=strategy),
        DissimilarityIHDClassifier(estimator=DecisionTreeClassifier(), coef_k=1, din_k=True, k=1, r_size=10, strategy=strategy),
        DissimilarityIHDClassifier(estimator=GaussianNB(), coef_k=1, din_k=True, k=1, r_size=10, strategy=strategy),
    ]

    for estimator in estimators:
        print(estimator)
        experiment_kernel(estimator=estimator, strategy_name=strategy)


def experiment_kernel(estimator, strategy_name):
    # Nomeia o classificador, essa validação é para saber se precisa ou não incluir os parametros
    if 'estimator' in estimator.get_params():
        str_classifier = type(estimator).__name__ # Não inclui os parametros
    else:
        str_classifier = str(estimator) # Inclui os parametros

    for dataset in function_datasets_list:
        X, y = dataset()  # Load the dataset
        name_function_dataset = dataset.__name__
        # print(f"Testing dataset: {name_function_dataset}")

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, test_size=0.5, shuffle=False)

        # Testing random_state from 20 to 50
        for i in range(range_experimento[0], range_experimento[1]):
            # print(f"Using random_state={i}")

            # Check if random_state needs to be set for any estimator
            # Add the estimator's name to the variable
            str_estimator = ""
            if 'estimator' in estimator.get_params():
                str_estimator = str(estimator.estimator)
                if hasattr(estimator.estimator, 'random_state'):
                    estimator.estimator.set_params(random_state=i)
            else:
                str_estimator = "Empty estimator"

            # If the classifier has random_state, set the value
            if hasattr(estimator, 'random_state'):
                estimator.set_params(random_state=i)

            # Test the model
            accuracy, cm = model_test(estimator=estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
            
            # Store the result
            results.append({
                'dataset': name_function_dataset,
                'classifier': str_classifier,
                'estimator': str_estimator,
                'random_state': i,
                'accuracy': accuracy,
                'strategy': strategy_name                
            })
            # print(f"Accuracy: {accuracy}")

    # After completing all experiments, save to a file or display
    df_results = pd.DataFrame(results)
    df_results.to_csv(f'experiment_results_{strategy_name}.csv', index=False)
    # print(f"Results saved to 'experiment_results_{strategy_name}.csv'.")

try:
    run_experiment_1()
except KeyboardInterrupt:
    print("\nExecução interrompida pelo usuário. Encerrando...")

