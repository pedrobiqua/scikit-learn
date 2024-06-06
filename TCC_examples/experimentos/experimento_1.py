import numpy as np
import sys

###### LIB ARFF_CONVERT DATASETS ######
from dataset_arff.arffconvert import load_SEAGenerator_test_f2_f4 # Carrega a base de dados que está no formato arff para numpy X e Y
from dataset_arff.arffconvert import load_AgrawalGenerator_test_mode_f2_f9 # Carrega a base de dados que está no formato arff para numpy X e Y
from dataset_arff.arffconvert import load_AssetNegotiationGenerator_f1_f5 # Carrega a base de dados que está no formato arff para numpy X e Y
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

import pandas as pd
import re

# Função responsavel por rodar os algoritmos
def model_test(estimator, X_train, y_train, X_test, y_test):
    
    estimator.fit(X_train, y_train)

    ypred = estimator.predict(X_test)
    tx_acerto = accuracy_score(y_test, ypred)
    print(tx_acerto)

    # Matriz de confusão
    cm=confusion_matrix(y_test, ypred)
    print(cm)

    return tx_acerto, cm

def experimento_1():
    try:
        # Vamos testar o experimento 30 vezes, usando seeds de 20 a 50
        range_experimento = [20, 50]

        # Datasets que iremos utilizar
        function_datasets_list = [load_AssetNegotiationGenerator_f1_f5, load_AgrawalGenerator_test_mode_f2_f9, load_SEAGenerator_test_f2_f4]
        
        # Montagem da saida
        results_score = {
            'RandomState': list(range(range_experimento[0], range_experimento[1])) * len(function_datasets_list),
            'Dataset' : list() # Será armazenado o dataset utilizado
        } # cada uma dessas chaves será a coluna

        # Todos os classificadores que estamos utilizando
        classifiers = ["KNeighborsClassifier(n_neighbors=1)", 
                         "KNeighborsClassifier(n_neighbors=3)", 
                         "DecisionTreeClassifier()",
                         "GaussianNB()",
                         "DissimilarityRNGClassifier(KNeighborsClassifier(n_neighbors=1))",
                         "DissimilarityRNGClassifier(KNeighborsClassifier(n_neighbors=3))",
                         "DissimilarityRNGClassifier(DecisionTreeClassifier())",
                         "DissimilarityRNGClassifier(GaussianNB())"
                    ]

        # Cria uma chave com uma lista para cada classificador usado 
        for clf_name in classifiers:
            results_score[clf_name] = []
        
        # Testa todos os datasets
        for dataset in function_datasets_list:
            X, y = dataset()
            name_function_dataset = dataset.__name__
            results_score['Dataset'].extend([name_function_dataset] * (range_experimento[1] - range_experimento[0]))

            print(f"Testando o dataset: {name_function_dataset}")

            for i in range(range_experimento[0], range_experimento[1]): # Vamos testar o random_state de 20 a 50
                # Classificadores a serem testados
                print(f"random_state={i}")
                estimators= [KNeighborsClassifier(n_neighbors=1), 
                            KNeighborsClassifier(n_neighbors=3), 
                            DecisionTreeClassifier(random_state=i), 
                            GaussianNB()]
                
                # 50% treino e 50% para teste
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, test_size=0.5, shuffle=False)

                for estimator in estimators:
                    tx_acerto, cm = model_test(estimator=estimator, X_train=X_train, y_train=y_train,X_test=X_test, y_test=y_test)

                    # Remove o random_state=<value>
                    cleaned_str = re.sub(r'random_state\s*=\s*\d+\s*,?', '', str(estimator))
                    cleaned_str = re.sub(r',\s*\)', ')', cleaned_str)
                    
                    results_score[cleaned_str].append(tx_acerto)


                for estimator in estimators:
                    tx_acerto, cm = model_test(estimator=DissimilarityRNGClassifier(estimator=estimator, random_state=i), X_train=X_train, y_train=y_train,X_test=X_test, y_test=y_test)
                    
                    # Remove o random_state=<value>
                    name_estimator = f"DissimilarityRNGClassifier({str(estimator)})"
                    cleaned_str = re.sub(r'random_state\s*=\s*\d+\s*,?', '', name_estimator)
                    cleaned_str = re.sub(r',\s*\)', ')', cleaned_str)
                    
                    results_score[cleaned_str].append(tx_acerto)

        # Após o loop, organize e salve os resultados em um arquivo Excel
        classifiers_names = ['RDN_STATE', 'DATASET', 'KNN_1', 'KNN_3', 'DT', 'GNB', 'Diss_RNG_KNN_1', 'Diss_RNG_KNN_3', 'Diss_RNG_DT', 'Diss_RNG_GNB']

        # Cria o dataframe com os resultados das acurácias
        df = pd.DataFrame(results_score)

        df.columns = classifiers_names # Muda o nome das colunas para a lista que já temos

        # Salvar depois os resultados em um excel para validar e analisar melhor
        df.to_csv('results_50_50.csv', index=False)

    except KeyboardInterrupt:
        print(f"Tecla Crtl + c precionada!")

#Inicializa os experimento
experimento_1()