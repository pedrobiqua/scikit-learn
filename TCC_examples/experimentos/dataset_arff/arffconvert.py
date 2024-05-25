from scipy.io import arff
import numpy as np
from os import getcwd
#import pandas as pd
import os

# Author: Pedro Bianchini de Quadros      <quadros.pedro@pucpr.edu.br>
#         Gabriel Antonio Gomes de Farias <gomes.farias@pucpr.edu.br>
# Co-Author: Jean Paul Barddal
#  ESCOLA POLITÉCNICA - PUCPR - CIÊNCIA DA COMPUTAÇÃO

#IMPORTANTE: LEMBRE DE ARRUMAR O PATH SE FOR USAR O CÓDIGO
# DATASETS GERADOS PELO MOA

DATA_MODULE = "datasets"

# Exemplo de como chamar
def load_SEAGenerator(*, return_X_y=True, frame=False):
    # Caminho para o arquivo arff
    # path_SEAGenerator = os.path.join(DATA_MODULE, 'arff', 'SEA_base.arff')
    # TODO: Fazer isso de forma mais inteligente
    path_SEAGenerator = "/home/pedro/projects/scikit-learn/TCC_examples/datasets/arff/SEA_base.arff"
    # Carrega arquivo arff
    data, meta = arff.loadarff(f"{path_SEAGenerator}")
    # Faz os tratamentos especificos
    target = np.array(data.tolist(), dtype=object)[:, -1].astype('U6')
    flat_data = np.array(data.tolist(), dtype=object)[:,:-1]

    # TODO: Montar dataframe, caso precise!
    if frame:
        print("Montar o dataframe")
        return flat_data, target
    
    # Devolve X, y | Ou seja data e target
    if return_X_y:
        return flat_data, target    
    
    # TODO: Depois montar uma classe, inspirada na Bunch do Sckit-learning
    return flat_data, target

def load_SEAGenerator_test_mode(*, return_X_y=True, frame=False):
    
    # Config moa: WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.SEAGenerator -b) -d (generators.SEAGenerator -f 3 -b) -p 50000 -w 1) -f (C:\Users\pedro\Área de Trabalho\Projetos\TCC\SEA_base_drift_test_1.arff) -m 100000
    
    path_SEAGenerator = "/home/pedro/projects/scikit-learn/TCC_examples/datasets/arff/SEA_base_drift_test_1.arff"
    # Carrega arquivo arff
    data, meta = arff.loadarff(f"{path_SEAGenerator}")
    # Faz os tratamentos especificos
    target = np.array(data.tolist(), dtype=object)[:, -1].astype('U6')
    flat_data = np.array(data.tolist(), dtype=object)[:,:-1]
    return flat_data, target

def load_SEAGenerator_test_f2_f4(*, return_X_y=True, frame=False):
    # WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.SEAGenerator -f 2 -i 10 -b) -d (generators.SEAGenerator -f 4 -i 5 -b) -p 50000 -w 1 -r 10) -f \\wsl.localhost\Ubuntu\home\pedro\projects\dataset_arff_covert\datasets\arff\SEAGenerator_base_drifit_f2_f4.arff -m 100000
    path_SEAGenerator = "C:/Users/Gabz/Documents/scikit-learn-1/TCC_examples/datasets/arff/SEAGenerator_base_drifit_f2_f4.arff"
    # Carrega arquivo arff
    data, meta = arff.loadarff(f"{path_SEAGenerator}")
    # Faz os tratamentos especificos
    target = np.array(data.tolist(), dtype=object)[:, -1].astype('U6')
    flat_data = np.array(data.tolist(), dtype=object)[:,:-1]
    return flat_data, target

# TODO: O dataset ainda precisa fazer os tratamentos especificos
def load_AgrawalGenerator_test_mode(*, return_X_y=True, frame=False):
    # Config moa: WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.AgrawalGenerator -b) -d (generators.AgrawalGenerator -f 5 -b) -p 50000 -w 1) -f (C:\Users\pedro\Área de Trabalho\Projetos\TCC\Agrawal_drift_test_1.arff) -m 100000
    path = "/home/pedro/projects/scikit-learn/TCC_examples/datasets/arff/Agrawal_drift_test_1.arff"
    # Carrega arquivo arff
    data, meta = arff.loadarff(f"{path}")
    print(data)
    # Faz os tratamentos especificos
    target = np.array(data.tolist(), dtype=object)[:, -1].astype('U6')
    flat_data = np.array(data.tolist(), dtype=object)[:,:-1]
    return flat_data, target

# TODO: O dataset ainda precisa fazer os tratamentos especificos
def load_AgrawalGenerator_test_mode_f2_f9(*, return_X_y=True, frame=False):
    # Config moa: WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.AgrawalGenerator -f 2 -i 10 -b) -d (generators.AgrawalGenerator -f 10 -b) -p 50000 -w 1 -r 10) -f \\wsl.localhost\Ubuntu\home\pedro\projects\dataset_arff_covert\datasets\arff\AgrawalGenerator_base_drifit_f2_f9.arff -m 100000
    path = "/home/pedro/projects/scikit-learn/TCC_examples/datasets/arff/AgrawalGenerator_base_drifit_f2_f9.arff"
    # Carrega arquivo arff
    data, meta = arff.loadarff(f"{path}")
    # print(data)
    # Faz os tratamentos especificos
    target = np.array(data.tolist(), dtype=object)[:, -1].astype('U6')
    flat_data = np.array(data.tolist(), dtype=object)[:,:-1]
    return flat_data, target
