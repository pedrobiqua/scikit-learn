from scipy.io import arff
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os

# Author: Pedro Bianchini de Quadros      <quadros.pedro@pucpr.edu.br>
#         Gabriel Antonio Gomes de Farias <gomes.farias@pucpr.edu.br>
# Co-Author: Jean Paul Barddal
#  ESCOLA POLITÉCNICA - PUCPR - CIÊNCIA DA COMPUTAÇÃO

# DATASETS GERADOS PELO MOA

DATA_MODULE = "datasets"

def load_SEAGenerator():
    """
    FUNÇÃO TESTE USADA COMO EXEMPLO
    """
    # Caminho para o arquivo arff
    # path_SEAGenerator = os.path.join(DATA_MODULE, 'arff', 'SEA_base.arff')
    path_SEAGenerator = "../../datasets/arff/SEA_base_drift_test_1.arff"
    flat_data, target = format_dataset(path=path_SEAGenerator)
    return flat_data, target

def load_SEAGenerator_test_mode(*, return_X_y=True, frame=False):
    
    # Config moa: WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.SEAGenerator -b) -d (generators.SEAGenerator -f 3 -b) -p 50000 -w 1) -f (C:\Users\pedro\Área de Trabalho\Projetos\TCC\SEA_base_drift_test_1.arff) -m 100000
    
    path_SEAGenerator = "../../datasets/arff/SEA_base_drift_test_1.arff"
    flat_data, target = format_dataset(path=path_SEAGenerator)
    return flat_data, target

def load_SEAGenerator_test_f2_f4(*, return_X_y=True, frame=False):
    # WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.SEAGenerator -f 2 -i 10 -b) -d (generators.SEAGenerator -f 4 -i 5 -b) -p 50000 -w 1 -r 10) -f \\wsl.localhost\Ubuntu\home\pedro\projects\dataset_arff_covert\datasets\arff\SEAGenerator_base_drifit_f2_f4.arff -m 100000
    path_SEAGenerator = "../../datasets/arff/SEAGenerator_base_drifit_f2_f4.arff"
    flat_data, target = format_dataset(path=path_SEAGenerator)
    return flat_data, target

# TODO: O dataset ainda precisa fazer os tratamentos especificos
def load_AgrawalGenerator_test_mode(*, return_X_y=True, frame=False):
    # Config moa: WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.AgrawalGenerator -b) -d (generators.AgrawalGenerator -f 5 -b) -p 50000 -w 1) -f (C:\Users\pedro\Área de Trabalho\Projetos\TCC\Agrawal_drift_test_1.arff) -m 100000
    path = "../../datasets/arff/Agrawal_drift_test_1.arff"
    flat_data, target = format_dataset(path=path)
    return flat_data, target

# TODO: O dataset ainda precisa fazer os tratamentos especificos
def load_AgrawalGenerator_test_mode_f2_f9(*, return_X_y=True, frame=False):
    # Config moa: WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.AgrawalGenerator -f 2 -i 10 -b) -d (generators.AgrawalGenerator -f 10 -b) -p 50000 -w 1 -r 10) -f \\wsl.localhost\Ubuntu\home\pedro\projects\dataset_arff_covert\datasets\arff\AgrawalGenerator_base_drifit_f2_f9.arff -m 100000
    path = "../../datasets/arff/AgrawalGenerator_base_drifit_f2_f9.arff"
    flat_data, target = format_dataset(path=path)
    return flat_data, target

def load_AssetNegotiationGenerator_f1_f5(*, return_X_y=True, frame=False):
    # WriteStreamToARFFFile -s (ConceptDriftStream -s generators.AssetNegotiationGenerator -d (generators.AssetNegotiationGenerator -f 5) -p 50000 -w 1 -r 10) -f \\wsl.localhost\Ubuntu\home\pedro\projects\dataset_arff_covert\datasets\arff\AssetNegotiationGenerator_base_drifit_f1_f5.arff -m 100000
    path = "../../datasets/arff/AssetNegotiationGenerator_base_drifit_f1_f5.arff"
    flat_data, target = format_dataset(path=path)
    return flat_data, target

def format_dataset(path):
    # Caminho relativo das bases
    folders = path.split("/")
    base_dir = os.path.dirname(__file__)
    join_path = os.path.join(base_dir, *folders)
    absolute_path = os.path.abspath(join_path)
    print(absolute_path)

    # Carrega arquivo arff
    data, meta = arff.loadarff(absolute_path)
    # print(data)
    # Faz os tratamentos especificos
    target = np.array(data.tolist(), dtype=object)[:, -1].astype('U6')
    flat_data = np.array(data.tolist(), dtype=object)[:,:-1]

    encoder   = OneHotEncoder(handle_unknown="ignore")
    flat_data = pd.DataFrame(flat_data)
    flat_data = flat_data.convert_dtypes()
    flat_cat  = flat_data.select_dtypes(exclude="number")
    flat_num  = flat_data.select_dtypes(include="number")
    X_encoded = pd.DataFrame(encoder.fit_transform(flat_cat).toarray())
    flat_data = pd.concat([flat_num, X_encoded], axis=1).to_numpy()

    # Retorna os valores formatados
    return flat_data, target