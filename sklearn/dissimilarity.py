# Author: Pedro Bianchini de Quadros <quadros.pedro@pucpr.edu.br>
#         Gabriel Antonio Gomes de Farias <gomes.farias@pucpr.edu.br>
# Co-Author: Jean Paul Barddal
# License: BSD 3 clause

import warnings
from numbers import Integral, Real
from abc import ABCMeta, abstractmethod

import math
import numpy as np
import pandas as pd
import scipy.sparse as sp
#import imblearn as imb
import heapq
import deslib as dsl
from deslib.util.instance_hardness import kdn_score

from .base import (
    BaseEstimator,
    ClassifierMixin,
    MultiOutputMixin,
    RegressorMixin,
    _fit_context,
)
from .utils import check_random_state
from .utils._param_validation import HasMethods, Interval, StrOptions
from .utils.multiclass import class_distribution
from .utils.random import _random_choice_csc
from .utils.stats import _weighted_percentile
from .utils.validation import (
    _check_sample_weight,
    _num_samples,
    check_array,
    check_consistent_length,
    check_is_fitted,
)
from .neighbors import KNeighborsClassifier
from .cluster import KMeans

class _BaseDissimilarity(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):
    """ Abstract base class for Dissimilarity based estimators"""
    
    def _calc_dist_euclidean(self, p_t, p_r):
        """
        Calculates the Euclidean distance between T and R sets, and thus returning the dissimilarity factor.

        According to this paper:
            The dissimilarity approach: a review
                Yandre M. G.;
                CostaDiego Bertolini;
                Alceu S. Britto Jr.;
                George D. C. Cavalcanti;
                Luiz E. S. Oliveira;

        :param t: Values of X from the training split
        :param r: Values of R selected for random, graph, and clustering
        :return: DataFrame containing the distance matrix.
        """
        T = p_t
        R = p_r

        temp = T - R
        euclid_dist = np.sqrt(np.dot(temp.T, temp))
        return euclid_dist
    
    def _get_dissim_representation(self, X):
        """
        Make the space dissimilarity, using matrix with `numpy`
        """
        if self.instances_X_r is None:
            raise ValueError("R instances not set.")
        
        # Montando uma matriz de dissimilaridade usando o conjunto de testes
        # TODO: A matriz atual não consegue suportar um conjunto de R muito grande.
        dissim_matrix_returned = np.zeros((X.shape[0], self.instances_X_r.shape[0]))
        for i, instances_t in enumerate(X):
            for j, instance_r in enumerate(self.instances_X_r):
                dissim_matrix_returned[i, j] = self._calc_dist_euclidean(instances_t, instance_r)
        
        return dissim_matrix_returned
    
    def predict(self, X):
        """Perform classification on test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted target values for X.
        """

        if self.instances_X_r is None:
            raise ValueError("Missing value R")
        
        dissimilarity_test = self._get_dissim_representation(X)
        return self.estimator.predict(dissimilarity_test)



class DissimilarityRNGClassifier(_BaseDissimilarity):
    """
    A Dissimilarity Classifier model using an random number generator to select a given seed for the subset of R
    """


    """    
    Parameters
    ----------
    random_state : int, RandomState instance or None, default=None
        Controls the randomness to generate the R values.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
 

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,) or list of such arrays
        Unique class labels observed in `y`. For multi-output classification
        problems, this attribute is a list of arrays as each output has an
        independent set of possible classes.
    . . . (Pegar exemplo do dummy)
        
    n_classes_ : int or list of int
        Number of label for each output.
    . . . (Pegar exemplo do dummy)

    See Also
    --------
    . . . (Pegar exemplo do dummy)

    Examples
    --------
    . . . (Pegar exemplo do dummy)

    """

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit", "predict"]), None],
        # "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "strategy": [
            StrOptions({"most_frequent", "prior", "stratified", "uniform", "constant"})
        ],
        "random_state": ["random_state"],
    }

    def __init__(self, estimator=None, *, strategy="prior", random_state=None):
        # Estimators
        self.estimator = estimator

        # Parameters
        self.strategy = strategy
        self.random_state = random_state

        # Atributes
        self.classes_ = None
        self.dissim_matrix_ = None
        self.instances_X_r = None


    def _index_random_choice(self, X, y, random_state):
        """Chosses an random index of X values"""
        n_classes = len(self.classes_)
        n_instances = 1
        result_diff = 0
        list_indexes = []

        # N = Quantidade de classes que serão usadas por cada classe
        n = len(np.unique(y)) * 3

        # TODO: Rever o código usado
        if(n > n_classes):
            n_instances = int(n / n_classes)

        for i, c in enumerate(self.classes_):
            indexes = np.where(c == y)[0]
            quantidade_instancias = len(indexes)

            if(quantidade_instancias < (n_instances + result_diff)):
                choices = random_state.choice(quantidade_instancias, quantidade_instancias, replace=False)
                result_diff += n_instances - quantidade_instancias
            else:
                choices = random_state.choice(quantidade_instancias, n_instances + result_diff, replace=False)
                result_diff = 0
            
            list_indexes.append(indexes[choices])
        
        return np.concatenate(list_indexes)
    
    

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """Fit the baseline classifier.
        TODO: Lembrar de arrumar os comentários
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        ############################# IMPLEMENTAÇÃO TCC #############################
        # Etapas
        # 1. Organizar os parâmetros e validar eles inicialmente:
        """
        # Devolve um random para poder utilizarmos, se for None ele não seta um novo utiliza um aleatório
        random_state = check_random_state(self.random_state) # Exemplo de uso da variavel: random_state.randint(MAX_INT, size=len(self.estimators_))
        self._validate_estimator(self._get_estimator()) # Estimator vai ser complicado implementar, ESTUDAR ISSO!
        # De acordo com a seleção de R fazer tal ação
        """
        self._validate_data(X, cast_to_ndarray=False) # Validação dos dados
        self._strategy = self.strategy # Não sei se ainda vou utilizar isso

        # Classes do problema
        self.classes_ = np.unique(y)

        # 2. Selecionar o confunto "R"
        random_state = check_random_state(self.random_state)

        # Obtem os indices que serão usados
        selected_instances = self._index_random_choice(X, y, random_state)

        self.instances_X_r = X[selected_instances, :]
        # 3. Montar a matriz de dissimilaridade

        if self.instances_X_r is None:
            raise ValueError()
        
        # Ver sobre estimators
        if self.estimator is None:
            self.estimator = KNeighborsClassifier()
        
        # Com os R selecionados podemos montar a matriz identidade
        self.dissim_matrix_ = self._get_dissim_representation(X)

        # Utiliza a matrix de dissimilidade
        self.estimator.fit(self.dissim_matrix_, y)

        return self


class DissimilarityCentroidClassifier(_BaseDissimilarity):

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit", "predict"]), None],
        "random_state": ["random_state"],
    }

    def __init__(self, estimator=None, n_clusters=3, *, random_state=None):
        # Estimators
        self.estimator = estimator
        
        # Parameters
        self.random_state = random_state

        # Atributes
        self.n_clusters = n_clusters
        self.classes_ = None
        self.dissim_matrix_ = None
        self.instances_X_r = None
    
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        
        self._validate_data(X, cast_to_ndarray=False)
        self.classes_ = np.unique(y)
        random_state = check_random_state(self.random_state)

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=random_state)
        kmeans.fit(X)
        centroids = kmeans.cluster_centers_
        self.instances_X_r = centroids

        if self.instances_X_r is None:
            raise ValueError()
        
        if self.estimator is None:
            self.estimator = KNeighborsClassifier()
        
        self.dissim_matrix_ = self._get_dissim_representation(X)
        self.estimator.fit(self.dissim_matrix_, y)

        return self

    
    
class DissimilarityIHDClassifier(_BaseDissimilarity):
    """
    A Dissimilarity Classifier model using instance hardeness threshold for selection of the reference (R) subset 
    """

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit", "predict"]), None],
        # "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "strategy": [
            StrOptions({"most_frequent", "prior", "stratified", "uniform", "constant"})
        ],
        "random_state": ["random_state"],
    }

    def __init__(self, estimator=None, *, strategy="prior", random_state=None, k=None, din_k=True, coef_k=2):
        # Estimators
        self.estimator = estimator

        # Parameters
        self.strategy = strategy
        self.random_state = random_state

        # Atributes
        self.classes_ = None
        self.dissim_matrix_ = None
        self.instances_X_r = None

        # Neighbors
        self.k = k
        self.din_k = din_k

        # Coeficiente p euristica da escala de vizinhos p/tam de k
        self.coef_k = coef_k

    @_fit_context(prefer_skip_nested_validation=True)            
    def fit(self, X, y, sample_weight=None, batch_size=2000, max_r_size=5000, use_batch=True):
        """
        Fit the model with an option to use batching or process the entire dataset at once.

        Parameters:
        - X: Training data
        - y: Training labels
        - batch_size: Size of the batches for processing data in chunks (used if use_batch=True).
        - max_r_size: Maximum number of instances in the reference set R.
        - use_batch: Boolean flag to enable or disable batching (default: True).
        """
    
        # Valida os dados
        self._validate_data(X, cast_to_ndarray=False)
        self._strategy = self.strategy
        
        # 1. Classes do problema
        self.classes_ = np.unique(y)
        
        # 2. Determinar o tamanho de R dinamicamente com base no tamanho do dataset
        n_samples = len(X)
        
        if max_r_size is None:
            # Dinamicamente ajusta o tamanho de R baseado no tamanho do dataset
            if n_samples <= 5000:
                max_r_size = min(n_samples, 800)  # Para datasets pequenos, limite de até 800 instâncias
            elif n_samples <= 100000:
                max_r_size = int(math.sqrt(n_samples) * 2)  # Para datasets médios, usa sqrt(n)
            else:
                max_r_size = int(math.log(n_samples) * 200)  # Para datasets grandes, usa log(n) * 200
            
            print(f"Using dynamic R size: {max_r_size}")
        
        # Inicializa uma lista vazia para as instâncias reamostradas
        X_resampled = []
        y_resampled = []
        
        if use_batch:
            # --- Process with Batching ---
            # Processa os dados em lotes (batches)
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X[start:end]
                y_batch = y[start:end]
                
                # Processa cada classe separadamente para calcular a dureza das instâncias
                for class_label in self.classes_:
                    # Obtém as amostras para a classe atual no batch atual
                    class_indices = np.where(y_batch == class_label)[0]
                    X_class = X_batch[class_indices]
                    y_class = y_batch[class_indices]
                    
                    if len(X_class) == 0:
                        continue  # Pula se não houver instâncias dessa classe no lote
                    
                    # Calcula a dureza das instâncias e os vizinhos para a classe atual
                    s, nx = self._instance_hardness(X_class, y_class)
                    
                    # Seleciona as instâncias mais difíceis (para a classe atual)
                    top_indices = heapq.nlargest(min(len(s), max_r_size // len(self.classes_)), 
                                                range(len(s)), key=lambda i: s[i])
                    
                    # Adiciona as instâncias reamostradas e os rótulos
                    X_resampled.append(X_class[top_indices])
                    y_resampled.append(y_class[top_indices])
        else:
            # --- Process Entire Dataset Without Batching ---
            # Processa cada classe separadamente para calcular a dureza das instâncias sem batching
            for class_label in self.classes_:
                # Obtém as amostras para a classe atual
                class_indices = np.where(y == class_label)[0]
                X_class = X[class_indices]
                y_class = y[class_indices]
                
                if len(X_class) == 0:
                    continue  # Pula se não houver instâncias dessa classe
                
                # Calcula a dureza das instâncias e os vizinhos para a classe atual
                s, nx = self._instance_hardness(X_class, y_class)
                
                # Seleciona as instâncias mais difíceis (para a classe atual)
                top_indices = heapq.nlargest(min(len(s), max_r_size // len(self.classes_)), 
                                            range(len(s)), key=lambda i: s[i])
                
                # Adiciona as instâncias reamostradas e os rótulos
                X_resampled.append(X_class[top_indices])
                y_resampled.append(y_class[top_indices])
        
        # Converte as listas para arrays após processar cada classe
        X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)
        
        # Limita o tamanho total de R se necessário
        if len(X_resampled) > max_r_size:
            top_indices = np.random.choice(len(X_resampled), max_r_size, replace=False)
            X_resampled = X_resampled[top_indices]
            y_resampled = y_resampled[top_indices]
        
        # Armazena as instâncias reamostradas como o conjunto de referência R
        self.instances_X_r = X_resampled

        # Monta a matriz de dissimilaridade para todo o conjunto de dados de treinamento
        self.dissim_matrix_ = self._get_dissim_representation(X)
        
        # Ajusta o estimador utilizando a matriz de dissimilaridade
        self.estimator.fit(self.dissim_matrix_, y)
        
        return self


    def predict(self, X):
        """Perform classification on test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted target values for X.
        """

        if self.instances_X_r is None:
            raise ValueError("Missing value R")
        
        dissimilarity_test = self._get_dissim_representation(X)
        return self.estimator.predict(dissimilarity_test)
    
    def _instance_hardness(self, X, y):
        """
        Author: Gabriel Antonio Gomes de Farias

        Use: 
            Calculates the hardness of a instance pertaining to a class and returns a array with each sample hardness value.


        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        s : instance-hardness score (float)
        nx : neighbor
        """

        n_samples = len(X)
    
        if self.din_k:
            # Dinamicamente ajusta k com base no tamanho do dataset
            if n_samples > 5000:
                self.coef_k = 3  # Para datasets muito grandes
            elif n_samples > 1000:
                self.coef_k = 2  # Para datasets médios
            else:
                self.coef_k = 1.5  # Para pequenos datasets
            
            # Ajuste dinâmico de k com um limite superior
            if n_samples > 1000:
                self.k = min(50, int(math.log(n_samples) * self.coef_k))  # Limita k a 50 no máximo
            else:
                self.k = int(math.sqrt(n_samples) * self.coef_k)
            print(f"Using dynamic k: {self.k}")
        else:
            if self.k is None:
                raise ValueError("k must be set when din_k is False.")
            print(f"Using manually set k: {self.k}")
        
        # Calcula a dureza das instâncias com a pontuação K-Disagreeing Neighbors (KDN)
        s, nx = kdn_score(X, y, k=self.k)
        
        # Retorna os escores de dureza e as informações dos vizinhos
        return s, nx