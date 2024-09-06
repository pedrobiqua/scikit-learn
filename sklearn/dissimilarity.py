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
        # Substituir isso daqui por representative_set_size
        n = len(np.unique(y)) * 3 # Isso é por classe

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

    def score(self, X, y, sample_weight=None):
        """Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : None or array-like of shape (n_samples, n_features)
            Test samples. Passing None as test samples gives the same result
            as passing real test samples, since DummyClassifier
            operates independently of the sampled observations.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) w.r.t. y.
        """
        if X is None:
            X = np.zeros(shape=(len(y), 1))
        return super().score(X, y, sample_weight)


class DissimilarityCentroidClassifier(_BaseDissimilarity):

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit", "predict"]), None],
        "random_state": ["random_state"],
    }


    def _kmeans_per_class(self, X, y):
        """
        Separate instances per class and fit KMeans with the selected classes.
        """
        centroids_list = []
        for n_class in self.classes_:
            instances = X[y == n_class]
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
            kmeans.fit(instances)
            centroids = kmeans.cluster_centers_
            centroids_list.append(centroids)
        centroids_array = np.vstack(centroids_list)
        
        return centroids_array

            

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

        self.instances_X_r = self._kmeans_per_class(X, y)

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

        # Unique Attributes
        self.k = None


    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        self._validate_data(X, cast_to_ndarray=False) # Validação dos dados
        self._strategy = self.strategy # Não sei se ainda vou utilizar isso

        # 1. Classes do problema
        self.classes_ = np.unique(y)

        # 2. Calcula dureza das classes
        s, nx = self._instance_hardness(X,y)

        # 3. Reamostragem
        # Adicionar aqui a quantidade de R
        X_resampled = np.array(heapq.nlargest(len(nx[0]), s))

        # 4. Reamostrage treino
        self.instances_X_r = X_resampled

        # Com os R selecionados podemos montar a matriz identidade
        self.dissim_matrix_ = self._get_dissim_representation(X)

        # Utiliza a matrix de dissimilidade
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


        @param: array X (Array with samples) ; y labels ;  int k ("Safe" value of neighbors to estimate certain region)
        @return: 2D tuple float score:int neighbor
        """

        #Calculate instance hardness on the training data
        # Calculate instance hardness on the training data
        k = 3
        s, nx = kdn_score(X,y, k=k)   
        return s, nx
        # X_R_subset is now ready for use in further analysis, such as building a dissimilarity matrix

    def predict_proba(self, X):
        """
        Return probability estimates for the test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        P : ndarray of shape (n_samples, n_classes) or list of such arrays
            Returns the probability of the sample for each class in
            the model, where classes are ordered arithmetically, for each
            output.
        """
        check_is_fitted(self)

        # numpy random_state expects Python int and not long as size argument
        # under Windows
        n_samples = _num_samples(X)
        rs = check_random_state(self.random_state)

        n_classes_ = self.n_classes_
        classes_ = self.classes_
        class_prior_ = self.class_prior_
        constant = self.constant
        if self.n_outputs_ == 1:
            # Get same type even for self.n_outputs_ == 1
            n_classes_ = [n_classes_]
            classes_ = [classes_]
            class_prior_ = [class_prior_]
            constant = [constant]

        P = []
        for k in range(self.n_outputs_):
            if self._strategy == "most_frequent":
                ind = class_prior_[k].argmax()
                out = np.zeros((n_samples, n_classes_[k]), dtype=np.float64)
                out[:, ind] = 1.0
            elif self._strategy == "prior":
                out = np.ones((n_samples, 1)) * class_prior_[k]

            elif self._strategy == "stratified":
                out = rs.multinomial(1, class_prior_[k], size=n_samples)
                out = out.astype(np.float64)

            elif self._strategy == "uniform":
                out = np.ones((n_samples, n_classes_[k]), dtype=np.float64)
                out /= n_classes_[k]

            elif self._strategy == "constant":
                ind = np.where(classes_[k] == constant[k])
                out = np.zeros((n_samples, n_classes_[k]), dtype=np.float64)
                out[:, ind] = 1.0

            P.append(out)

        if self.n_outputs_ == 1:
            P = P[0]

        return P

    def predict_log_proba(self, X):
        """
        Return log probability estimates for the test vectors X.

        Parameters
        ----------
        X : {array-like, object with finite length or shape}
            Training data.

        Returns
        -------
        P : ndarray of shape (n_samples, n_classes) or list of such arrays
            Returns the log probability of the sample for each class in
            the model, where classes are ordered arithmetically for each
            output.
        """
        proba = self.predict_proba(X)
        if self.n_outputs_ == 1:
            return np.log(proba)
        else:
            return [np.log(p) for p in proba]

    def _more_tags(self):
        return {
            "poor_score": True,
            "no_validation": True,
            "_xfail_checks": {
                "check_methods_subset_invariance": "fails for the predict method",
                "check_methods_sample_order_invariance": "fails for the predict method",
            },
        }
    
