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
    """ 
    Abstract base class for dissimilarity-based estimators.
    
    This class provides the common functionalities for dissimilarity-based classifiers, 
    such as calculating the dissimilarity matrix using Euclidean distance and performing classification 
    based on the dissimilarity space.
    
    The approach is based on the paper:
    
        The Dissimilarity Approach: A Review
        Yandre M. G. Costa, Diego Bertolini, Alceu S. Britto Jr., 
        George D. C. Cavalcanti, Luiz E. S. Oliveira
    """

    def _calc_dist_euclidean(self, p_t, p_r):
        """
        Calculate the Euclidean distance between the sets T (test instances) and R (reference instances),
        thus returning the dissimilarity factor between each test and reference instance.
        
        Parameters
        ----------
        p_t : array-like of shape (n_features,)
            Values of X (test instances) from the dataset.

        p_r : array-like of shape (n_features,)
            Values of R (reference instances) selected for random, graph, or clustering-based methods.

        Returns
        -------
        euclid_dist : float
            The Euclidean distance between the test and reference instance.
        """
        T = p_t  # Test instance
        R = p_r  # Reference instance

        # Calculate Euclidean distance as the square root of the dot product of the difference
        temp = T - R
        euclid_dist = np.sqrt(np.dot(temp.T, temp))
        return euclid_dist
    
    def _get_dissim_representation(self, X):
        """
        Build the dissimilarity matrix for the given test instances `X` using the reference subset `R`.

        This function computes the dissimilarity representation by calculating the distance of each test 
        instance in `X` to each reference instance in `R` using the Euclidean distance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data for which to compute the dissimilarity matrix.

        Returns
        -------
        dissim_matrix_returned : ndarray of shape (n_samples, n_reference_samples)
            The dissimilarity matrix representing distances between test instances and reference points.

        Raises
        ------
        ValueError
            If the reference subset `R` has not been set.
        """
        # Check if reference instances have been set
        if self.instances_X_r is None:
            raise ValueError("R instances not set.")
        
        # Initialize the dissimilarity matrix
        dissim_matrix_returned = np.zeros((X.shape[0], self.instances_X_r.shape[0]))

        # Compute the dissimilarity matrix by calculating the Euclidean distance
        for i, instances_t in enumerate(X):
            for j, instance_r in enumerate(self.instances_X_r):
                dissim_matrix_returned[i, j] = self._calc_dist_euclidean(instances_t, instance_r)
        
        return dissim_matrix_returned
    
    def predict(self, X):
        """
        Perform classification on the given test vectors `X` based on the dissimilarity space.

        The dissimilarity matrix is computed between the test data `X` and the reference set `R`.
        The estimator is then used to make predictions based on this dissimilarity representation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted target values for each test instance in `X`.

        Raises
        ------
        ValueError
            If the reference set `R` has not been set before calling `predict`.
        """
        # Ensure reference instances (R) are set
        if self.instances_X_r is None:
            raise ValueError("Missing value R")
        
        # Compute the dissimilarity matrix for the test data
        dissimilarity_test = self._get_dissim_representation(X)

        # Use the estimator to predict based on the dissimilarity matrix
        return self.estimator.predict(dissimilarity_test)


class DissimilarityRNGClassifier(_BaseDissimilarity):
    """
    A Dissimilarity Classifier model using a random number generator (RNG) to select a subset of reference instances (R).
    The selected reference instances are used to compute the dissimilarity matrix, which is then used to train the classifier.

    Parameters
    ----------
    estimator : estimator object, default=None
        The base estimator used to classify instances after the dissimilarity transformation.
        If None, defaults to KNeighborsClassifier.

    r_per_class : int, default=3
        The number of reference points to be selected per class. The total number of reference points (R)
        will be the number of classes multiplied by `r_per_class`.

    strategy : {"most_frequent", "prior", "stratified", "uniform", "constant"}, default="prior"
        Strategy used to determine how the reference points are selected and used:
        - "most_frequent": Select the most frequent labels.
        - "prior": Use prior knowledge or distribution of labels.
        - "stratified": Select instances in a stratified manner based on the class distribution.
        - "uniform": Select reference points uniformly at random.
        - "constant": Select a constant set of reference points.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness used to select the reference points.
        Pass an integer for reproducible results across multiple runs.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels observed in the target labels `y`.

    instances_X_r : ndarray of shape (n_reference_samples, n_features)
        The reference subset (R) selected using the random number generator.

    dissim_matrix_ : ndarray of shape (n_samples, n_reference_samples)
        The dissimilarity matrix computed between the input data and the reference points.

    See Also
    --------
    DummyClassifier : A simple baseline classifier that makes predictions using simple rules.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from sklearn.dissimilarity import DissimilarityRNGClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = DissimilarityRNGClassifier(estimator=KNeighborsClassifier(), r_per_class=5, random_state=42)
    >>> clf.fit(X, y)
    """

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit", "predict"]), None],
        "strategy": [
            StrOptions({"most_frequent", "prior", "stratified", "uniform", "constant"})
        ],
        "random_state": ["random_state"],
    }

    def __init__(self, estimator=None, r_per_class=3, *, strategy="prior", random_state=None):
        """
        Initialize the DissimilarityRNGClassifier.

        Parameters
        ----------
        estimator : estimator object, default=None
            The base classifier to be used after dissimilarity transformation. Defaults to KNeighborsClassifier.

        r_per_class : int, default=3
            Number of reference points to be selected per class.

        strategy : {"most_frequent", "prior", "stratified", "uniform", "constant"}, default="prior"
            Strategy for selecting reference points using random selection.

        random_state : int, RandomState instance or None, default=None
            Controls the randomness in selecting reference points.
        """
        self.estimator = estimator
        self.r_per_class = r_per_class
        self.strategy = strategy
        self.random_state = random_state

        # Attributes
        self.classes_ = None
        self.dissim_matrix_ = None
        self.instances_X_r = None
        self.n = None

    def _index_random_choice(self, X, y, random_state):
        """
        Select random indices for reference points based on the provided random state.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : array-like of shape (n_samples,)
            Target labels.

        random_state : RandomState
            A random number generator instance for reproducibility.

        Returns
        -------
        indices : ndarray of shape (n_reference_samples,)
            Indices of the selected reference points.
        """
        n_classes = len(self.classes_)
        n_instances = 1
        result_diff = 0
        list_indexes = []

        # Total number of reference points to select (n = classes * r_per_class)
        n = self.n
        
        if n > n_classes:
            n_instances = int(n / n_classes)

        for c in self.classes_:
            indexes = np.where(c == y)[0]
            quantidade_instancias = len(indexes)

            if quantidade_instancias < (n_instances + result_diff):
                choices = random_state.choice(quantidade_instancias, quantidade_instancias, replace=False)
                result_diff += n_instances - quantidade_instancias
            else:
                choices = random_state.choice(quantidade_instancias, n_instances + result_diff, replace=False)
                result_diff = 0
            
            list_indexes.append(indexes[choices])
        
        return np.concatenate(list_indexes)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """
        Fit the classifier by selecting reference points using a random number generator and 
        computing the dissimilarity matrix.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target labels.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        self : object
            Fitted classifier.
        """
        # Validate input data
        self._validate_data(X, cast_to_ndarray=False)

        # 1. Get the unique classes from the target labels
        self.classes_ = np.unique(y)
        self.n = len(self.classes_) * self.r_per_class

        # 2. Select the reference subset (R) using random number generator
        random_state = check_random_state(self.random_state)
        selected_instances = self._index_random_choice(X, y, random_state)
        self.instances_X_r = X[selected_instances, :]

        # 3. Compute the dissimilarity matrix
        if self.instances_X_r is None:
            raise ValueError("No reference subset selected.")

        # 4. If no estimator is provided, default to KNeighborsClassifier
        if self.estimator is None:
            self.estimator = KNeighborsClassifier()

        # Compute dissimilarity matrix for the entire dataset
        self.dissim_matrix_ = self._get_dissim_representation(X)

        # Fit the base estimator using the dissimilarity matrix
        self.estimator.fit(self.dissim_matrix_, y)

        return self


class DissimilarityCentroidClassifier(_BaseDissimilarity):
    """
    A classifier based on centroids obtained through K-Means clustering, which computes the dissimilarity between
    instances in the feature space. This classifier allows for different strategies to generate the reference 
    points (centroids) either per class or for all classes.

    Parameters
    ----------
    estimator : estimator object, default=None
        The base estimator used to classify instances after the dissimilarity transformation.
        If None, defaults to KNeighborsClassifier.

    n_clusters : int, default=3
        The number of clusters to form using KMeans for each class or for the entire dataset, depending on the strategy.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of KMeans clustering and reference point selection.
        Pass an int for reproducible results.

    strategy : {"per_class", "all_class"}, default="per_class"
        The strategy to apply for centroid computation:
        - "per_class": Perform KMeans clustering separately for each class.
        - "all_class": Perform KMeans clustering on the entire dataset, ignoring class labels.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels observed in `y`.

    dissim_matrix_ : ndarray of shape (n_samples, n_clusters)
        Dissimilarity matrix computed between the input data and the reference points (centroids).

    instances_X_r : ndarray of shape (n_clusters, n_features)
        The reference points (centroids) obtained from KMeans clustering.

    See Also
    --------
    KNeighborsClassifier : The default classifier used if no estimator is provided.
    KMeans : KMeans clustering used to generate reference points.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from sklearn.dissimilarity import DissimilarityCentroidClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = DissimilarityCentroidClassifier(estimator=KNeighborsClassifier(), n_clusters=3, strategy="per_class")
    >>> clf.fit(X, y)
    >>> y_pred = clf.predict(X)
    """

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit", "predict"]), None],
        "random_state": ["random_state"],
        "strategy": [StrOptions({"per_class", "all_class"})],
    }

    def __init__(self, estimator=None, n_clusters=3, *, random_state=None, strategy="per_class"):
        """
        Initialize the DissimilarityCentroidClassifier.

        Parameters
        ----------
        estimator : estimator object, default=None
            The base classifier to be used after dissimilarity transformation. Defaults to KNeighborsClassifier.

        n_clusters : int, default=3
            Number of clusters for KMeans.

        random_state : int, RandomState instance or None, default=None
            Controls the randomness of clustering and reference point selection.

        strategy : {"per_class", "all_class"}, default="per_class"
            The strategy for centroid computation.
        """
        # Initialize the estimator
        self.estimator = estimator
        
        # Set the parameters
        self.random_state = random_state
        self.strategy = strategy

        # Initialize attributes
        self.n_clusters = n_clusters
        self.classes_ = None
        self.dissim_matrix_ = None
        self.instances_X_r = None

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """
        Fit the DissimilarityCentroidClassifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target labels.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        self : object
            Fitted classifier.
        """
        # Validate input data
        self._validate_data(X, cast_to_ndarray=False)
        self.classes_ = np.unique(y)  # Store unique classes

        # Select the strategy to compute the reference points (centroids)
        if self.strategy == "per_class":
            self.instances_X_r = self._kmeans_per_class(X, y, self.random_state)
        elif self.strategy == "all_class":
            self.instances_X_r = self._kmeans_all_class(X, self.random_state)

        if self.instances_X_r is None:
            raise ValueError("Reference points (centroids) not set.")

        # Use KNeighborsClassifier as the default estimator if none is provided
        if self.estimator is None:
            self.estimator = KNeighborsClassifier()

        # Compute the dissimilarity matrix
        self.dissim_matrix_ = self._get_dissim_representation(X)

        # Fit the estimator on the dissimilarity matrix
        self.estimator.fit(self.dissim_matrix_, y)

        return self

    def _kmeans_per_class(self, X, y, random_state):
        """
        Apply KMeans clustering to each class separately to obtain centroids for each class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        y : array-like of shape (n_samples,)
            The target labels.

        random_state : int, RandomState instance or None
            Random state for KMeans clustering.

        Returns
        -------
        centroids_array : ndarray of shape (n_clusters_per_class * n_classes, n_features)
            The centroids obtained from KMeans clustering for each class.
        """
        centroids_list = []
        # Iterate over each class and apply KMeans clustering
        for n_class in self.classes_:
            instances = X[y == n_class]  # Select instances of the current class
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=random_state)
            kmeans.fit(instances)
            centroids = kmeans.cluster_centers_  # Centroids for the current class
            centroids_list.append(centroids)
        
        # Stack all centroids into a single array
        centroids_array = np.vstack(centroids_list)
        return centroids_array

    def _kmeans_all_class(self, X, random_state):
        """
        Apply KMeans clustering to the entire dataset, ignoring class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        random_state : int, RandomState instance or None
            Random state for KMeans clustering.

        Returns
        -------
        centroids : ndarray of shape (n_clusters, n_features)
            The centroids obtained from KMeans clustering on the entire dataset.
        """
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=random_state)
        kmeans.fit(X)
        centroids = kmeans.cluster_centers_  # Centroids for the entire dataset
        return centroids


class DissimilarityIHDClassifier(_BaseDissimilarity):
    """
    A classifier that selects a reference subset based on instance hardness thresholds. The reference subset
    (R) is determined using the hardest-to-classify instances, computed either per class or across all classes.

    Parameters
    ----------
    estimator : estimator object, default=None
        The base estimator used to classify instances after the dissimilarity transformation.
        If None, defaults to KNeighborsClassifier.

    strategy : {"per_class", "all_class"}, default="per_class"
        Strategy to select the reference instances:
        - "per_class": Select the hardest instances separately for each class.
        - "all_class": Select the hardest instances from the entire dataset, without considering class labels.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the instance selection process and reference point determination.

    k : int, default=None
        The number of neighbors used in the instance hardness calculation. If `din_k` is True, `k` is dynamically
        adjusted based on the dataset size.

    din_k : bool, default=True
        Whether to dynamically adjust `k` based on the size of the dataset.

    coef_k : float, default=2
        Coefficient used to scale the number of neighbors `k` when `din_k` is True.

    r_size : int, default=10
        The number of hardest instances to retain in the reference subset.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels observed in `y`.

    dissim_matrix_ : ndarray of shape (n_samples, r_size)
        Dissimilarity matrix computed between the input data and the reference points (hard instances).

    instances_X_r : ndarray of shape (r_size, n_features)
        The reference instances (the hardest-to-classify ones) selected during fitting.

    See Also
    --------
    KNeighborsClassifier : The default classifier used if no estimator is provided.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from sklearn.dissimilarity import DissimilarityIHDClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = DissimilarityIHDClassifier(estimator=KNeighborsClassifier(), r_size=15)
    >>> clf.fit(X, y)
    >>> y_pred = clf.predict(X)
    """

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit", "predict"]), None],
        "strategy": [StrOptions({"per_class", "all_class"})],
        "random_state": ["random_state"],
    }

    def __init__(self, estimator=None, *, strategy="per_class", random_state=None, k=None, din_k=True, coef_k=2, r_size=10):
        """
        Initialize the DissimilarityIHDClassifier.

        Parameters
        ----------
        estimator : estimator object, default=None
            The base classifier to be used after dissimilarity transformation. Defaults to KNeighborsClassifier.

        strategy : {"per_class", "all_class"}, default="per_class"
            Strategy for selecting the reference points based on instance hardness.

        random_state : int, RandomState instance or None, default=None
            Controls randomness in instance selection.

        k : int, default=None
            Number of neighbors for instance hardness calculation.

        din_k : bool, default=True
            Whether to dynamically adjust the value of `k`.

        coef_k : float, default=2
            Coefficient for scaling `k` when `din_k` is True.

        r_size : int, default=10
            The size of the reference subset (number of hardest instances to select).
        """
        self.estimator = estimator
        self.strategy = strategy
        self.random_state = random_state
        self.r_size = r_size

        self.k = k
        self.din_k = din_k
        self.coef_k = coef_k

        self.classes_ = None
        self.dissim_matrix_ = None
        self.instances_X_r = None

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """
        Fit the classifier, selecting the reference subset (R) using instance hardness and
        computing the dissimilarity matrix for the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : object
            Fitted classifier.
        """
        # Validate input data
        self._validate_data(X, cast_to_ndarray=False)

        # 1. Extract unique classes from the target labels
        self.classes_ = np.unique(y)

        # 2. Select reference subset based on the strategy
        if self.strategy == "per_class":
            X_resampled = self._instance_hardness_per_class(X, y)
        elif self.strategy == "all_class":
            X_resampled = self._instance_hardness_all_class(X, y)

        # Store the reference subset (R)
        self.instances_X_r = X_resampled

        # Compute the dissimilarity matrix for the training data
        self.dissim_matrix_ = self._get_dissim_representation(X)

        # If no estimator is provided, use KNeighborsClassifier by default
        if self.estimator is None:
            self.estimator = KNeighborsClassifier()

        # Fit the estimator using the dissimilarity matrix
        self.estimator.fit(self.dissim_matrix_, y)

        return self

    def _instance_hardness(self, X, y):
        """
        Compute instance hardness scores using the k-disagreeing neighbors (KDN) method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        s : ndarray of shape (n_samples,)
            Instance hardness scores for each sample.

        nx : ndarray of shape (n_samples, k)
            Indices of the k nearest neighbors for each sample.
        """
        n_samples = len(X)

        # Dynamically adjust k based on dataset size if din_k is enabled
        if self.din_k:
            if n_samples > 5000:
                self.coef_k = 3  # Larger datasets
            elif n_samples > 1000:
                self.coef_k = 2  # Medium datasets
            else:
                self.coef_k = 1.5  # Smaller datasets

            self.k = min(50, int(math.log(n_samples) * self.coef_k)) if n_samples > 1000 else int(math.sqrt(n_samples) * self.coef_k)
        else:
            if self.k is None:
                raise ValueError("k must be set when din_k is False.")

        # Calculate hardness scores using k-disagreeing neighbors
        s, nx = kdn_score(X, y, k=self.k)
        return s, nx

    def _instance_hardness_per_class(self, X, y):
        """
        Compute instance hardness scores for each class separately and select the hardest instances.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        ihd_array : ndarray of shape (r_size, n_features)
            Hardest instances from each class.
        """
        ihd_list = []
        s, nx = self._instance_hardness(X, y)

        for class_ in self.classes_:
            aux_s = s[y == class_]
            class_index = [i for i, label in enumerate(y) if label == class_]
            top_indices = heapq.nlargest(self.r_size, range(len(aux_s)), key=lambda i: aux_s[i])
            top_indices = [class_index[i] for i in top_indices]
            ihd_list.append(X[top_indices])

        # Combine instances from all classes
        ihd_array = np.vstack(ihd_list)
        return ihd_array

    def _instance_hardness_all_class(self, X, y):
        """
        Compute instance hardness scores across all classes and select the hardest instances.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        X_top_hard : ndarray of shape (r_size, n_features)
            Hardest instances from the entire dataset.
        """
        s, nx = self._instance_hardness(X, y)
        top_indices = heapq.nlargest(self.r_size, range(len(s)), key=lambda i: s[i])
        return X[top_indices]

