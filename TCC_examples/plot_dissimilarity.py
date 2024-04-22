import numpy as np
from sklearn.dissimilarity import DissimilarityRNGClassifier
from sklearn.datasets import load_iris

# df = load_iris()
X = np.array([-1, 1, 1, 1])
y = np.array([0, 1, 1, 1])
dummy = DissimilarityRNGClassifier().fit(X, y)
print("Rodou!")
dummy.predict(X)
print(dummy.score(X, y))