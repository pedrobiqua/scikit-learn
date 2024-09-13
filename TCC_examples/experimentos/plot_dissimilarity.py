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
from sklearn.dissimilarity import DissimilarityIHDClassifier
from sklearn.dissimilarity import DissimilarityCentroidClassifier
from dataset_arff.arffconvert import load_SEAGenerator_test_f2_f4 
from dataset_arff.arffconvert import load_AssetNegotiationGenerator_f1_f5 
from dataset_arff.arffconvert import load_AgrawalGenerator_test_mode_f2_f9

# from dataset_arff.arffconvert import load_SEAGenerator
# data = load_iris()
#data = load_digits()
#X, y = data.data, data.target
 
X, y = load_AssetNegotiationGenerator_f1_f5() # Usando o arff | ESSE PROCESSO DEMORA UM POUCO, ELE CONVERTE O ARFF PARA UM CONJUNTO X e y, onde o X é as features e o Y é as classes
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

# TODO: PERGUNTA: Qual a qauntidade ideal de instancias R para montar o espaço de dissmilaridade?
# dissimilarity = DissimilarityRNGClassifier(estimator=KNeighborsClassifier(), random_state=60)
dissimilarity = DissimilarityIHDClassifier(estimator=KNeighborsClassifier(),random_state=42, k=1, din_k=True,coef_k=2)
# dissimilarity = DissimilarityCentroidClassifier(estimator=KNeighborsClassifier(), n_clusters=3, random_state=42)
print(dissimilarity) # Mostra o teste que estamos usando
dissimilarity.fit(X_train, y_train)

score = model_selection.cross_val_score(dissimilarity, X_test, y_test, cv=10)
print(f"Acc mean:{score.mean()}\nAcc std:{score.std()}\nAcc per fold:{score}")

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
print(cm)