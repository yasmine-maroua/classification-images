import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(0)

#  ----------- 3.2) Première modélisation ---------------
#  ----- 3.2.1) ----------------------------------------------

def neighbors(X_train, Y_train, x_test, k):
    dist = np.sum((X_train - x_test.reshape(1,-1))**2 ,1)
    idx = np.argsort(dist)
    return Y_train[idx[:k]]

def prediction(neighbors):
    counts = np.bincount(neighbors)
    return np.argmax(counts)

def evaluation(X_train, Y_train, X_valid, Y_valid, k):
    correct_predictions = 0
    for i in range(X_valid.shape[0]):
        nearest_neighbors = neighbors(X_train, Y_train, X_valid[i], k)
        if (prediction(nearest_neighbors) == Y_valid[i]):
            correct_predictions +=1

    accuracy = correct_predictions / X_valid.shape[0]
    return accuracy

def tracer_courbe(X_train,Y_train,X_valid,Y_valid):
    list_accuracy = []
    #for k in tqdm(range(1,50,4)):
    for k in tqdm(range(1, 31, 2)):
        list_accuracy.append(evaluation(X_train, Y_train, X_valid, Y_valid, k))

    x=range(1,31,2)#range(1,50,4)
    y=list_accuracy
    plt.plot(x,y)
    plt.xlabel("k")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

    meilleur_k = x[np.argmax(list_accuracy)]
    return meilleur_k


#  ----- 3.2.2) 3.2.3) ----------------------------------------------
import pickle
with open("dataset_images_train", 'rb') as fo:
    dict_train = pickle.load(fo, encoding='bytes')
data_X = dict_train["data"]
data_Y = dict_train["target"]

data_X = data_X[:4000,:]

n = data_X.shape[0]
indices = np.random.permutation(n)
train_size = int(0.8 * n)
train_idx = indices[:train_size]
valid_idx = indices[train_size:]

X_train = data_X[train_idx]
Y_train = data_Y[train_idx]

X_valid = data_X[valid_idx]
Y_valid = data_Y[valid_idx]

mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8

X_train = (X_train - mean) / std
X_valid = (X_valid - mean) / std

meilleur_k = tracer_courbe(X_train,Y_train,X_valid,Y_valid)

print(meilleur_k)

#  ----- 3.2.4) ----------------------------------------------
with open("data_images_test", "rb") as fo:
    dict_test = pickle.load(fo)

predictions = []
X_test = dict_test["data"]
X_test = (X_test - mean) / std
for i in tqdm(range(X_test.shape[0])):
    nearest_neighbors = neighbors(X_train, Y_train, X_test[i],meilleur_k)
    pred = prediction(nearest_neighbors)
    predictions.append(pred)

predictions = np.array(predictions)

np.savetxt(
    "images_test_predictions.csv",
    predictions,
    fmt="%.18e"
)