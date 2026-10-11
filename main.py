import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

np.random.seed(42)

import pickle
with open("dataset_images_train", 'rb') as fo:
    dict_train = pickle.load(fo, encoding='bytes')

#  ------------------- 3)Phase d’entraînement et de validation
#  ----------- 3.1) Visualisations des données ---------------
data_X = dict_train["data"]
data_Y = dict_train["target"]

#  ----- 3.1.1) ----------------------------------------------
image_1 = data_X[0].reshape(3,32,32).transpose(1,2,0)

plt.imshow(image_1)
plt.title("Première image de la base d'entraînement")
plt.axis("off")
plt.show()

#  ----- 3.1.2) ----------------------------------------------
def images_classe(classe,X,y):
    X = X.reshape(-1,3,32,32)
    indices = np.where(y==classe)[0][:10]
    plt.figure(figsize=(8, 4))
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i+1)
        img = X[idx].transpose(1, 2, 0)
        plt.imshow(img)
        plt.axis('off')
    plt.suptitle(f"10 exemples de la classe {classe}")
    plt.show()

images_classe(8,data_X,data_Y)

#  ----- 3.1.3) ----------------------------------------------
X_reduit = data_X[:3000]
y_reduit = data_Y[:3000]

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_reduit.reshape(X_reduit.shape[0],-1))
classe = ["avion","voiture","oiseau","chat","daim","chiens","grenouille","cheval","bateau","camion"]

plt.figure(figsize=(10,6))

for i in range(10):
    plt.scatter(
        X_tsne[y_reduit == i,0],
        X_tsne[y_reduit == i,1],
        label= classe[i],
        s=10
    )

plt.legend()
plt.title("Projection TSNE des images")
plt.show()
