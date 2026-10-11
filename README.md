# Classification d'images 

Classification multi-classes d'images couleur 32×32 en 10 classes (avion, voiture, oiseau, chat, daim, chien, grenouille, cheval, bateau, camion).

Six approches sont comparées, du plus simple au plus avancé : **KNN**, **régression logistique**, **réseau de neurones à couches linéaires**, **CNN** et **ResNet-18** avec augmentation de données.

> Réalisé par **Maroua BEHIH**. Le rapport complet est disponible dans [`Rapport_BEHIH_Maroua.pdf`](Rapport_BEHIH_Maroua.pdf).

## Résultats

| Méthode | Score |
|---|---|
| Prédictions aléatoires (référence) | 10,4 % |
| KNN (4000 images) | ≈ 22 % (validation) |
| Régression logistique | 39,5 % |
| Réseau de neurones (couches linéaires, ReLU) | 49,5 % |
| CNN (3 convolutions + BatchNorm + Dropout) | 73,5 % |
| ResNet-18 + data augmentation + TTA | **94,1 %** |

Les scores sont ceux obtenus sur le jeu de test (10 000 images), sauf indication contraire. Les paramètres testés et l'analyse de chaque méthode sont détaillés dans le rapport.

## Données

Les deux fichiers de données se trouvent **à la racine du projet**, à côté des scripts, et sont chargés automatiquement :

- `dataset_images_train` : 20 000 images d'entraînement étiquetées ;
- `data_images_test` : 10 000 images de test non étiquetées.

Les scripts affichent la **précision sur un jeu de validation** (extrait des données d'entraînement).

Chaque image est un vecteur de 3072 valeurs (1024 rouge, 1024 vert, 1024 bleu), qui se redimensionne en tenseur `(3, 32, 32)`. Le jeu d'entraînement contient 20 000 images étiquetées, le jeu de test 10 000 images.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

Chaque script se lance indépendamment :

```bash
python main.py                          # visualisations (image, classes, t-SNE)
python KNN.py                           # k plus proches voisins + courbe précision / k
python logistic_regression.py           # régression logistique multivariée
python neural_network.py                # réseau de neurones à couches linéaires
python CNN.py                           # réseau de convolution
python ResNet18_DataAugmentation.py     # ResNet-18 (long : un GPU est recommandé)
```

Les scripts de classification écrivent leurs prédictions sur le jeu de test dans `images_test_predictions.csv` (le fichier est écrasé à chaque exécution).

## Organisation du code

| Fichier | Rôle |
|---|---|
| `main.py` | Visualisation des données : première image, 10 images d'une classe, projection t-SNE |
| `KNN.py` | K plus proches voisins, recherche du meilleur k |
| `logistic_regression.py` | Régression logistique multivariée (PyTorch) |
| `neural_network.py` | Perceptron multicouche (couches linéaires + ReLU), entraînement par mini-batchs |
| `CNN.py` | CNN à 3 couches de convolution, BatchNorm, Dropout, scheduler StepLR |
| `ResNet18_DataAugmentation.py` | ResNet-18, SGD + CosineAnnealing, AutoAugment, Test Time Augmentation |
