# ARCHITECTURE DE FICHIERS
project_root/
│
├── main.py
├── modules/
│   └── preprocess.py
└── sources/
    └── mnist.pkl.gz


# DATA (/sources/mnist.pkl.gz)
Pas de DB, analyse ORM, RGPD ici, les données sont au format binaires Pickle mnist.pkl, téléchargé dans le repertoire '/sources' 

# PRE TRAITEMENT (/modules/preprocess.py)
## Vectorisation
Chaque image 28×28 → vecteur de dimension 784
## Réduction de dimension (recommandée)
Clustering en 784D est très difficile (curse of dimensionality).

# OBJECTIF 1: CLUSTERING
K-Means
Regroupement hiérarchique
DBSCAN
BIRCH
Methode avancée : auto encoder+clustering

## Affichage des clusters avant/après apprentissage
Projection 2D avec UMAP

# Objectif 2 :
## 2.1 regression logistique (scikit-learn)
Dataset 8×8 pixels
Normalisation
Vectorisation
Classification supervisée
Logging MLflow

## 2.2 CNN 
2 version sde CNN


# Lancements
## MLFLOW 
uvx mlflow server --host 127.0.0.1 --port 5000

# divers modeles CNN sauvés dans MLFLOW
Ces modèles ont ensuite étaient testés avec une appllicaiton streamlit out l'utilisateur dessine un chiffre

=> jamais réussi à faire fonctionner correctement malgré de nombreux  essais:
    - modèles CNN (simple, avec rotation/etirement des images sources, avec d'autres caractéristiques)
    - gestion de l'image produite par le dessin de l'utilsateur dans l'applicaiton web 

Je n'ai pas trouvé la solution bien que les statisqitques,et métriques des modèles finaux sont bonnes ??!!


