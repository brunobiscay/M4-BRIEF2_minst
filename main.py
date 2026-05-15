"""
MNIST clustering with MLflow
- Metrics: accuracy, precision, recall, F1, ROC AUC
- Visualization: before/after clustering (2D projection)
- Artifacts logged to MLflow
"""

import os
import tempfile
from collections import Counter

import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import numpy as np
import seaborn as sns
from skimage.transform import resize
from sklearn.cluster import (
    DBSCAN,
    AgglomerativeClustering,
    Birch,
    KMeans,
)
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from modules.preprocess import prepare_cnn_data
from modules.preprocess_20260513_11h30 import load_mnist, preprocess_pipeline

# ======================================================
# CONFIG
# ======================================================

MNIST_PATH = "sources/mnist.pkl.gz"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT = "mnist-unsupervised-clustering"

N_CLASSES = 10
RANDOM_STATE = 42

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

# ======================================================
# UTILS – CLUSTER → CLASS
# ======================================================


def cluster_to_label_mapping(y_true, y_cluster):
    mapping = {}
    for cluster in np.unique(y_cluster):
        labels = y_true[y_cluster == cluster]
        mapping[cluster] = Counter(labels).most_common(1)[0][0]
    return mapping


def clusters_to_predictions(y_cluster, mapping):
    return np.array([mapping[c] for c in y_cluster])


# ======================================================
# UTILS – METRICS
# ======================================================


def log_classification_metrics(y_true, y_pred):
    mlflow.log_metric("accuracy", accuracy_score(y_true, y_pred))
    mlflow.log_metric(
        "precision_macro",
        precision_score(y_true, y_pred, average="macro", zero_division=0),
    )
    mlflow.log_metric(
        "recall_macro",
        recall_score(y_true, y_pred, average="macro", zero_division=0),
    )
    mlflow.log_metric(
        "f1_macro",
        f1_score(y_true, y_pred, average="macro", zero_division=0),
    )


def log_roc_curve(y_true, y_pred, run_name):
    y_true_bin = label_binarize(y_true, classes=range(N_CLASSES))
    y_pred_bin = label_binarize(y_pred, classes=range(N_CLASSES))

    auc = roc_auc_score(y_true_bin, y_pred_bin, multi_class="ovr")
    mlflow.log_metric("roc_auc_ovr", auc)

    plt.figure(figsize=(7, 6))
    for i in range(N_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        plt.plot(fpr, tpr, alpha=0.6, label=f"Class {i}")

    plt.plot([0, 1], [0, 1], "k--")
    plt.title(f"ROC Curve – {run_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "roc_curve.png")
        plt.savefig(path)
        plt.close()
        mlflow.log_artifact(path)


def log_confusion_matrix(y_true, y_pred, run_name):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, cmap="Blues", fmt="d", annot=False)
    plt.title(f"Confusion Matrix – {run_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "confusion_matrix.png")
        plt.savefig(path)
        plt.close()
        mlflow.log_artifact(path)


# ======================================================
# VISUALIZATION – BEFORE / AFTER
# ======================================================


def project_2d(X):
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        random_state=RANDOM_STATE,
    )
    return tsne.fit_transform(X)


def log_cluster_visualization(X_2d, labels, title, artifact_name):
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=labels,
        cmap="tab10",
        s=4,
        alpha=0.6,
    )
    plt.colorbar(sc)
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, artifact_name)
        plt.savefig(path)
        plt.close()
        mlflow.log_artifact(path)


# ======================================================
# AUTOENCODER
# ======================================================


def autoencoder_embedding(X, latent_dim=10, epochs=20):
    input_dim = X.shape[1]

    inputs = Input(shape=(input_dim,))
    x = Dense(128, activation="relu")(inputs)
    x = Dense(64, activation="relu")(x)
    latent = Dense(latent_dim, activation="relu")(x)

    x = Dense(64, activation="relu")(latent)
    x = Dense(128, activation="relu")(x)
    outputs = Dense(input_dim, activation="sigmoid")(x)

    autoencoder = Model(inputs, outputs)
    encoder = Model(inputs, latent)

    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(X, X, epochs=epochs, batch_size=256, verbose=0)

    return encoder.predict(X)


# ======================================================
# Pour regression logistique, MNIST → 8×8
# ❝On ne peut redimensionner en 8×8 que des images, pas des vecteurs PCA ou normalisés❞
# ======================================================


def reshape_to_8x8(X_flat, original_dim=28):
    """
    Redimensionne MNIST 28x28 → 8x8 via resize (interpolation moyenne)

    Parameters
    ----------
    X_flat : np.ndarray
        Shape (n_samples, 784)
    original_dim : int
        Taille originale (28 pour MNIST)

    Returns
    -------
    X_8x8 : np.ndarray
        Shape (n_samples, 64)
    """
    if X_flat.ndim != 2 or X_flat.shape[1] != original_dim * original_dim:
        raise ValueError(
            f"reshape_to_8x8 attend des images aplaties {original_dim}x{original_dim}, "
            f"reçu shape={X_flat.shape}"
        )

    n_samples = X_flat.shape[0]
    X_images = X_flat.reshape(n_samples, original_dim, original_dim)

    X_8x8 = np.zeros((n_samples, 8, 8))
    for i in range(n_samples):
        X_8x8[i] = resize(
            X_images[i],
            (8, 8),
            anti_aliasing=True,
            preserve_range=True,
        )

    return X_8x8.reshape(n_samples, 64)


# ======================================================
# MAIN EXPERIMENT RUNNER
# ======================================================


def run_experiment(name, model, X, y, X_2d):
    with mlflow.start_run(run_name=name):
        y_cluster = model.fit_predict(X)

        mapping = cluster_to_label_mapping(y, y_cluster)
        y_pred = clusters_to_predictions(y_cluster, mapping)

        log_classification_metrics(y, y_pred)
        log_roc_curve(y, y_pred, name)
        log_confusion_matrix(y, y_pred, name)

        log_cluster_visualization(
            X_2d,
            y,
            "Projection 2D – Labels réels",
            "projection_labels.png",
        )

        log_cluster_visualization(
            X_2d,
            y_cluster,
            f"Projection 2D – Clusters ({name})",
            "projection_clusters.png",
        )


# ======================================================
# MAIN
# ======================================================


def main():
    # X, y, _, _ = preprocess_pipeline(MNIST_PATH, n_components=30)
    X, y, X_test, y_test = preprocess_pipeline(
        MNIST_PATH,
        n_components=30,
    )
    X_2d = project_2d(X)

    run_experiment(
        "KMeans",
        KMeans(n_clusters=10, random_state=RANDOM_STATE),
        X,
        y,
        X_2d,
    )

    run_experiment(
        "Hierarchical",
        AgglomerativeClustering(n_clusters=10),
        X,
        y,
        X_2d,
    )

    run_experiment(
        "DBSCAN",
        DBSCAN(eps=3.0, min_samples=5),
        X,
        y,
        X_2d,
    )

    run_experiment(
        "BIRCH",
        Birch(n_clusters=10),
        X,
        y,
        X_2d,
    )

    # Autoencoder + KMeans
    with mlflow.start_run(run_name="Autoencoder + KMeans"):
        X_latent = autoencoder_embedding(X)
        kmeans = KMeans(n_clusters=10, random_state=RANDOM_STATE)

        y_cluster = kmeans.fit_predict(X_latent)
        mapping = cluster_to_label_mapping(y, y_cluster)
        y_pred = clusters_to_predictions(y_cluster, mapping)

        log_classification_metrics(y, y_pred)
        log_roc_curve(y, y_pred, "Autoencoder + KMeans")
        log_confusion_matrix(y, y_pred, "Autoencoder + KMeans")

        X_2d_latent = project_2d(X_latent)

        log_cluster_visualization(
            X_2d_latent,
            y,
            "Latent space – Labels réels",
            "latent_labels.png",
        )

        log_cluster_visualization(
            X_2d_latent,
            y_cluster,
            "Latent space – Clusters",
            "latent_clusters.png",
        )

    # ==================================================
    # LOGISTIC REGRESSION – SUPERVISED (8x8)
    # ==================================================

    with mlflow.start_run(run_name="Logistic Regression 8x8"):
        # 🔁 Recharge MNIST BRUT (pas PCA, pas scaler)
        X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_mnist(MNIST_PATH)

        X_raw = np.vstack([X_train_raw, X_test_raw])
        y_raw = np.hstack([y_train_raw, y_test_raw])

        # Flatten si besoin (28x28 → 784)
        if X_raw.ndim == 3:
            X_raw = X_raw.reshape(X_raw.shape[0], -1)

        # ✅ Resize images en 8x8
        X_8x8 = reshape_to_8x8(X_raw)

        # Split correct
        X_train, X_test, y_train, y_test = train_test_split(
            X_8x8,
            y_raw,
            test_size=0.2,
            stratify=y_raw,
            random_state=RANDOM_STATE,
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = LogisticRegression(
            multi_class="ovr",
            max_iter=2000,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        log_classification_metrics(y_test, y_pred)
        log_roc_curve(y_test, y_pred, "Logistic Regression 8x8")
        log_confusion_matrix(y_test, y_pred, "Logistic Regression 8x8")

        # PCA UNIQUEMENT POUR VISUALISATION
        from sklearn.decomposition import PCA

        pca_visu = PCA(n_components=2, random_state=RANDOM_STATE)
        X_test_pca = pca_visu.fit_transform(X_test)

        kmeans = KMeans(n_clusters=10, random_state=RANDOM_STATE)
        clusters = kmeans.fit_predict(X_test)

        log_cluster_visualization(
            X_test_pca,
            clusters,
            "Clustering Logistic Regression (PCA 2D)",
            "logreg_clustering.png",
        )

    # ==================================================
    # CNN – SUPERVISED (28x28x1)
    # ==================================================

    with mlflow.start_run(run_name="CNN 28x28"):
        # Préparation données
        X_train, X_test, y_train, y_test, y_train_cat, y_test_cat = prepare_cnn_data(
            MNIST_PATH
        )

        # ==================================================
        # MODEL CNN
        # ==================================================

        from tensorflow.keras.layers import BatchNormalization, Dropout

        model = Sequential(
            [
                Conv2D(
                    32,
                    (3, 3),
                    padding="same",
                    activation="relu",
                    input_shape=(28, 28, 1),
                ),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation="relu"),
                MaxPooling2D((2, 2)),
                Dropout(0.25),
                Conv2D(64, (3, 3), padding="same", activation="relu"),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation="relu"),
                MaxPooling2D((2, 2)),
                Dropout(0.25),
                Flatten(),
                Dense(128, activation="relu"),
                BatchNormalization(),
                Dropout(0.5),
                Dense(10, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        # ==================================================
        # TRAIN
        # ==================================================

        # Avec EarlyStopping
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = os.path.join(tmp, "best_model.h5")

            # ==================================================
            # CALLBACKS
            # ==================================================

            early_stop = EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )

            checkpoint = ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            )

            # ==================================================
            # TRAIN
            # ==================================================

            # DATA AUGMENTATION

            datagen = ImageDataGenerator(
                rotation_range=15,
                zoom_range=0.15,
                width_shift_range=0.15,
                height_shift_range=0.15,
                shear_range=0.1,
            )

            history = model.fit(
                datagen.flow(X_train, y_train_cat, batch_size=128, shuffle=True),
                epochs=20,
                validation_data=(X_test, y_test_cat),
                callbacks=[early_stop, checkpoint],
                verbose=1,
            )

            mlflow.log_param("epochs", 20)
            mlflow.log_param("batch_size", 128)
            mlflow.log_param("early_stopping_patience", 5)

            # ==================================================
            # LOAD BEST MODEL
            # ==================================================
            from tensorflow.keras.models import load_model

            best_model = load_model(checkpoint_path)

            # ==================================================
            # (DEBUG CNN) 14.5.2026
            # ==================================================
            import matplotlib.pyplot as plt

            plt.imshow(X_test[0].reshape(28, 28), cmap="gray")
            plt.title(f"Label réel: {y_test[0]}")
            plt.show()

            pred = best_model.predict(X_test[0].reshape(1, 28, 28, 1))
            print("Prediction:", np.argmax(pred))

            # ==================================================
            # LOG MODEL MLFLOW
            # ==================================================
            mlflow.keras.log_model(best_model, "cnn_best_model")

        # ==================================================
        # PREDICTIONS
        # ==================================================

        y_proba = best_model.predict(X_test)
        y_pred = np.argmax(y_proba, axis=1)

        # ==================================================
        # METRICS MLFLOW
        # ==================================================
        log_classification_metrics(y_test, y_pred)
        log_roc_curve(y_test, y_pred, "CNN 28x28")
        log_confusion_matrix(y_test, y_pred, "CNN 28x28")

        # ==================================================
        # CLUSTERING VISUALIZATION (PCA)
        # ==================================================
        # Flatten + projection PCA
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        X_test_pca = pca.fit_transform(X_test_flat)

        kmeans = KMeans(n_clusters=10, random_state=RANDOM_STATE)
        clusters = kmeans.fit_predict(X_test_flat)

        log_cluster_visualization(
            X_test_pca,
            clusters,
            "CNN Clustering (PCA 2D)",
            "cnn_clustering.png",
        )


if __name__ == "__main__":
    main()
