import gzip
import logging
import pickle

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

logger = logging.getLogger(__name__)


def load_mnist(path):
    """
    Charge mnist.pkl.gz
    Gère automatiquement :
    - ((X_train, y_train), (X_test, y_test))
    - ((X_train, y_train), (X_valid, y_valid), (X_test, y_test))
    """
    logger.info(f"Loading MNIST from {path}")

    with gzip.open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    if len(data) == 2:
        logger.info("Detected format (train, test)")
        (X_train, y_train), (X_test, y_test) = data
        return X_train, y_train, X_test, y_test

    elif len(data) == 3:
        logger.info("Detected format (train, valid, test)")
        (X_train, y_train), (_, _), (X_test, y_test) = data
        return X_train, y_train, X_test, y_test

    else:
        raise ValueError("Unknown MNIST format")


def preprocess_pipeline(
    mnist_path,
    n_components=None,
    test_size=0.2,
    random_state=42,
    scale=True,
):
    # ==================================================
    # LOAD DATA
    # ==================================================
    X_train, y_train, X_test, y_test = load_mnist(mnist_path)

    X = np.vstack([X_train, X_test])
    y = np.hstack([y_train, y_test])

    # ==================================================
    # FLATTEN SI IMAGES 2D (ex: 28x28)
    # ==================================================
    if X.ndim == 3:
        # (n_samples, H, W) → (n_samples, H*W)
        X = X.reshape(X.shape[0], -1)

    # ==================================================
    # NORMALISATION
    # ==================================================
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # ==================================================
    # PCA
    # ==================================================
    if n_components is not None:
        pca = PCA(n_components=n_components, random_state=random_state)
        X = pca.fit_transform(X)

    # ==================================================
    # SPLIT FINAL
    # ==================================================
    X, X_test, y, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    return X, y, X_test, y_test


def prepare_cnn_data(mnist_path):
    """
    Prépare MNIST pour CNN (28x28x1 + one-hot labels)
    """

    from modules.preprocess import load_mnist

    X_train, y_train, X_test, y_test = load_mnist(mnist_path)

    # ==================================================
    # RESHAPE → (n_samples, 28, 28, 1)
    # ==================================================
    if X_train.ndim == 3:
        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

    # ==================================================
    # NORMALISATION [0,1]
    # ==================================================
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # ==================================================
    # ONE-HOT ENCODING
    # ==================================================
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    return X_train, X_test, y_train, y_test, y_train_cat, y_test_cat
