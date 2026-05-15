import logging

import cv2
import mlflow
import mlflow.keras
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ==================================================
# CONFIG
# ==================================================


mlflow.set_tracking_uri("http://127.0.0.1:5000")


st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ⚠️ À adapter avec TON run MLflow
# MODEL_URI = "runs:/YOUR_RUN_ID/cnn_best_model"
# modele intiial MODEL_URI = "runs:/2e200ddc24b042ab96b393dda4e83e6d/cnn_best_model"
# Modele avec des données augmentées via ImageDataGenerator
MODEL_URI = "runs:/d248afd42ed74c9581c0f71d57ce6142/cnn_best_model"


# ==================================================
# LOAD MODEL
# ==================================================


@st.cache_resource
def load_model():
    logger.info("Loading model from MLflow...")
    return mlflow.keras.load_model(MODEL_URI)


model = load_model()


# ==================================================
# PREPROCESS DRAWING
# ==================================================


def preprocess_image_v2(image):
    image = image.convert("L")
    img = np.array(image)

    # Inversion
    img = 255 - img

    # Seuillage
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    # Ouverture morphologique
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # Trouver le bounding box du chiffre
    coords = np.column_stack(np.where(img > 0))
    if coords.size == 0:
        return np.zeros((1, 28, 28, 1))

    y_min, x_min = coords.min(0)
    y_max, x_max = coords.max(0)

    digit = img[y_min:y_max, x_min:x_max]

    # Resize proportionnel (20x20 comme MNIST)
    digit = cv2.resize(digit, (24, 24))

    # Boost du contraste
    digit = cv2.normalize(digit, None, 0, 255, cv2.NORM_MINMAX)

    # Padding pour arriver à 28x28
    # padded = np.pad(digit, ((4, 4), (4, 4)), mode="constant")

    # Padding centré
    padded = np.zeros((28, 28))
    # padded[4:24, 4:24] = digit     # ajustement
    padded[2:26, 2:26] = digit

    # Normalisation
    padded = padded / 255.0

    return padded.reshape(1, 28, 28, 1)


def preprocess_image_v1(image):
    """
    Convertit le dessin utilisateur → format CNN (28x28x1)
    """
    image = image.convert("L")  # grayscale
    image = image.resize((28, 28))

    img = np.array(image)

    # Inversion (fond noir / chiffre blanc)
    img = 255 - img

    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    return img


def preprocess_image_v3(image):
    image = image.convert("L")
    img = np.array(image)

    # Inversion
    img = 255 - img

    # Threshold
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    # ✅ Thinning (IMPORTANT)
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)

    # Bounding box
    coords = np.column_stack(np.where(img > 0))
    if coords.size == 0:
        return np.zeros((1, 28, 28, 1))

    y_min, x_min = coords.min(0)
    y_max, x_max = coords.max(0)
    digit = img[y_min:y_max, x_min:x_max]

    # ✅ Taille MNIST
    digit = cv2.resize(digit, (20, 20))

    # Padding centré
    padded = np.zeros((28, 28))
    padded[4:24, 4:24] = digit

    padded = padded / 255.0

    return padded.reshape(1, 28, 28, 1)


def preprocess_image_v4(image):
    image = image.convert("L")
    img = np.array(image)

    # Inversion
    img = 255 - img

    # Threshold fort
    _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    # ✅ épaissir traits
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)

    # ✅ nettoyage
    img = cv2.medianBlur(img, 3)

    # Bounding box
    coords = np.column_stack(np.where(img > 0))
    if coords.size == 0:
        return np.zeros((1, 28, 28, 1))

    y_min, x_min = coords.min(0)
    y_max, x_max = coords.max(0)

    digit = img[y_min:y_max, x_min:x_max]

    # Resize
    digit = cv2.resize(digit, (24, 24))

    # Center
    padded = np.zeros((28, 28))
    padded[2:26, 2:26] = digit

    padded = padded / 255.0

    return padded.reshape(1, 28, 28, 1)


# ==================================================
# UI
# ==================================================

st.title("✍️ MNIST Digit Recognizer")
st.write("Dessine un chiffre (0–9) dans la zone ci-dessous 👇")

# Canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Boutons
col1, col2 = st.columns(2)

with col1:
    predict_button = st.button("🔍 Prédire")

with col2:
    clear_button = st.button("🧹 Effacer")

# Reset canvas
if clear_button:
    st.session_state["canvas"] = None
    st.rerun()

# ==================================================
# PREDICTION
# ==================================================

if canvas_result.image_data is not None and predict_button:
    image = Image.fromarray(canvas_result.image_data.astype("uint8"))

    processed = preprocess_image_v4(image)

    # ================================
    # PREDICTION
    # ================================
    prediction = model.predict(processed)
    pred_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    # debug 14.05.206 / une classe à 0.99 → bon; 0.1 - modele perdu - mauvais input
    st.write("Raw prediction:", prediction[0])
    st.image(processed.reshape(28, 28), caption="Image envoyée au modèle")

    logger.info(f"Prediction: {pred_class} ({confidence:.2f})")

    # ================================
    # LOGGING MLFLOW
    # ================================
    try:
        with mlflow.start_run(run_name="streamlit_prediction", nested=True):
            mlflow.log_metric("confidence", confidence)
            mlflow.log_param("predicted_class", pred_class)
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")

    # ================================
    # AFFICHAGE RESULTATS
    # ================================
    st.success(f"✅ Chiffre prédit : **{pred_class}**")
    st.write(f"Confiance : **{confidence:.2%}**")

    # ================================
    # TOP 3 PREDICTIONS
    # ================================
    top3 = prediction[0].argsort()[-3:][::-1]
    st.write("🥇 Top 3 prédictions :", top3)

    # ================================
    # BAR CHART
    # ================================
    st.subheader("📊 Distribution des probabilités")
    st.bar_chart(prediction[0])

    # ================================
    # IMAGE TRANSFORMEE
    # ================================
    st.subheader("🖼️ Image transformée (28×28)")
    st.image(
        processed.reshape(28, 28),
        caption="Image normalisée injectée dans le CNN",
        clamp=True,
    )
