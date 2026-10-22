
import os
import json
import streamlit as st
import numpy as np
import tensorflow as tf

from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas


# -------------------------------
# CONFIGURATION
# -------------------------------

MODEL_PATH = "models/digit_model.keras"
METRICS_PATH = "results/metrics.json"

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="wide"
)


# -------------------------------
# LOAD TRAINED CNN MODEL
# -------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


st.title("🔢 Handwritten Digit Recognition")
st.subheader("Deep Learning Mini-Project Using CNN")

st.write(
    "Draw a handwritten digit or upload an image. "
    "The CNN model predicts a digit between 0 and 9."
)

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Run train_model.py first.")
    st.stop()

model = load_model()


# -------------------------------
# IMAGE PREPROCESSING
# -------------------------------

def preprocess_image(image):
    # Convert image to grayscale
    image = image.convert("L")

    pixels = np.array(image)

    # Convert black ink on a white background
    # to a white digit on a black background.
    if pixels.mean() > 127:
        image = ImageOps.invert(image)

    pixels = np.array(image)

    # Find the digit's boundaries
    ys, xs = np.where(pixels > 30)

    if len(xs) > 0:
        top = max(0, int(ys.min()) - 2)
        bottom = min(pixels.shape[0], int(ys.max()) + 3)
        left = max(0, int(xs.min()) - 2)
        right = min(pixels.shape[1], int(xs.max()) + 3)

        pixels = pixels[top:bottom, left:right]
        image = Image.fromarray(pixels)

    # Resize while preserving the aspect ratio
    image.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Place the digit in the center of a 28x28 image
    canvas = Image.new("L", (28, 28), color=0)

    x_offset = (28 - image.width) // 2
    y_offset = (28 - image.height) // 2

    canvas.paste(image, (x_offset, y_offset))

    # Normalize pixels to 0-1
    array = np.array(canvas, dtype=np.float32) / 255.0

    # CNN input shape: batch, height, width, channel
    array = array.reshape(1, 28, 28, 1)

    return array, canvas


# -------------------------------
# PREDICTION FUNCTION
# -------------------------------

def predict_digit(image):
    processed_image, preview_image = preprocess_image(image)

    probabilities = model.predict(
        processed_image,
        verbose=0
    )[0]

    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit]) * 100

    st.image(
        preview_image,
        caption="Preprocessed 28 x 28 image",
        width=180
    )

    st.success(f"Predicted Digit: {predicted_digit}")

    st.metric(
        "Prediction Confidence",
        f"{confidence:.2f}%"
    )

    st.write("Probability for each digit (0-9)")

    st.bar_chart({
        str(digit): float(probabilities[digit])
        for digit in range(10)
    })

    if confidence < 60:
        st.warning(
            "The model is uncertain. Try a clearer handwritten digit."
        )


# -------------------------------
# APPLICATION TABS
# -------------------------------

draw_tab, upload_tab, results_tab = st.tabs([
    "✍️ Draw Digit",
    "📤 Upload Image",
    "📊 Model Results"
])


# -------------------------------
# TAB 1: DRAW A DIGIT
# -------------------------------

with draw_tab:
    st.header("Draw a Digit")

    st.write(
        "Use your mouse to draw one large digit on the white canvas."
    )

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=18,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=280,
        width=280,
        drawing_mode="freedraw",
        update_streamlit=True,
        return_image_data=True,
        key="digit_canvas"
    )

    if st.button("Predict Drawn Digit", key="predict_drawn"):
        if canvas_result.image_data is None:
            st.warning("Please draw a digit first.")

        elif not np.any(canvas_result.image_data[:, :, :3] < 200):
            st.warning("The canvas is empty. Please draw a digit.")

        else:
            drawn_image = Image.fromarray(
                canvas_result.image_data.astype(np.uint8)
            ).convert("RGB")

            predict_digit(drawn_image)


# -------------------------------
# TAB 2: UPLOAD AN IMAGE
# -------------------------------

with upload_tab:
    st.header("Upload a Handwritten Digit")

    st.write(
        "Upload a clear image containing one digit."
    )

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["png", "jpg", "jpeg", "bmp"],
        key="digit_upload"
    )

    if uploaded_file is not None:
        try:
            uploaded_image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.image(
                uploaded_image,
                caption="Uploaded image",
                width=250
            )

            if st.button(
                "Predict Uploaded Digit",
                key="predict_uploaded"
            ):
                predict_digit(uploaded_image)

        except Exception as error:
            st.error(f"Could not process the image: {error}")


# -------------------------------
# TAB 3: MODEL RESULTS
# -------------------------------

with results_tab:
    st.header("CNN Model Evaluation")

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as file:
            metrics = json.load(file)

        col1, col2 = st.columns(2)

        col1.metric(
            "MNIST Test Accuracy",
            f"{metrics['test_accuracy'] * 100:.2f}%"
        )

        col2.metric(
            "MNIST Test Loss",
            f"{metrics['test_loss']:.4f}"
        )

        st.caption(
            "Results calculated using the held-out MNIST test dataset."
        )

    else:
        st.info(
            "Run train_model.py to generate evaluation metrics."
        )

    accuracy_path = "results/accuracy.png"
    loss_path = "results/loss.png"

    if os.path.exists(accuracy_path):
        st.subheader("Training and Validation Accuracy")
        st.image(accuracy_path, width=700)

    if os.path.exists(loss_path):
        st.subheader("Training and Validation Loss")
        st.image(loss_path, width=700)


# -------------------------------
# FOOTER
# -------------------------------

st.divider()

st.caption(
    "Handwritten Digit Recognition Using CNN | "
    "Deep Learning Mini-Project"
)
