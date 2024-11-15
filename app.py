import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
import numpy as np
import plotly.graph_objects as go
import cv2
from dotenv import load_dotenv
import google.generativeai as genai
import PIL.Image
import gc

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Directory for saving saliency maps
output_dir = "saliency_maps"
os.makedirs(output_dir, exist_ok=True)

# Functions
def generate_explanation(img_path, model_prediction, confidence):
    """Generate AI-powered explanation for the saliency map."""
    prompt = f"""You are an expert neurologist. Explain the saliency map for an MRI scan classified as '{model_prediction}' with {confidence*100:.2f}% confidence."""
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

def generate_saliency_map(model, img_array, class_index, img_size):
    """Generate a saliency map to highlight areas of importance for the model's prediction."""
    with tf.GradientTape() as tape:
        img_tensor = tf.convert_to_tensor(img_array)
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        target_class = predictions[:, class_index]

    gradients = tape.gradient(target_class, img_tensor)
    gradients = tf.math.abs(gradients)
    gradients = tf.reduce_max(gradients, axis=-1)
    gradients = gradients.numpy().squeeze()
    gradients = cv2.resize(gradients, img_size)

    # Normalize and apply threshold
    center = (gradients.shape[0] // 2, gradients.shape[1] // 2)
    radius = min(center[0], center[1]) - 10
    y, x = np.ogrid[:gradients.shape[0], :gradients.shape[1]]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    gradients = gradients * mask

    # Heatmap overlay
    gradients = cv2.GaussianBlur(gradients, (11, 11), 0)
    heatmap = cv2.applyColorMap(np.uint8(255 * gradients), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Superimpose
    original_img = img_array[0] * 255
    superimposed_img = heatmap * 0.7 + original_img * 0.3
    return superimposed_img.astype(np.uint8)

def load_custom_cnn_model(model_path):
    """Load a custom CNN model."""
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
    model.compile(optimizer=Adamax(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', Precision(), Recall()])
    model.load_weights(model_path)
    return model

def load_xception_model(model_path):
    """Load the Xception transfer learning model."""
    base_model = tf.keras.applications.Xception(include_top=False, weights="imagenet", input_shape=(150, 150, 3), pooling='max')
    model = Sequential([
        base_model,
        Flatten(),
        Dropout(rate=0.3),
        Dense(128, activation='relu'),
        Dropout(rate=0.25),
        Dense(4, activation='softmax')
    ])
    model.compile(optimizer=Adamax(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', Precision(), Recall()])
    model.load_weights(model_path)
    return model

# UI with Streamlit
st.title("🧠 Brain Tumor Classification")
st.write("Upload a brain MRI scan for AI-driven classification and saliency map analysis.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    selected_model = st.radio("Select Model", ("Transfer Learning - Xception", "Custom CNN"))
    img_size = (150, 150)

    if selected_model == "Transfer Learning - Xception":
        model = load_xception_model("xception_model.weights.h5")
    else:
        model = load_custom_cnn_model("cnn_model.h5")

    labels = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']
    img = image.load_img(uploaded_file, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction[0])
    result = labels[class_index]
    confidence = prediction[0][class_index]

    st.markdown(f"### Prediction: {result} (Confidence: {confidence:.2%})")

    # Saliency Map
    saliency_map = generate_saliency_map(model, img_array, class_index, img_size)
    explanation = generate_explanation(uploaded_file.name, result, confidence)

    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Uploaded MRI Image", use_column_width=True)
    with col2:
        st.image(saliency_map, caption="Saliency Map", use_column_width=True)

    st.markdown("### AI-Generated Explanation")
    st.write(explanation)

    # Clear memory
    del model
    gc.collect()
    tf.keras.backend.clear_session()
