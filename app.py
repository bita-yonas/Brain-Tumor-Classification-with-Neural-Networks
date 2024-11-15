import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing import image
import numpy as np
import plotly.graph_objects as go
import cv2
import google.generativeai as genai
import PIL.Image
import gc
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set to CPU only for custom CNN model
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

output_dir = "saliency_maps"
os.makedirs(output_dir, exist_ok=True)

# Function to generate an explanation using Google Generative AI
def generate_explanation(img_path, model_prediction, confidence):
    prompt = f"""
    Your AI-generated expert analysis for MRI classification.
    The prediction is: '{model_prediction}' with confidence: {confidence * 100:.2f}%.
    Kindly summarize the reasoning for this prediction.
    """
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

# Function to generate saliency maps
def generate_saliency_map(model, img_array, class_index, img_size):
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
    heatmap = cv2.applyColorMap(np.uint8(255 * gradients), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, img_size)
    superimposed_img = heatmap * 0.7 + img_array[0] * 255 * 0.3
    superimposed_img = superimposed_img.astype(np.uint8)
    saliency_map_path = os.path.join(output_dir, "saliency_map.jpg")
    cv2.imwrite(saliency_map_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
    return saliency_map_path

# Load Xception model
def load_xception_model(model_path):
    img_shape = (150, 150, 3)
    base_model = tf.keras.applications.Xception(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')
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

# Load custom CNN model
def load_custom_cnn_model(model_path):
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

# Streamlit app
st.title("🧠 Brain Tumor Classification")
st.write("Upload a brain MRI scan for AI-driven classification and analysis.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    selected_model = st.radio(
        "Select Model",
        ("Transfer Learning - Xception", "Custom CNN")
    )

    if selected_model == "Transfer Learning - Xception":
        model = load_xception_model("xception_model.weights.h5")
    else:
        model = load_custom_cnn_model("cnn_model.h5")

    labels = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']
    img_size = (150, 150)
    img = image.load_img(uploaded_file, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction[0])
    result = labels[class_index]
    confidence = prediction[0][class_index]

    st.markdown(f"### Prediction: **{result}**")
    st.markdown(f"Confidence: **{confidence:.2%}**")

    saliency_map_path = generate_saliency_map(model, img_array, class_index, img_size)
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Uploaded MRI Image", use_column_width=True)
    with col2:
        st.image(saliency_map_path, caption="Saliency Map", use_column_width=True)

    st.write("## Expert Analysis of Saliency Map")
    explanation = generate_explanation(saliency_map_path, result, confidence)
    st.write(explanation)

    # Clear resources
    del model
    gc.collect()
    tf.keras.backend.clear_session()
