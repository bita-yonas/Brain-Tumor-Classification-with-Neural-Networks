import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing import image
import numpy as np
import plotly.graph_objects as go
import cv2
import google.generativeai as genai
import gc
from fpdf import FPDF
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure genai with API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set to CPU only for custom CNN model
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Ensure the `saliency_maps` directory exists
output_dir = "saliency_maps"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def sanitize_text(text):
    """Replace non-ASCII characters with ASCII equivalents."""
    replacements = {
        "–": "-",  # en dash
        "“": '"',  # left double quotation mark
        "”": '"',  # right double quotation mark
        "’": "'",  # right single quotation mark
        "•": "-",  # bullet point
        # Add more replacements as needed
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def generate_explanation(img_path, model_prediction, confidence):
    prompt = f"""
    Your detailed prompt for explanation generation here.
    """
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content(prompt)
    return sanitize_text(response.text)

def generate_saliency_map(model, img_array, class_index, img_size):
    with tf.GradientTape() as tape:
        img_tensor = tf.convert_to_tensor(img_array)
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        target_class = predictions[:, class_index]

    gradients = tape.gradient(target_class, img_tensor)
    gradients = tf.math.abs(gradients)
    gradients = tf.reduce_max(gradients, axis=-1).numpy().squeeze()

    if gradients.max() > 0:
        gradients /= gradients.max()

    gradients_resized = cv2.resize(gradients, img_size)
    heatmap = cv2.applyColorMap(np.uint8(255 * gradients_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    original_img = (img_array[0] * 255).astype("uint8")
    superimposed_img = cv2.addWeighted(heatmap, 0.6, original_img, 0.4, 0)

    saliency_map_path = os.path.join(output_dir, "saliency_map.jpg")
    cv2.imwrite(saliency_map_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

    return saliency_map_path

def generate_report(prediction, confidence, explanation, historical_cases, next_steps):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Brain Tumor Classification Report", ln=True, align="C")
    pdf.cell(200, 10, txt=f"Prediction: {prediction}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {confidence:.2%}", ln=True)
    pdf.cell(200, 10, txt="Explanation:", ln=True)
    pdf.multi_cell(0, 10, txt=sanitize_text(explanation))
    pdf.cell(200, 10, txt="Historical Cases:", ln=True)
    pdf.multi_cell(0, 10, txt=sanitize_text(historical_cases))
    pdf.cell(200, 10, txt="Recommended Next Steps:", ln=True)
    pdf.multi_cell(0, 10, txt=sanitize_text(next_steps))

    report_path = os.path.join(output_dir, "brain_tumor_classification_report.pdf")
    pdf.output(report_path)
    return report_path

def load_xception_model(model_path):
    img_shape = (150, 150, 3)
    base_model = Xception(include_top=False, weights=None, input_shape=img_shape, pooling='max')

    model = Sequential([
        base_model,
        Flatten(),
        Dropout(rate=0.3),
        Dense(128, activation='relu', name='dense'),
        Dropout(rate=0.25),
        Dense(4, activation='softmax', name='dense_1')
    ])

    model.build(input_shape=(None,) + img_shape)
    model.load_weights(model_path)
    return model

def load_custom_cnn_model(model_path):
    model = load_model(model_path)
    return model

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
        img_size = (150, 150)
    else:
        model = load_custom_cnn_model("cnn_model.h5")
        img_size = (224, 224)

    labels = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']
    img = image.load_img(uploaded_file, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction[0])
    result = labels[class_index]
    confidence = prediction[0][class_index]

    st.markdown(f"""
        <style>
            .result-card {{
                background: linear-gradient(145deg, #1e1e1e, #333);
                padding: 20px;
                border-radius: 15px;
                color: white;
                box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.3);
                text-align: center;
                transition: transform 0.3s ease;
            }}
            .result-card:hover {{
                transform: scale(1.05);
            }}
            .confidence-bar {{
                background: #333;
                border-radius: 10px;
                overflow: hidden;
                margin-top: 10px;
                height: 15px;
            }}
            .confidence-fill {{
                height: 100%;
                background: linear-gradient(90deg, #4caf50, #8bc34a);
                width: {confidence*100}%;
                transition: width 0.6s ease;
            }}
            .result-text {{
                font-size: 1.5em;
                font-weight: bold;
            }}
            .confidence-text {{
                font-size: 1.2em;
                color: #FFD700;
            }}
        </style>

        <div class="result-card">
            <div class="result-text">Prediction: {result}</div>
            <div class="confidence-text">Confidence: {confidence:.2%}</div>
            <div class="confidence-bar">
                <div class="confidence-fill"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    saliency_map_path = generate_saliency_map(model, img_array, class_index, img_size)
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Uploaded MRI Image", use_container_width=True)
    with col2:
        st.image(saliency_map_path, caption='Model-Generated Saliency Map', use_container_width=True)

    st.write("## Class Probability Breakdown")
    probabilities = prediction[0]
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_probabilities = probabilities[sorted_indices]

    fig = go.Figure(go.Bar(
        x=sorted_probabilities,
        y=sorted_labels,
        orientation='h',
        marker=dict(
            color=sorted_probabilities,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title="Probability",
                tickformat=".0%",
                titlefont=dict(color='white')
            )
        ),
        text=[f"{p:.2%}" for p in sorted_probabilities],
        textposition="inside",
        insidetextanchor="middle"
    ))

    fig.update_layout(
        title='Class Probability Breakdown',
        xaxis=dict(title='Probability', tickformat=".0%", showgrid=False, color='white'),
        yaxis=dict(title='Class', showgrid=False, autorange="reversed", color='white'),
        title_font=dict(size=20, color='white'),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        height=450,
        width=700,
    )

    st.plotly_chart(fig)

    explanation = generate_explanation(saliency_map_path, result, confidence)
    st.write("## Expert Analysis of Saliency Map")
    st.write(explanation)

    historical_cases = "Based on previous cases, this prediction aligns with known patterns in similar patients."
    next_steps = "Schedule a follow-up with a neurologist for monitoring and assessment."
    report_path = generate_report(result, confidence, explanation, historical_cases, next_steps)

    with open(report_path, "rb") as f:
        st.download_button(
            label="Download Report as PDF",
            data=f,
            file_name="brain_tumor_classification_report.pdf",
            mime="application/pdf",
        )

    del model
    gc.collect()
    tf.keras.backend.clear_session()
