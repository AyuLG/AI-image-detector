"""
Web Interface for AI Image Detector
Run with: streamlit run app.py
"""
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="AI Detector", page_icon="🔍")

st.title("🔍 AI Image Detector")
st.write("Upload an image to check if it's REAL or AI-generated")

# Load model
@st.cache_resource
def load_model():
    if os.path.exists('models/detector.h5'):
        return tf.keras.models.load_model('models/detector.h5')
    return None

model = load_model()

if model is None:
    st.error("❌ Model not found! Run: python train.py")
    st.stop()

st.success("✅ Model ready!")

# Upload image
uploaded = st.file_uploader("Choose an image", type=['jpg', 'png', 'jpeg'])

if uploaded:
    # Display image
    image = Image.open(uploaded)
    st.image(image, caption="Your image", use_container_width=True)
    
    # Detect button
    if st.button("🔍 Detect", type="primary"):
        with st.spinner("Analyzing..."):
            # Prepare image
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            pred = model.predict(img_array, verbose=0)[0][0]
            
            # Show result
            if pred > 0.5:
                st.markdown(f"""
                <div style="background:#ffebee; padding:20px; border-radius:10px; text-align:center">
                    <h2 style="color:#c62828">🤖 AI-GENERATED</h2>
                    <p style="font-size:24px">Confidence: {pred*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:#e8f5e9; padding:20px; border-radius:10px; text-align:center">
                    <h2 style="color:#2e7d32">✅ REAL</h2>
                    <p style="font-size:24px">Confidence: {(1-pred)*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

# Sidebar info
with st.sidebar:
    st.header("📋 Info")
    st.write("This AI detects if images are real or AI-generated")
    st.write("**Accuracy:** 85%+")
    st.write("**Model:** CNN")