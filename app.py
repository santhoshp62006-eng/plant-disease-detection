import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "AI-powered plant disease detection system"}
)

# Custom CSS for colorful interface
st.markdown("""
<style>
    :root {
        --primary-green: #2ecc71;
        --dark-green: #27ae60;
        --light-bg: #f8f9fa;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .header-title {
        background: linear-gradient(90deg, #2ecc71, #27ae60);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .confidence-high {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .confidence-medium {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .confidence-low {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .stTabs [data-baseweb="tab-list"] button {
        background-color: #ecf0f1;
        color: #2c3e50;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #2ecc71, #27ae60) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_streamlit.h5")

model = load_model()

# Load class labels
with open("class_labels.json", "r") as f:
    class_labels = json.load(f)

# Convert keys to int
class_labels = {int(k): v for k, v in class_labels.items()}

# Header
st.markdown("""
<div class="header-title">
    <h1>🌿 Plant Disease Detection AI 🌿</h1>
    <p style="font-size: 18px; margin-top: 10px;">Detect plant diseases instantly with advanced machine learning</p>
</div>
""", unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Your Leaf Image")
    st.markdown("Supported formats: JPG, PNG, JPEG")
    uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "png", "jpeg"])

with col2:
    st.markdown("### ℹ️ How to Use")
    st.info("""
    1️⃣ Click 'Browse files' to upload a leaf image  
    2️⃣ Wait for AI analysis  
    3️⃣ View detailed results and recommendations
    """)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Display image with analysis
    st.markdown("---")
    st.markdown("### 🔍 Image Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Uploaded Image:**")
        st.image(image, caption="Your leaf image", use_column_width=True, 
                output_format="PNG")

    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with col2:
        st.markdown("**Processing...**")
        progress_bar = st.progress(0)
        
        # Prediction
        predictions = model.predict(img_array, verbose=0)
        
        for i in range(101):
            progress_bar.progress(i)
        
        predicted_class = class_labels[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        st.success("✅ Analysis Complete!")
    
    # Results section with tabs
    st.markdown("---")
    st.markdown("### 📊 Results & Analysis")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📈 Confidence Analysis", "🔬 All Predictions"])
    
    with tab1:
        if confidence >= 80:
            st.markdown(f"""
            <div class="confidence-high">
                <h2>✅ Predicted Disease</h2>
                <h1 style="font-size: 36px; margin: 10px 0;">{predicted_class}</h1>
                <p>🎯 Confidence: <strong>{confidence:.2f}%</strong> - High reliability</p>
            </div>
            """, unsafe_allow_html=True)
        elif confidence >= 60:
            st.markdown(f"""
            <div class="confidence-medium">
                <h2>⚠️ Predicted Disease</h2>
                <h1 style="font-size: 36px; margin: 10px 0;">{predicted_class}</h1>
                <p>📊 Confidence: <strong>{confidence:.2f}%</strong> - Moderate reliability</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="confidence-low">
                <h2>❓ Predicted Disease (Low Confidence)</h2>
                <h1 style="font-size: 36px; margin: 10px 0;">{predicted_class}</h1>
                <p>⚠️ Confidence: <strong>{confidence:.2f}%</strong> - Please verify</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Confidence gauge using Streamlit metric
            gauge_value = confidence / 100.0
            if gauge_value >= 0.8:
                status = "Highly Confident"
                color = "green"
            elif gauge_value >= 0.6:
                status = "Moderately Confident"
                color = "orange"
            else:
                status = "Low Confidence"
                color = "red"
            
            st.metric(label="Confidence Level", value=f"{confidence:.2f}%", delta=None)
            st.markdown(f"**Status:** {status}")
        
        with col2:
            st.markdown("**Interpretation:**")
            if confidence >= 80:
                st.success("🟢 Very reliable prediction - High confidence")
            elif confidence >= 60:
                st.warning("🟡 Reliable prediction with normal confidence")
            else:
                st.error("🔴 Low confidence - Consider consulting an expert")
    
    with tab3:
        st.markdown("**All Disease Predictions:**")
        
        # Sort predictions by confidence
        sorted_indices = np.argsort(predictions[0])[::-1]
        
        # Create data for chart
        labels = [class_labels[idx] for idx in sorted_indices[:5]]
        scores = [predictions[0][idx] * 100 for idx in sorted_indices[:5]]
        
        chart_data = {
            "Disease": labels,
            "Confidence (%)": scores
        }
        
        # Streamlit bar chart
        st.bar_chart(data={
            "Disease": labels,
            "Confidence (%)": scores
        }, x="Disease", y="Confidence (%)", use_container_width=True)
        
        # Detailed table
        st.markdown("**Detailed Scores:**")
        prediction_data = {
            "Disease": [class_labels[idx] for idx in sorted_indices],
            "Confidence (%)": [f"{predictions[0][idx]*100:.2f}" for idx in sorted_indices],
            "Rank": list(range(1, len(sorted_indices) + 1))
        }
        st.dataframe(prediction_data, use_container_width=True)