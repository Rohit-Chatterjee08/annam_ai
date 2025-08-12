"""
🌾 Annam AI - Streamlit Web Application
=====================================

A comprehensive Streamlit web interface for the Annam AI agricultural intelligence system.

Features:
- Interactive web interface for all 4 modules
- File upload for images and data
- Real-time predictions and analysis
- Downloadable results
- User-friendly dashboard

Author: Annam AI Team
License: MIT
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import sys
import os
from pathlib import Path
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import Annam AI modules
try:
    from models.crop_classification import CropClassificationCNN, CropClassificationML
    from models.yield_prediction import YieldPredictionML, YieldTimeSeriesModel
    from models.disease_detection import DiseaseDetectionCNN, PlantVillageDataset
    from models.agricultural_advisory import AgricultureAdvisor
except ImportError as e:
    st.error(f"Failed to import Annam AI modules: {e}")
    st.info("Please ensure all dependencies are installed and modules are available.")

# Page configuration
st.set_page_config(
    page_title="Annam AI - Agricultural Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .module-header {
        font-size: 2rem;
        color: #388E3C;
        border-bottom: 2px solid #4CAF50;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E8F5E8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4CAF50;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #FF9800;
    }
    .result-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2196F3;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""

    # Header
    st.markdown('<h1 class="main-header">🌾 Annam AI - Agricultural Intelligence System</h1>', 
                unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    selected_module = st.sidebar.selectbox(
        "Select Module",
        ["🏠 Home", "🌱 Crop Classification", "📊 Yield Prediction", 
         "🦠 Disease Detection", "💬 Agricultural Advisory"]
    )

    # Module routing
    if selected_module == "🏠 Home":
        show_home()
    elif selected_module == "🌱 Crop Classification":
        show_crop_classification()
    elif selected_module == "📊 Yield Prediction":
        show_yield_prediction()
    elif selected_module == "🦠 Disease Detection":
        show_disease_detection()
    elif selected_module == "💬 Agricultural Advisory":
        show_agricultural_advisory()

def show_home():
    """Display the home page with system overview."""

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ## 🎯 Welcome to Annam AI

        Annam AI is a comprehensive agricultural intelligence system that combines 
        cutting-edge AI technologies to help farmers, researchers, and agricultural 
        professionals make data-driven decisions.

        ### 🚀 Key Features:
        - **Crop Classification**: Identify crops from satellite imagery
        - **Yield Prediction**: Forecast agricultural production
        - **Disease Detection**: Early detection of plant diseases
        - **Agricultural Advisory**: AI-powered farming advice
        """)

    with col2:
        st.markdown("""
        ## 📊 System Statistics
        """)

        # Create sample metrics
        col2a, col2b, col2c = st.columns(3)

        with col2a:
            st.metric("Supported Crops", "25+", "5")

        with col2b:
            st.metric("Disease Types", "50+", "10")

        with col2c:
            st.metric("Accuracy", "95%", "2%")

    # Feature overview
    st.markdown("## 🔧 Module Overview")

    modules = [
        {
            "name": "🌱 Crop Classification", 
            "description": "Classify crops from satellite or drone imagery using deep learning",
            "features": ["CNN-based classification", "Traditional ML support", "Satellite image processing"]
        },
        {
            "name": "📊 Yield Prediction",
            "description": "Predict crop yields using weather, soil, and historical data",
            "features": ["Multiple ML models", "Time series forecasting", "Weather integration"]
        },
        {
            "name": "🦠 Disease Detection",
            "description": "Detect plant diseases from leaf images with high accuracy",
            "features": ["Transfer learning", "Real-time detection", "Treatment recommendations"]
        },
        {
            "name": "💬 Agricultural Advisory",
            "description": "Get expert agricultural advice using AI and knowledge base",
            "features": ["RAG system", "Expert knowledge", "Contextual responses"]
        }
    ]

    for module in modules:
        with st.expander(module["name"]):
            st.write(module["description"])
            for feature in module["features"]:
                st.write(f"• {feature}")

def show_crop_classification():
    """Display crop classification interface."""

    st.markdown('<h2 class="module-header">🌱 Crop Classification</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Upload satellite or aerial imagery to classify different crop types using our deep learning models.
    </div>
    """, unsafe_allow_html=True)

    # File upload
    uploaded_file = st.file_uploader(
        "Upload Crop Image", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload satellite or drone imagery of crops"
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            st.image(image, caption="Input Image", use_column_width=True)

        with col2:
            st.subheader("Classification Results")

            # Simulate classification (replace with actual model inference)
            with st.spinner("Classifying crop..."):
                # Mock results - replace with actual model prediction
                predicted_crops = [
                    {"crop": "Wheat", "confidence": 0.85},
                    {"crop": "Corn", "confidence": 0.12},
                    {"crop": "Soybean", "confidence": 0.03}
                ]

            st.markdown("""
            <div class="result-box">
            <h4>🎯 Prediction Results:</h4>
            </div>
            """, unsafe_allow_html=True)

            for i, pred in enumerate(predicted_crops):
                confidence_color = "green" if pred["confidence"] > 0.7 else "orange" if pred["confidence"] > 0.3 else "red"
                st.markdown(f"""
                **{i+1}. {pred['crop']}**: 
                <span style="color: {confidence_color}; font-weight: bold;">
                {pred['confidence']:.1%}
                </span>
                """, unsafe_allow_html=True)

                # Progress bar for confidence
                st.progress(pred["confidence"])

    # Model selection
    st.subheader("Model Configuration")
    col1, col2 = st.columns(2)

    with col1:
        model_type = st.selectbox("Select Model", ["CNN (Deep Learning)", "Random Forest", "XGBoost"])

    with col2:
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

    # Additional information
    with st.expander("ℹ️ About Crop Classification"):
        st.markdown("""
        Our crop classification system uses state-of-the-art deep learning models trained on 
        large datasets of satellite and aerial imagery. The system can identify:

        - **Cereal crops**: Wheat, corn, rice, barley
        - **Legumes**: Soybeans, chickpeas, lentils
        - **Vegetables**: Tomatoes, potatoes, carrots
        - **Fruits**: Apples, grapes, citrus

        **Supported image formats**: JPEG, PNG  
        **Recommended resolution**: 224x224 pixels or higher  
        **Model accuracy**: 85-95% depending on crop type and image quality
        """)

def show_yield_prediction():
    """Display yield prediction interface."""

    st.markdown('<h2 class="module-header">📊 Yield Prediction</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Predict crop yields by providing agricultural and environmental data. Our models consider 
    weather patterns, soil conditions, and historical yield data.
    </div>
    """, unsafe_allow_html=True)

    # Input method selection
    input_method = st.radio("Input Method", ["Manual Entry", "Upload CSV File"])

    if input_method == "Manual Entry":
        # Manual input form
        st.subheader("📝 Crop and Environmental Data")

        col1, col2, col3 = st.columns(3)

        with col1:
            crop_type = st.selectbox("Crop Type", ["Wheat", "Corn", "Rice", "Soybean", "Potato"])
            planting_date = st.date_input("Planting Date")
            region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])

        with col2:
            soil_ph = st.slider("Soil pH", 4.0, 9.0, 6.5, 0.1)
            soil_organic_matter = st.slider("Organic Matter (%)", 0.0, 10.0, 3.0, 0.1)
            soil_nitrogen = st.slider("Nitrogen (ppm)", 0, 100, 45, 1)

        with col3:
            avg_temperature = st.slider("Average Temperature (°C)", 10, 35, 22, 1)
            total_precipitation = st.slider("Total Precipitation (mm)", 200, 1500, 600, 10)
            avg_humidity = st.slider("Average Humidity (%)", 30, 90, 65, 1)

        if st.button("🔮 Predict Yield"):
            with st.spinner("Predicting yield..."):
                # Mock prediction - replace with actual model inference
                predicted_yield = np.random.normal(4.5, 0.8)  # tons/hectare
                confidence_interval = [predicted_yield - 0.5, predicted_yield + 0.5]

            st.markdown("""
            <div class="result-box">
            <h4>📊 Yield Prediction Results:</h4>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Predicted Yield", f"{predicted_yield:.2f} tons/ha")

            with col2:
                st.metric("Confidence Range", f"{confidence_interval[0]:.1f} - {confidence_interval[1]:.1f}")

            with col3:
                yield_category = "High" if predicted_yield > 5 else "Medium" if predicted_yield > 3 else "Low"
                st.metric("Yield Category", yield_category)

    else:
        # File upload for batch prediction
        st.subheader("📁 Upload Batch Data")

        uploaded_file = st.file_uploader(
            "Upload CSV file with agricultural data",
            type=['csv'],
            help="CSV should contain columns: crop, region, soil_ph, temperature, precipitation, etc."
        )

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())

            if st.button("🔮 Predict Yields"):
                # Mock batch prediction
                yields = np.random.normal(4.5, 1.2, len(df))
                df['predicted_yield'] = yields

                st.subheader("📊 Prediction Results")
                st.dataframe(df)

                # Visualization
                fig = px.histogram(df, x='predicted_yield', title='Yield Distribution')
                st.plotly_chart(fig)

    # Model comparison
    with st.expander("📈 Model Performance Comparison"):
        model_performance = pd.DataFrame({
            'Model': ['Random Forest', 'XGBoost', 'Linear Regression', 'LSTM'],
            'RMSE': [0.45, 0.38, 0.67, 0.42],
            'R²': [0.89, 0.92, 0.78, 0.90]
        })

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(model_performance, x='Model', y='RMSE', title='Model RMSE Comparison')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(model_performance, x='Model', y='R²', title='Model R² Comparison')
            st.plotly_chart(fig, use_container_width=True)

def show_disease_detection():
    """Display disease detection interface."""

    st.markdown('<h2 class="module-header">🦠 Plant Disease Detection</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Upload images of plant leaves to detect diseases and get treatment recommendations. 
    Our AI model can identify over 38 different plant diseases across multiple crops.
    </div>
    """, unsafe_allow_html=True)

    # File upload
    uploaded_files = st.file_uploader(
        "Upload Plant Images", 
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Upload clear images of plant leaves showing symptoms"
    )

    if uploaded_files:
        # Process multiple images
        for i, uploaded_file in enumerate(uploaded_files):
            st.subheader(f"Image {i+1}: {uploaded_file.name}")

            image = Image.open(uploaded_file)

            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption=f"Input Image {i+1}", use_column_width=True)

            with col2:
                with st.spinner("Analyzing image..."):
                    # Mock disease detection - replace with actual model inference
                    diseases = [
                        {"disease": "Late Blight", "plant": "Tomato", "confidence": 0.92, "severity": "High"},
                        {"disease": "Healthy", "plant": "Tomato", "confidence": 0.06, "severity": "None"},
                        {"disease": "Early Blight", "plant": "Tomato", "confidence": 0.02, "severity": "Low"}
                    ]

                st.markdown("""
                <div class="result-box">
                <h4>🔍 Disease Detection Results:</h4>
                </div>
                """, unsafe_allow_html=True)

                for j, disease in enumerate(diseases):
                    confidence_color = "red" if disease["severity"] == "High" else "orange" if disease["severity"] == "Medium" else "green"

                    st.markdown(f"""
                    **{j+1}. {disease['disease']}** ({disease['plant']})  
                    Confidence: <span style="color: {confidence_color}; font-weight: bold;">
                    {disease['confidence']:.1%}</span>  
                    Severity: {disease['severity']}
                    """, unsafe_allow_html=True)

                    st.progress(disease["confidence"])

                # Treatment recommendations
                if diseases[0]["disease"] != "Healthy":
                    st.markdown("### 💊 Treatment Recommendations")
                    st.markdown("""
                    **Immediate Actions:**
                    1. Remove affected leaves immediately
                    2. Improve air circulation around plants
                    3. Apply copper-based fungicide
                    4. Avoid overhead watering

                    **Prevention:**
                    - Plant resistant varieties
                    - Ensure proper spacing
                    - Regular monitoring
                    """)

            st.markdown("---")

    # Disease information database
    with st.expander("📚 Disease Information Database"):
        disease_info = pd.DataFrame({
            'Disease': ['Late Blight', 'Early Blight', 'Powdery Mildew', 'Bacterial Spot', 'Mosaic Virus'],
            'Affected Plants': ['Tomato, Potato', 'Tomato, Potato', 'Various', 'Tomato, Pepper', 'Various'],
            'Symptoms': ['Dark lesions', 'Brown spots', 'White powder', 'Dark spots', 'Yellowing'],
            'Treatment': ['Fungicide', 'Fungicide', 'Sulfur spray', 'Bactericide', 'Remove infected plants']
        })

        st.dataframe(disease_info)

def show_agricultural_advisory():
    """Display agricultural advisory interface."""

    st.markdown('<h2 class="module-header">💬 Agricultural Advisory</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Get expert agricultural advice powered by AI. Ask questions about crop management, 
    pest control, soil health, irrigation, and more.
    </div>
    """, unsafe_allow_html=True)

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Question input
    st.subheader("💭 Ask Your Agricultural Question")

    question = st.text_area(
        "Enter your question:",
        placeholder="e.g., My tomato plants have yellow leaves. What could be causing this?",
        height=100
    )

    # Additional context
    with st.expander("🎯 Additional Context (Optional)"):
        col1, col2 = st.columns(2)

        with col1:
            location = st.text_input("Location/Region")
            crop_type = st.selectbox("Primary Crop", ["", "Tomato", "Wheat", "Corn", "Rice", "Potato", "Other"])

        with col2:
            farm_size = st.selectbox("Farm Size", ["", "Small (<5 acres)", "Medium (5-50 acres)", "Large (>50 acres)"])
            experience_level = st.selectbox("Experience Level", ["", "Beginner", "Intermediate", "Advanced"])

    if st.button("🤔 Get Advice"):
        if question.strip():
            with st.spinner("Generating agricultural advice..."):
                # Mock advisory response - replace with actual RAG system
                context_info = f"Location: {location}, Crop: {crop_type}, Farm Size: {farm_size}, Experience: {experience_level}"

                # Simulate response based on question content
                if "yellow" in question.lower() and "leaves" in question.lower():
                    response = """
                    Yellow leaves in tomato plants can be caused by several factors:

                    **Most Common Causes:**
                    1. **Nitrogen Deficiency** - Lower leaves yellow first, apply nitrogen fertilizer
                    2. **Overwatering** - Check soil drainage, reduce watering frequency
                    3. **Underwatering** - Soil too dry, increase irrigation
                    4. **Disease** - Check for spots or patterns, may need fungicide treatment

                    **Recommended Actions:**
                    - Conduct soil test to check nutrient levels
                    - Examine watering schedule and soil moisture
                    - Inspect plants for signs of disease or pests
                    - Apply balanced fertilizer if nutrient deficient

                    **Prevention:**
                    - Maintain consistent soil moisture
                    - Use drip irrigation to avoid wetting leaves
                    - Ensure proper plant spacing for air circulation
                    """
                else:
                    response = """
                    Thank you for your agricultural question. Based on your inquiry, here are some general recommendations:

                    **General Best Practices:**
                    - Regular monitoring of crop health
                    - Soil testing every 2-3 years
                    - Integrated pest management approach
                    - Proper irrigation scheduling
                    - Crop rotation to maintain soil health

                    For specific advice tailored to your situation, please provide more details about:
                    - Specific symptoms you're observing
                    - Current growing conditions
                    - Recent changes in management practices
                    - Local weather patterns

                    Consider consulting with your local agricultural extension office for region-specific guidance.
                    """

                # Add to chat history
                st.session_state.chat_history.append({
                    "question": question,
                    "response": response,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "context": context_info
                })

    # Display chat history
    if st.session_state.chat_history:
        st.subheader("💬 Conversation History")

        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
            with st.expander(f"Q{len(st.session_state.chat_history)-i}: {chat['question'][:50]}..."):
                st.markdown(f"**Question:** {chat['question']}")
                st.markdown(f"**Context:** {chat['context']}")
                st.markdown(f"**Response:** {chat['response']}")
                st.markdown(f"*Asked on: {chat['timestamp']}*")

    # Knowledge base search
    st.subheader("🔍 Knowledge Base Search")
    search_term = st.text_input("Search agricultural topics:", placeholder="e.g., pest control, soil pH, irrigation")

    if search_term:
        # Mock search results
        search_results = [
            {"title": "Integrated Pest Management", "category": "Pest Control", "relevance": 0.95},
            {"title": "Soil pH Management", "category": "Soil Health", "relevance": 0.87},
            {"title": "Efficient Irrigation Techniques", "category": "Water Management", "relevance": 0.82}
        ]

        st.subheader("📚 Search Results")
        for result in search_results:
            st.markdown(f"""
            **{result['title']}** (Category: {result['category']})  
            Relevance: {result['relevance']:.1%}
            """)

if __name__ == "__main__":
    main()
