import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime
import hashlib
import json
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import time
import random
from PIL import Image

# Custom CSS for beautiful design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border-left: 5px solid #667eea;
    }
    
    .crop-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        transition: transform 0.3s;
    }
    
    .crop-card:hover {
        transform: translateY(-5px);
    }
    
    .blockchain-card {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    
    .success-card {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
    }
    
    .prediction-badge {
        display: inline-block;
        padding: 0.25rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .tab-content {
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def get_optimal_ph(crop):
    """Get optimal pH range for crop"""
    ph_ranges = {
        'rice': '5.5-6.5',
        'maize': '5.8-7.0',
        'wheat': '6.0-7.5',
        'cotton': '5.5-7.5',
        'sugarcane': '6.0-7.5',
        'default': '6.0-7.0'
    }
    return ph_ranges.get(crop.lower(), ph_ranges['default'])

def get_optimal_temp(crop):
    """Get optimal temperature range for crop"""
    temp_ranges = {
        'rice': '20-35°C',
        'maize': '18-27°C',
        'wheat': '15-24°C',
        'default': '20-30°C'
    }
    return temp_ranges.get(crop.lower(), temp_ranges['default'])

def get_optimal_rainfall(crop):
    """Get optimal rainfall for crop"""
    rainfall_ranges = {
        'rice': '150-300 cm',
        'maize': '60-100 cm',
        'wheat': '30-100 cm',
        'default': '50-150 cm'
    }
    return rainfall_ranges.get(crop.lower(), rainfall_ranges['default'])

def get_optimal_npk(crop):
    """Get optimal NPK ratio for crop"""
    npk_ratios = {
        'rice': '4:2:1',
        'maize': '3:1:2',
        'wheat': '2:1:1',
        'default': '4:2:2'
    }
    return npk_ratios.get(crop.lower(), npk_ratios['default'])

class CropPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.crop_data = None
        self.features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        self.current_crop = None
        self.weather_api_key = None
        
    def load_data(self):
        """Load and preprocess the crop dataset"""
        try:
            data = pd.read_csv(r'C:/Users/varma/Desktop/agri/Crop_recommendation.csv')
            self.crop_data = data
            
            # Encode labels
            data['label_encoded'] = self.label_encoder.fit_transform(data['label'])
            
            return data
        except:
            # Create sample data if file not found
            st.warning("Sample dataset not found. Using synthetic data for demonstration.")
            np.random.seed(42)
            n_samples = 2200
            data = pd.DataFrame({
                'N': np.random.randint(0, 200, n_samples),
                'P': np.random.randint(5, 145, n_samples),
                'K': np.random.randint(5, 205, n_samples),
                'temperature': np.random.uniform(8, 44, n_samples),
                'humidity': np.random.uniform(15, 100, n_samples),
                'ph': np.random.uniform(3.5, 10, n_samples),
                'rainfall': np.random.uniform(20, 300, n_samples),
                'label': np.random.choice(['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 
                                          'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
                                          'banana', 'mango', 'grapes', 'watermelon', 'muskmelon',
                                          'apple', 'orange', 'papaya'], n_samples)
            })
            self.crop_data = data
            
            # Encode labels
            data['label_encoded'] = self.label_encoder.fit_transform(data['label'])
            
            return data
    
    def train_model(self):
        """Train the machine learning model"""
        if self.crop_data is None:
            self.load_data()
        
        X = self.crop_data[self.features]
        y = self.crop_data['label_encoded']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate accuracy
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy
    
    def predict_crops(self, input_data):
        """Predict top crops based on input conditions"""

        # -------- CRITICAL FIX --------
        if self.model is None:
            acc = self.train_model()
            st.info(f"Model trained automatically (Accuracy: {acc:.2%})")

        
        # Scale input data
        input_scaled = self.scaler.transform([input_data])
        
        # Get probabilities for all crops
        probabilities = self.model.predict_proba(input_scaled)[0]
        
        # Get top 5 predictions
        top_indices = probabilities.argsort()[-5:][::-1]
        top_crops = self.label_encoder.inverse_transform(top_indices)
        top_probs = probabilities[top_indices]
        
        predictions = []
        for crop, prob in zip(top_crops, top_probs):
            # Calculate suitability score (0-100)
            score = int(prob * 100)
            
            # Determine reason based on conditions
            reason = self._get_crop_reason(crop, input_data, score)
            
            # Calculate estimated profit
            profit = self._estimate_profit(crop, input_data)
            
            predictions.append({
                'crop': crop,
                'probability': prob,
                'score': score,
                'reason': reason,
                'estimated_profit': profit,
                'suitability': self._get_suitability_level(score)
            })
        
        return predictions
    
    def _get_crop_reason(self, crop, conditions, score):
        """Generate reason for crop recommendation"""
        reasons = {
            'rice': f"Rice requires high humidity ({conditions[4]:.1f}%) and temperature ({conditions[3]:.1f}°C) with moderate rainfall.",
            'maize': f"Maize thrives in moderate temperatures with good phosphorus levels ({conditions[1]} ppm).",
            'chickpea': f"Chickpea prefers lower humidity ({conditions[4]:.1f}%) and specific pH range ({conditions[5]:.1f}).",
            'kidneybeans': f"Kidneybeans need specific N:P:K ratio ({conditions[0]}:{conditions[1]}:{conditions[2]}) for optimal growth.",
            'pigeonpeas': f"Pigeonpeas are drought-resistant and suitable for current rainfall ({conditions[6]:.1f} mm).",
            'mothbeans': f"Mothbeans thrive in warm temperatures ({conditions[3]:.1f}°C) with moderate conditions.",
            'mungbean': f"Mungbean requires balanced nutrients and current conditions are optimal.",
            'blackgram': f"Blackgram needs specific soil conditions with pH around {conditions[5]:.1f}.",
            'lentil': f"Lentils prefer cooler temperatures and current climate is suitable.",
            'pomegranate': f"Pomegranate needs specific humidity ({conditions[4]:.1f}%) and temperature range.",
            'banana': f"Banana requires high potassium ({conditions[2]} ppm) and consistent moisture.",
            'mango': f"Mango trees need warm temperatures ({conditions[3]:.1f}°C) and good drainage.",
            'grapes': f"Grapes require specific temperature range and current conditions are ideal.",
            'watermelon': f"Watermelon needs warm climate and good water supply ({conditions[6]:.1f} mm rainfall).",
            'muskmelon': f"Muskmelon thrives in current temperature ({conditions[3]:.1f}°C) and humidity.",
            'apple': f"Apples prefer cooler temperatures and specific soil pH ({conditions[5]:.1f}).",
            'orange': f"Oranges need warm climate and current conditions are favorable.",
            'papaya': f"Papaya requires tropical conditions with good rainfall ({conditions[6]:.1f} mm)."
        }
        
        return reasons.get(crop, f"Suitable with {score}% match based on current soil and climate conditions.")
    
    def _estimate_profit(self, crop, conditions):
        """Estimate profit based on crop and conditions"""
        # Base profit per acre (in USD)
        base_profits = {
            'rice': 800, 'maize': 600, 'chickpea': 900, 'kidneybeans': 750,
            'pigeonpeas': 850, 'mothbeans': 500, 'mungbean': 550, 'blackgram': 650,
            'lentil': 700, 'pomegranate': 1200, 'banana': 1500, 'mango': 1800,
            'grapes': 2000, 'watermelon': 900, 'muskmelon': 800, 'apple': 1600,
            'orange': 1400, 'papaya': 1100
        }
        
        base_profit = base_profits.get(crop, 500)
        
        # Adjust based on conditions
        adjustment = 1.0
        if conditions[3] > 30:  # Temperature too high
            adjustment *= 0.9
        if conditions[4] > 85:  # Humidity too high
            adjustment *= 0.85
        if conditions[5] < 6 or conditions[5] > 7.5:  # pH not optimal
            adjustment *= 0.8
        
        # Market demand factor (simulated)
        demand_factor = 1.1 + (np.random.random() * 0.3)
        
        return int(base_profit * adjustment * demand_factor)
    
    def _get_suitability_level(self, score):
        """Get suitability level based on score"""
        if score >= 90:
            return "Excellent"
        elif score >= 75:
            return "Very Good"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Moderate"
        else:
            return "Low"
    
    def get_real_time_weather(self, city, api_key):
        """Get real-time weather data using OpenWeatherMap API"""
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
            response = requests.get(url)
            data = response.json()
            
            if data['cod'] == 200:
                weather_data = {
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    # Estimate rainfall from weather description
                    'rainfall': 0
                }
                
                if 'rain' in data:
                    weather_data['rainfall'] = data['rain'].get('1h', 0) * 24
                
                return weather_data
            else:
                return None
        except:
            return None

def main():
    # Initialize session state
# -------- FIX: Always use the correct predictor --------
    if 'predictor' not in st.session_state or not isinstance(st.session_state.predictor, CropPredictor):
        st.session_state.predictor = CropPredictor()
    # Ensure dataset is loaded before any analysis tab runs
    st.session_state.predictor.load_data()
# Ensure model is trained ONCE when page loads
    if st.session_state.predictor.model is None:
        with st.spinner("Training Crop Model..."):
            acc = st.session_state.predictor.train_model()
            st.success(f"Crop Model Ready! Accuracy: {acc:.2%}")

    if 'blockchain' not in st.session_state:
        st.session_state.blockchain = []  # Initialize as empty list
        st.session_state.last_block_hash = '0' * 64  # Initialize genesis block hash
    if 'predictions' not in st.session_state:
        st.session_state.predictions = []
    if 'selected_crop' not in st.session_state:
        st.session_state.selected_crop = None
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    st.markdown("""
        <style>
        .minimal-title {
            background: linear-gradient(90deg, 
                #2E7D32 0%, 
                #4CAF50 50%, 
                #8BC34A 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 3.5rem !important;
            font-weight: 900 !important;
            text-align: center !important;
            margin-bottom: 1rem !important;
            text-shadow: 0 2px 10px rgba(76, 175, 80, 0.2);
            letter-spacing: 0.5px;
            position: relative;
            display: inline-block;
            left: 50%;
            transform: translateX(-50%);
        }
        
        .minimal-title::after {
            content: '';
            position: absolute;
            bottom: -10px;
            left: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, 
                #4CAF50 0%, 
                #FFC107 50%, 
                #4CAF50 100%);
            border-radius: 2px;
        }
        
        .minimal-subtitle {
            color: #555 !important;
            font-size: 1.4rem !important;
            text-align: center !important;
            margin-bottom: 2.5rem !important;
            font-weight: 400 !important;
            padding: 0 2rem;
        }
        </style>
        
        <h1 class="minimal-title">FUTURE CROP PREDICTION SYSTEM</h1>
        <p class="minimal-subtitle">
            <strong>AI-Powered</strong> Smart Agriculture Decision System with Blockchain Traceability
        </p>
    """, unsafe_allow_html=True)


    # Main content with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Crop Prediction", "Analysis", "Blockchain Trace", "Settings"])
    
    with tab1:
        st.markdown("### Predict Optimal Future Crops")
        st.markdown("Enter soil parameters and get AI-powered crop recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Soil Parameters")
            
            # Soil parameters inputs
            n = st.slider("Nitrogen (N) - ppm", 0, 200, 90, help="Nitrogen level in parts per million")
            p = st.slider("Phosphorus (P) - ppm", 0, 200, 42, help="Phosphorus level in parts per million")
            k = st.slider("Potassium (K) - ppm", 0, 200, 43, help="Potassium level in parts per million")
            ph = st.slider("Soil pH", 0.0, 14.0, 6.5, 0.1, help="Soil acidity/alkalinity level")
        
        with col2:
            st.markdown("#### Climate Parameters")
            
            use_real_weather = st.checkbox("Use Real-time Weather Data")
            
            if use_real_weather:
                city = st.text_input("City Name", "Delhi", key="city_input")
                api_key_input = st.text_input("OpenWeatherMap API Key", type="password", key="api_key_input_weather")
                
                if st.button("Fetch Weather Data", key="fetch_weather"):
                    if api_key_input:
                        weather_data = st.session_state.predictor.get_real_time_weather(city, api_key_input)
                        if weather_data:
                            temp = weather_data['temperature']
                            humidity = weather_data['humidity']
                            rainfall = weather_data['rainfall']
                            st.success(f"Current: {temp}°C, {humidity}% humidity")
                            # Store values in session state
                            st.session_state.temp = temp
                            st.session_state.humidity = humidity
                            st.session_state.rainfall = rainfall
                        else:
                            st.error("Could not fetch weather data")
                            temp = st.number_input("Temperature (°C)", -10.0, 50.0, 25.0, 0.1, key="temp_input_weather")
                            humidity = st.slider("Humidity (%)", 0, 100, 70, key="humidity_input_weather")
                            rainfall = st.slider("Rainfall (mm)", 0, 500, 200, key="rainfall_input_weather")
                    else:
                        st.warning("Please enter API key")
                        temp = st.number_input("Temperature (°C)", -10.0, 50.0, 25.0, 0.1, key="temp_input_no_key")
                        humidity = st.slider("Humidity (%)", 0, 100, 70, key="humidity_input_no_key")
                        rainfall = st.slider("Rainfall (mm)", 0, 500, 200, key="rainfall_input_no_key")
                else:
                    temp = st.number_input("Temperature (°C)", -10.0, 50.0, 25.0, 0.1, key="temp_default")
                    humidity = st.slider("Humidity (%)", 0, 100, 70, key="humidity_default")
                    rainfall = st.slider("Rainfall (mm)", 0, 500, 200, key="rainfall_default")
            else:
                temp = st.number_input("Temperature (°C)", -10.0, 50.0, 25.0, 0.1, key="temp_manual")
                humidity = st.slider("Humidity (%)", 0, 100, 70, key="humidity_manual")
                rainfall = st.slider("Rainfall (mm)", 0, 500, 200, key="rainfall_manual")
        
        # Predict button
        if st.button("Predict Optimal Crops", use_container_width=True, key="predict_button"):
            with st.spinner("Analyzing conditions and predicting crops..."):
                # Prepare input data
                input_data = [n, p, k, temp, humidity, ph, rainfall]
                
                # Get predictions
                predictions = st.session_state.predictor.predict_crops(input_data)
                st.session_state.predictions = predictions
                
                st.success("Analysis complete! Top 5 crop recommendations:")
                
                # Display predictions
                for i, pred in enumerate(predictions, 1):
                    color_map = {
                        "Excellent": "#4CAF50",
                        "Very Good": "#8BC34A",
                        "Good": "#FFC107",
                        "Moderate": "#FF9800",
                        "Low": "#F44336"
                    }
                    
                    st.markdown(f"""
                    <div class="crop-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <h4 style="margin:0;">{i}. {pred['crop'].title()}</h4>
                                <span class="prediction-badge" style="background:{color_map[pred['suitability']]}; color:white;">
                                    {pred['suitability']} Suitability ({pred['score']}%)
                                </span>
                                <span class="prediction-badge" style="background:#2196F3; color:white;">
                                    ${pred['estimated_profit']}/acre
                                </span>
                            </div>
                        </div>
                        <p style="margin:0.5rem 0 0 0; color:#666;">{pred['reason']}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
        # Crop selection section
        if st.session_state.predictions:
            st.markdown("---")
            st.markdown("### Select Crop for Detailed Analysis")
            
            crop_options = [p['crop'] for p in st.session_state.predictions]
            selected = st.selectbox("Choose a crop to analyze further:", crop_options, key="crop_select")
            
            if st.button("Analyze Selected Crop", key="analyze_crop"):
                st.session_state.selected_crop = selected
                
                # Find the selected prediction
                selected_pred = next(p for p in st.session_state.predictions if p['crop'] == selected)
                
                # Display detailed analysis
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 2rem;">Suitability</div>
                        <div style="font-size: 1.5rem; font-weight: 600;">{selected_pred['score']}%</div>
                        <div>Suitability Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                

                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 2rem;">Ranking</div>
                        <div style="font-size: 1.5rem; font-weight: 600;">{crop_options.index(selected) + 1}/5</div>
                        <div>Ranking</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Crop-specific recommendations
                st.markdown("#### Crop-Specific Recommendations")
                st.info(f"""
                **Optimal Conditions for {selected.title()}:**
                - Maintain soil pH between {get_optimal_ph(selected)}
                - Target temperature range: {get_optimal_temp(selected)}
                - Required rainfall: {get_optimal_rainfall(selected)}
                - Fertilizer ratio (N:P:K): {get_optimal_npk(selected)}
                """)
    
    with tab2:
        st.markdown("### Data Analysis & Insights")
        
        # Safe dataset loading
        data = st.session_state.predictor.load_data()

        
        # Data visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Crop distribution
            crop_counts = data['label'].value_counts().reset_index()
            crop_counts.columns = ['Crop', 'Count']
            
            fig = px.bar(crop_counts, x='Crop', y='Count', 
                        title="Crop Distribution in Dataset",
                        color='Count', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Parameter distribution
            fig = px.box(data, y=['N', 'P', 'K'], 
                        title="Nutrient Distribution (N, P, K)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.markdown("#### Parameter Correlations")
        numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        corr_matrix = data[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=numeric_cols,
            y=numeric_cols,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=corr_matrix.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(title="Correlation Heatmap of Soil Parameters")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Blockchain Agri-Supply Chain")
        st.markdown("Track your produce from farm to consumer with immutable blockchain records")
        
        # Blockchain Dashboard
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.session_state.blockchain:
                total_transactions = len(st.session_state.blockchain)
                st.metric("Total Transactions", f"{total_transactions}")
            else:
                st.metric("Total Transactions", "0")
        with col2:
            # Count unique products from blockchain
            if st.session_state.blockchain:
                unique_products = len(set(block.get('product_id', '') for block in st.session_state.blockchain))
                st.metric("Products Tracked", f"{unique_products}")
            else:
                st.metric("Products Tracked", "0")
        with col3:
            st.metric("Verified Buyers", "346")
        
        # Product Registration Section
        st.markdown("---")
        st.markdown("## Product Registration on Blockchain")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Product Information")
            # Product registration form
            with st.form("product_registration"):
                product_name = st.text_input("Product Name *", placeholder="Enter product name", key="blockchain_product_name")
                product_type = st.selectbox("Product Type *", 
                                          ["Vegetables", "Fruits", "Grains", "Spices", "Dairy", "Other"],
                                          key="blockchain_product_type")
                batch_no = st.text_input("Batch Number *", placeholder="Enter batch number", key="blockchain_batch")
                harvest_date = st.date_input("Harvest Date *", key="blockchain_harvest")
                quantity = st.number_input("Quantity (kg) *", min_value=1, value=100, key="blockchain_quantity")
                certifications = st.multiselect("Certifications *", 
                                              ["Organic", "FSSAI", "APEDA", "Fair Trade", "ISO 22000"],
                                              key="blockchain_certifications")
                
                # Add to blockchain button
                submitted = st.form_submit_button("Register Product on Blockchain")
        
        with col2:
            st.markdown("### Blockchain Details")
            st.info("""
            **How it works:**
            1. Fill all product information fields (marked with *)
            2. Click 'Register Product on Blockchain'
            3. Your product will be registered with a unique blockchain hash
            4. Track your product through the supply chain
            
            **All fields are required** to create a blockchain transaction.
            """)
            
            # Display blockchain info if exists
            if st.session_state.blockchain:
                st.markdown("#### Recent Blockchain Entry")
                latest_block = st.session_state.blockchain[-1]
                st.write(f"**Product ID:** {latest_block.get('product_id', 'N/A')}")
                st.write(f"**Product Name:** {latest_block.get('product_name', 'N/A')}")
                st.write(f"**Blockchain Hash:** `{latest_block.get('hash', 'N/A')[:20]}...`")
                st.write(f"**Timestamp:** {latest_block.get('timestamp', 'N/A')}")
        
        # Process the form submission
        if submitted:
            # Validate all required fields
            validation_errors = []
            
            if not product_name or product_name.strip() == "":
                validation_errors.append("Product Name")
            
            if not batch_no or batch_no.strip() == "":
                validation_errors.append("Batch Number")
            
            if not certifications or len(certifications) == 0:
                validation_errors.append("Certifications (at least one)")
            
            if validation_errors:
                st.error(f"Please fill all required fields properly: {', '.join(validation_errors)}")
            else:
                # Generate unique product ID
                product_id = f"BLK{datetime.now().strftime('%Y%m%d')}{random.randint(1000, 9999)}"
                
                # Generate blockchain transaction
                block_data = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'product_id': product_id,
                    'product_name': product_name,
                    'product_type': product_type,
                    'batch_no': batch_no,
                    'harvest_date': str(harvest_date),
                    'quantity': f"{quantity} kg",
                    'certifications': certifications,
                    'action': 'PRODUCT_REGISTERED',
                    'farmer_id': 'FARM12345',
                    'farmer_name': 'Demo Farmer',
                    'previous_hash': st.session_state.last_block_hash,
                    'hash': hashlib.sha256(f"{product_id}{product_name}{batch_no}{datetime.now()}".encode()).hexdigest()
                }
                
                # Add to blockchain
                st.session_state.blockchain.append(block_data)
                st.session_state.last_block_hash = block_data['hash']
                
                st.success("Product Successfully Registered on Blockchain!")
                st.info(f"**Product ID:** {product_id}")
                st.info(f"**Blockchain Hash:** `{block_data['hash'][:64]}`")
                st.info(f"**Timestamp:** {block_data['timestamp']}")
        
        # Supply Chain Journey Visualization
        st.markdown("---")
        st.markdown("## Supply Chain Journey")
        
        # Sample supply chain data
        supply_chain = [
            {"stage": "Farm", "action": "Harvesting", "date": "2024-02-15", "location": "Thanjavur Farm", "verified": True},
            {"stage": "Processing", "action": "Cleaning & Sorting", "date": "2024-02-16", "location": "Chennai Facility", "verified": True},
            {"stage": "Packaging", "action": "Vacuum Packaging", "date": "2024-02-17", "location": "Packaging Unit", "verified": True},
            {"stage": "Logistics", "action": "Transport to Warehouse", "date": "2024-02-18", "location": "In Transit", "verified": True},
            {"stage": "Distribution", "action": "Retail Distribution", "date": "2024-02-19", "location": "Chennai Retail", "verified": True},
            {"stage": "Retail", "action": "Consumer Purchase", "date": "2024-02-20", "location": "Supermarket", "verified": False},
            {"stage": "Consumer", "action": "End Consumer", "date": "2024-02-21", "location": "Home", "verified": False}
        ]
        
        # Create timeline visualization using Streamlit columns
        for idx, stage in enumerate(supply_chain):
            col1, col2 = st.columns([0.1, 0.9])
            
            with col1:
                # Circle with number
                if stage['verified']:
                    st.markdown(f"""
                    <div style="width: 40px; height: 40px; border-radius: 50%; background: #4CAF50; 
                                display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                        {idx+1}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="width: 40px; height: 40px; border-radius: 50%; background: #FF9800; 
                                display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                        {idx+1}
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                # Stage details
                if stage['verified']:
                    st.markdown(f"""
                    <div style="padding: 10px; border-left: 3px solid #4CAF50; background: #E8F5E9; border-radius: 5px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between;">
                            <h4 style="margin: 0;">{stage['stage']} (Verified)</h4>
                            <span style="color: #666;">{stage['date']}</span>
                        </div>
                        <p style="margin: 5px 0;"><strong>Action:</strong> {stage['action']}</p>
                        <p style="margin: 5px 0;"><strong>Location:</strong> {stage['location']}</p>
                        <p style="margin: 0; color: #2E7D32;">
                            <strong>Status:</strong> Verified on Blockchain
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="padding: 10px; border-left: 3px solid #FF9800; background: #FFF3E0; border-radius: 5px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between;">
                            <h4 style="margin: 0;">{stage['stage']} (Pending)</h4>
                            <span style="color: #666;">{stage['date']}</span>
                        </div>
                        <p style="margin: 5px 0;"><strong>Action:</strong> {stage['action']}</p>
                        <p style="margin: 5px 0;"><strong>Location:</strong> {stage['location']}</p>
                        <p style="margin: 0; color: #F57C00;">
                            <strong>Status:</strong> Pending Verification
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Blockchain Explorer
        st.markdown("---")
        st.markdown("## Blockchain Explorer")
        
        if st.session_state.blockchain:
            # Convert to DataFrame for display
            blockchain_df = pd.DataFrame(st.session_state.blockchain)
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                search_transaction = st.text_input("Search Transaction Hash", key="search_hash")
            with col2:
                filter_type = st.selectbox("Filter by Action", ["All", "PRODUCT_REGISTERED", "TRANSFER", "VERIFICATION"], 
                                          key="filter_action")
            
            # Apply filters
            filtered_df = blockchain_df
            if search_transaction:
                filtered_df = filtered_df[filtered_df['hash'].str.contains(search_transaction, case=False, na=False)]
            if filter_type != "All":
                filtered_df = filtered_df[filtered_df['action'] == filter_type]
            
            # Display blockchain
            st.dataframe(filtered_df, use_container_width=True)
            
            # Visualize blockchain
            st.markdown("### Blockchain Visualization")
            
            # Create blockchain visualization
            blocks_html = "<div style='display: flex; overflow-x: auto; padding: 20px 0;'>"
            for idx, block in enumerate(st.session_state.blockchain[-10:] if st.session_state.blockchain else []):
                block_color = "#4CAF50" if idx % 2 == 0 else "#2196F3"
                blocks_html += f"""
                <div style='min-width: 200px; margin: 0 10px; padding: 15px; 
                            background: {block_color}; color: white; border-radius: 10px; text-align: center;'>
                    <h4 style='margin: 0 0 10px 0;'>Block #{idx+1}</h4>
                    <p style='font-size: 12px; margin: 5px 0;'>{block['timestamp']}</p>
                    <p style='font-size: 12px; margin: 5px 0;'><strong>{block.get('action', 'TRANSACTION')}</strong></p>
                    <p style='font-size: 10px; margin: 5px 0; word-break: break-all;'>
                        Hash: {block['hash'][:20]}...
                    </p>
                    <p style='font-size: 11px; margin: 5px 0;'>
                        Product: {block.get('product_name', 'N/A')}
                    </p>
                </div>
                """
            blocks_html += "</div>"
            
            st.markdown(blocks_html, unsafe_allow_html=True)
            
            # Blockchain statistics
            st.markdown("### Blockchain Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Blocks", len(st.session_state.blockchain))
            with col2:
                st.metric("Last Block Time", st.session_state.blockchain[-1]['timestamp'][11:19] if st.session_state.blockchain else "N/A")
            with col3:
                st.metric("Chain Integrity", "Verified" if len(st.session_state.blockchain) > 0 else "No Blocks")
        else:
            st.info("No blockchain transactions yet. Register a product to start!")
        
        # Smart Contracts Section
        st.markdown("---")
        st.markdown("## Smart Contracts")
        
        smart_contracts = [
            {"name": "Quality Assurance", "status": "Active", "address": "0x742d35Cc6634C0532925a3b8...", 
             "description": "Automatically verifies product quality standards"},
            {"name": "Payment Escrow", "status": "Active", "address": "0x21a31Ee1afC51d94C2eFca6...", 
             "description": "Holds payment until delivery confirmation"},
            {"name": "Carbon Credit Token", "status": "Development", "address": "0x3f5CE5FBFe3E...", 
             "description": "Issues carbon credits for sustainable farming"},
            {"name": "Supply Chain Finance", "status": "Active", "address": "0x4E83362442B8...", 
             "description": "Provides instant financing against blockchain inventory"}
        ]
        
        for contract in smart_contracts:
            with st.expander(f"{contract['name']} - {contract['status']}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Description:** {contract['description']}")
                    st.write(f"**Contract Address:** `{contract['address']}`")
                with col2:
                    status_color = "green" if contract['status'] == 'Active' else "orange"
                    st.markdown(f"<p style='color: {status_color};'><strong>Status:</strong> {contract['status']}</p>", 
                              unsafe_allow_html=True)
        
        # Market Price Oracle
        st.markdown("---")
        st.markdown("## Real-time Price Oracle")
        
        # Simulate fetching prices from blockchain oracle
        if st.button("Fetch Latest Prices from Oracle", key="fetch_prices"):
            with st.spinner("Fetching prices from blockchain oracle..."):
                time.sleep(2)
                
                # Simulated blockchain oracle data
                oracle_prices = {
                    'Rice': {'current': 2800, 'predicted': 2950, 'source': 'Blockchain Oracle'},
                    'Wheat': {'current': 2200, 'predicted': 2350, 'source': 'Smart Contract'},
                    'Tomato': {'current': 45, 'predicted': 50, 'source': 'Chainlink Oracle'},
                    'Cotton': {'current': 6800, 'predicted': 7200, 'source': 'Blockchain Oracle'}
                }
                
                st.success("Prices fetched from blockchain oracle!")
                
                # Display prices
                cols = st.columns(4)
                for idx, (commodity, data) in enumerate(oracle_prices.items()):
                    with cols[idx]:
                        st.metric(f"{commodity}", f"₹{data['current']}", 
                                 f"₹{data['predicted']} predicted")
                        st.caption(f"Source: {data['source']}")
    
    with tab4:
        st.markdown("### System Settings & Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Model Settings")
            
            if st.button("Retrain Model", key="retrain_model"):
                with st.spinner("Training model with latest data..."):
                    accuracy = st.session_state.predictor.train_model()
                    st.success(f"Model retrained! Accuracy: {accuracy:.2%}")
            
            st.markdown("#### API Configuration")
            api_key = st.text_input("OpenWeatherMap API Key", type="password", key="api_key_settings")
            if api_key:
                st.session_state.api_key = api_key
                st.session_state.predictor.weather_api_key = api_key
                st.success("API key configured successfully!")
        
        with col2:
            st.markdown("#### Data Management")
            
            if st.button("Load Sample Data", key="load_data"):
                st.session_state.predictor.load_data()
                st.success("Sample data loaded successfully!")
            

        st.markdown("---")
        st.markdown("#### About This System")
        st.info("""
        **AI-Powered Future Crop Predictor** v1.0
        - Uses machine learning to predict optimal crops
        - Considers soil health and climate conditions
        - Provides profit estimates based on market analysis
        - Blockchain traceability for premium products
        - Designed for sustainable agriculture
        """)
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("** Smart Agriculture**")
        st.caption("Data-driven farming decisions")
    with col2:
        st.markdown("** AI-Powered**")
        st.caption("Machine learning predictions")
    with col3:
        st.markdown("**Blockchain**")
        st.caption("Transparent supply chain")

if __name__ == "__main__":
    main()
