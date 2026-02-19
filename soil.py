
    
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import time
import random
import warnings
warnings.filterwarnings('ignore')
import io
from scipy import stats
import tempfile
import os
from pathlib import Path

def main():
# Remove default Streamlit padding and make full width
    # ===== TOP BACK BUTTON TO SOIL FERTIGATION =====
    col_back, _ = st.columns([1, 6])
    with col_back:
        if st.button("Back", key="back_to_soil_menu"):
            st.session_state.page = "soil"
            st.rerun()

    st.markdown("""
    <style>
        .main > div {
            padding-top: 0rem;
            padding-bottom: 0rem;
        }
        .stApp {
            max-width: 100% !important;
            padding: 0 !important;
        }
        header {
            display: none !important;
        }
        .stApp > header {
            display: none;
        }
        .stApp {
            margin-top: -80px;
        }
        .uploaded-data-table {
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 10px;
            background: white;
        }
        .csv-preview {
            font-size: 0.8rem;
        }
        .prediction-card {
            background: linear-gradient(135deg, #e3f2fd, #bbdefb);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin: 10px 0;
            border-left: 5px solid #1976d2;
        }
        .upload-section {
            background: #f5f5f5;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .full-width {
            width: 100%;
            padding: 0;
            margin: 0;
        }
        .main-header {
            font-size: 3.5rem !important;
            background: linear-gradient(90deg, #2E8B57, #3CB371);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.2rem !important;
            padding-top: 0.5rem !important;
        }
        .sub-header {
            color: #4682B4;
            font-size: 1.5rem !important;
            margin-bottom: 1rem !important;
            text-align: center;
        }
        .card {
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
            margin: 15px 5px;
            border-left: 5px solid #2E8B57;
            height: 100%;
        }
        .soil-card {
            background: linear-gradient(135deg, #8B4513, #A0522D);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 6px 12px rgba(0,0,0,0.2);
            margin: 15px 5px;
            border-left: 5px solid #D2691E;
            color: white;
            height: 100%;
        }
        .alert-card {
            background: linear-gradient(135deg, #ff6b6b, #ee5a52);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 6px 12px rgba(0,0,0,0.2);
            margin: 15px 5px;
            border-left: 5px solid #ff3838;
            color: white;
            animation: pulse 2s infinite;
            height: 100%;
        }
        .status-card {
            background: linear-gradient(135deg, #2c3e50, #34495e);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 6px 12px rgba(0,0,0,0.2);
            margin: 15px 5px;
            border-left: 5px solid #3498db;
            color: white;
            height: 100%;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(255, 56, 56, 0.7); }
            70% { box-shadow: 0 0 0 15px rgba(255, 56, 56, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 56, 56, 0); }
        }
        .status-on {
            color: #2E8B57;
            font-weight: bold;
            font-size: 1.2rem;
            animation: blink 1.5s infinite;
        }
        .status-off {
            color: #e74c3c;
            font-weight: bold;
            font-size: 1.2rem;
        }
        @keyframes blink {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #2E8B57, #3CB371);
            height: 8px;
            border-radius: 4px;
        }
        .stButton > button {
            background: linear-gradient(90deg, #2E8B57, #3CB371);
            color: white;
            border: none;
            padding: 12px 28px;
            border-radius: 10px;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s;
            width: 100%;
        }
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 15px rgba(46, 139, 87, 0.4);
        }
        .emergency-btn {
            background: linear-gradient(90deg, #e74c3c, #c0392b) !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 10px 10px 0 0;
            gap: 1px;
            padding: 10px 20px;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #2E8B57;
            color: white;
        }
        .metric-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border: 1px solid #e0e0e0;
            text-align: center;
            height: 140px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .metric-value {
            font-size: 2.2rem;
            font-weight: bold;
            color: #2E8B57;
            margin: 5px 0;
        }
        .metric-label {
            font-size: 1rem;
            color: #666;
            margin-bottom: 5px;
        }
        .metric-change {
            font-size: 0.9rem;
            font-weight: 600;
        }
        .soil-type-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 1rem;
            margin: 5px;
        }
        .prediction-badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 15px;
            font-size: 0.9rem;
            font-weight: 600;
            margin: 3px;
        }
    </style>
    """, unsafe_allow_html=True)

    # ============================================
    # 1. SOIL TYPE DATASET & CLASSES
    # ============================================

    # Define 5 soil types with comprehensive characteristics
    SOIL_TYPES = {
        'Sandy Soil': {
            'description': 'Low nutrient retention, fast drainage, requires frequent fertilization',
            'ideal_N': (25, 45),
            'ideal_P': (12, 25),
            'ideal_K': (15, 35),
            'ideal_moisture': (25, 45),
            'ideal_pH': (5.5, 6.5),
            'color': '#F4A460',
            'texture': 'Coarse, gritty',
            'drainage': 'Excellent',
            'water_holding': 'Poor'
        },
        'Clay Soil': {
            'description': 'High nutrient retention, poor drainage, requires careful water management',
            'ideal_N': (35, 55),
            'ideal_P': (20, 35),
            'ideal_K': (25, 45),
            'ideal_moisture': (35, 55),
            'ideal_pH': (6.0, 7.5),
            'color': '#8B4513',
            'texture': 'Fine, sticky when wet',
            'drainage': 'Poor',
            'water_holding': 'Excellent'
        },
        'Loamy Soil': {
            'description': 'Ideal balanced soil, perfect for most crops, optimal drainage and fertility',
            'ideal_N': (45, 65),
            'ideal_P': (25, 40),
            'ideal_K': (35, 55),
            'ideal_moisture': (40, 60),
            'ideal_pH': (6.0, 7.0),
            'color': '#CD853F',
            'texture': 'Medium, crumbly',
            'drainage': 'Good',
            'water_holding': 'Good'
        },
        'Silty Soil': {
            'description': 'Smooth texture, retains moisture well, good for water-loving plants',
            'ideal_N': (40, 60),
            'ideal_P': (15, 30),
            'ideal_K': (30, 50),
            'ideal_moisture': (45, 65),
            'ideal_pH': (5.5, 7.0),
            'color': '#BC8F8F',
            'texture': 'Smooth, silky',
            'drainage': 'Moderate',
            'water_holding': 'Very Good'
        },
        'Peaty Soil': {
            'description': 'High organic content, acidic, excellent for acid-loving plants',
            'ideal_N': (30, 50),
            'ideal_P': (10, 20),
            'ideal_K': (20, 40),
            'ideal_moisture': (55, 75),
            'ideal_pH': (4.5, 6.0),
            'color': '#2F4F4F',
            'texture': 'Spongy, fibrous',
            'drainage': 'Variable',
            'water_holding': 'Excellent'
        }
    }

    # Generate comprehensive dataset for training
    def generate_comprehensive_soil_dataset(n_samples=2000):
        """Generate synthetic dataset for soil type classification with realistic distributions"""
        data = []
        
        for soil_type, properties in SOIL_TYPES.items():
            for _ in range(n_samples // 5):
                # Base values around ideal ranges with normal distribution
                N_mean = np.mean(properties['ideal_N'])
                P_mean = np.mean(properties['ideal_P'])
                K_mean = np.mean(properties['ideal_K'])
                moisture_mean = np.mean(properties['ideal_moisture'])
                pH_mean = np.mean(properties['ideal_pH'])
                
                # Generate values with realistic variation
                N = np.random.normal(N_mean, 15)
                P = np.random.normal(P_mean, 12)
                K = np.random.normal(K_mean, 13)
                moisture = np.random.normal(moisture_mean, 12)
                temperature = np.random.normal(25, 5)  # Typical temperature range
                pH = np.random.normal(pH_mean, 0.5)
                
                # Ensure values are within realistic bounds
                N = np.clip(N, 5, 95)
                P = np.clip(P, 5, 85)
                K = np.clip(K, 5, 90)
                moisture = np.clip(moisture, 15, 85)
                temperature = np.clip(temperature, 12, 38)
                pH = np.clip(pH, 4.0, 8.5)
                
                # Add derived features
                nutrient_balance = abs(N - P) + abs(P - K) + abs(K - N)
                total_nutrients = N + P + K
                
                data.append({
                    'N': N,
                    'P': P,
                    'K': K,
                    'moisture': moisture,
                    'temperature': temperature,
                    'pH': pH,
                    'nutrient_balance': nutrient_balance,
                    'total_nutrients': total_nutrients,
                    'N_P_ratio': N / (P + 1),
                    'soil_type': soil_type,
                    'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 365))
                })
        
        return pd.DataFrame(data)

    # ============================================
    # 2. ML MODELS FOR SOIL IDENTIFICATION & PREDICTION
    # ============================================

    class AdvancedSoilIdentifier:
        def __init__(self):
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            self.is_trained = False
            self.feature_names = ['N', 'P', 'K', 'moisture', 'temperature', 'pH', 
                                 'nutrient_balance', 'total_nutrients', 'N_P_ratio']
            self.label_encoder = {}
            self.training_history = []
            self.last_training_time = None
            
        def train(self, df, auto_train=True):
            """Train soil type classifier automatically"""
            try:
                X = df[self.feature_names]
                y = df['soil_type']
                
                # Create label encoding
                self.label_encoder = {label: idx for idx, label in enumerate(sorted(y.unique()))}
                y_encoded = y.map(self.label_encoder)
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                )
                
                self.model.fit(X_train, y_train)
                
                # Calculate metrics
                y_pred = self.model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Get per-class accuracy
                report = classification_report(y_test, y_pred, target_names=list(self.label_encoder.keys()), output_dict=True)
                
                self.is_trained = True
                self.last_training_time = datetime.now()
                
                training_result = {
                    'accuracy': accuracy,
                    'n_samples': len(df),
                    'per_class_accuracy': {k: report[k]['precision'] for k in report if k in self.label_encoder},
                    'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_)),
                    'training_time': self.last_training_time,
                    'auto_trained': auto_train
                }
                
                self.training_history.append(training_result)
                
                return training_result
                
            except Exception as e:
                st.error(f"Training error: {str(e)}")
                return None
        
        def predict_soil_type(self, current_conditions):
            """Predict soil type from current conditions"""
            if not self.is_trained:
                # Return default prediction if not trained
                return "Loamy Soil", 0.5
            
            try:
                # Ensure we have all required features
                required_features = 9
                if len(current_conditions) < required_features:
                    # Add derived features
                    N, P, K, moisture, temperature, pH = current_conditions[:6]
                    nutrient_balance = abs(N - P) + abs(P - K) + abs(K - N)
                    total_nutrients = N + P + K
                    N_P_ratio = N / (P + 1)
                    current_conditions = current_conditions[:6] + [nutrient_balance, total_nutrients, N_P_ratio]
                
                prediction_encoded = self.model.predict([current_conditions])[0]
                prediction_proba = self.model.predict_proba([current_conditions])[0]
                
                # Get soil type from encoded value
                reverse_encoder = {v: k for k, v in self.label_encoder.items()}
                soil_type = reverse_encoder.get(prediction_encoded, "Loamy Soil")
                
                confidence = max(prediction_proba)
                
                return soil_type, confidence
                
            except:
                return "Loamy Soil", 0.5

    class ContinuousPredictionSystem:
        def __init__(self, buffer_size=500):
            self.data_buffer = pd.DataFrame(columns=[
                'timestamp', 'N', 'P', 'K', 'moisture', 'temperature', 'pH',
                'nutrient_balance', 'total_nutrients', 'N_P_ratio', 'soil_type_predicted'
            ])
            self.buffer_size = buffer_size
            self.predictions_history = []
            self.csv_file_path = None
            self.csv_last_modified = None
            self.csv_data = None
            self.last_processed_index = 0
            
        def set_csv_path(self, file_path):
            """Set the CSV file path and load initial data"""
            try:
                if not os.path.exists(file_path):
                    st.error(f"File not found: {file_path}")
                    return False
                
                self.csv_file_path = file_path
                self.csv_last_modified = os.path.getmtime(file_path)
                
                # Load initial data
                return self.load_csv_data()
                
            except Exception as e:
                st.error(f"Error setting CSV path: {str(e)}")
                return False
        
        def load_csv_data(self):
            """Load and parse CSV file for continuous prediction"""
            try:
                if self.csv_file_path and os.path.exists(self.csv_file_path):
                    # Read CSV file
                    df = pd.read_csv(self.csv_file_path)
                    
                    # Check for required columns
                    required_columns = ['N', 'P', 'K', 'moisture', 'temperature', 'pH']
                    if not all(col in df.columns for col in required_columns):
                        st.error(f"CSV must contain columns: {required_columns}")
                        return False
                    
                    self.csv_data = df
                    self.last_processed_index = 0
                    
                    # Process all available data
                    rows_added = 0
                    for i in range(len(df)):
                        row_added = self._add_csv_row_to_buffer(df.iloc[i])
                        if row_added:
                            rows_added += 1
                            self.last_processed_index = i + 1
                    
                    return True
                return False
                
            except Exception as e:
                st.error(f"Error loading CSV: {str(e)}")
                return False
        
        def _add_csv_row_to_buffer(self, row):
            """Add a single row from CSV to buffer"""
            try:
                # Calculate derived features
                nutrient_balance = abs(row['N'] - row['P']) + abs(row['P'] - row['K']) + abs(row['K'] - row['N'])
                total_nutrients = row['N'] + row['P'] + row['K']
                N_P_ratio = row['N'] / (row['P'] + 1) if row['P'] > 0 else 0
                
                # Add timestamp if not present
                if 'timestamp' in row and pd.notna(row['timestamp']):
                    try:
                        timestamp = pd.to_datetime(row['timestamp'])
                    except:
                        timestamp = datetime.now()
                else:
                    timestamp = datetime.now()
                
                # Add to buffer
                new_row = {
                    'timestamp': timestamp,
                    'N': row['N'],
                    'P': row['P'],
                    'K': row['K'],
                    'moisture': row['moisture'],
                    'temperature': row['temperature'],
                    'pH': row['pH'],
                    'nutrient_balance': nutrient_balance,
                    'total_nutrients': total_nutrients,
                    'N_P_ratio': N_P_ratio,
                    'soil_type_predicted': 'Unknown'
                }
                
                self.data_buffer = pd.concat([self.data_buffer, pd.DataFrame([new_row])], ignore_index=True)
                
                # Keep buffer size limited
                if len(self.data_buffer) > self.buffer_size:
                    self.data_buffer = self.data_buffer.iloc[-self.buffer_size:]
                
                return True
                
            except Exception as e:
                st.error(f"Error adding CSV row: {str(e)}")
                return False
        
        def check_for_updates(self):
            """Check if CSV file has been updated and process new rows"""
            try:
                if not self.csv_file_path or not os.path.exists(self.csv_file_path):
                    return False, 0
                
                current_mod_time = os.path.getmtime(self.csv_file_path)
                
                # Check if file has been modified
                if self.csv_last_modified is None or current_mod_time != self.csv_last_modified:
                    # Read the updated file
                    new_data = pd.read_csv(self.csv_file_path)
                    
                    # Check if there are new rows
                    if self.csv_data is not None and len(new_data) > len(self.csv_data):
                        # Get new rows
                        start_idx = len(self.csv_data)
                        new_rows = new_data.iloc[start_idx:]
                        
                        # Add new rows to buffer
                        rows_added = 0
                        for idx, row in new_rows.iterrows():
                            if self._add_csv_row_to_buffer(row):
                                rows_added += 1
                        
                        # Update file data
                        self.csv_data = new_data
                        self.last_processed_index = len(new_data)
                        self.csv_last_modified = current_mod_time
                        
                        if rows_added > 0:
                            return True, rows_added
                    
                    self.csv_last_modified = current_mod_time
                
                return False, 0
                
            except Exception as e:
                st.error(f"Error checking for updates: {str(e)}")
                return False, 0
        
        def predict_future(self, hours_ahead=24):
            """Predict future values based on current data patterns"""
            if len(self.data_buffer) < 10:
                return None
            
            try:
                predictions = {}
                for nutrient in ['N', 'P', 'K', 'moisture']:
                    # Get recent data
                    recent = self.data_buffer[nutrient].tail(50).values
                    
                    if len(recent) < 10:
                        continue
                    
                    # Use moving average + trend for prediction
                    window = min(10, len(recent))
                    current = np.mean(recent[-window:])
                    
                    # Calculate trend from last 20 values
                    if len(recent) >= 20:
                        trend_values = recent[-20:]
                        time_indices = np.arange(len(trend_values))
                        slope, _, _, _, _ = stats.linregress(time_indices, trend_values)
                    else:
                        slope = 0
                    
                    # Predict future value
                    predicted = current + (slope * (hours_ahead * 6))
                    
                    # Ensure predictions are within realistic bounds
                    if nutrient == 'N':
                        predicted = np.clip(predicted, 5, 95)
                    elif nutrient == 'P':
                        predicted = np.clip(predicted, 5, 85)
                    elif nutrient == 'K':
                        predicted = np.clip(predicted, 5, 90)
                    elif nutrient == 'moisture':
                        predicted = np.clip(predicted, 15, 85)
                    
                    # Calculate confidence based on data quality
                    volatility = np.std(recent[-10:]) if len(recent) >= 10 else 10
                    confidence = max(0.3, min(0.95, 1 - (volatility / 50)))
                    
                    predictions[nutrient] = {
                        'current': current,
                        'predicted': predicted,
                        'change': predicted - current,
                        'trend': 'UP' if slope > 0.01 else 'DOWN' if slope < -0.01 else 'STABLE',
                        'confidence': confidence,
                        'volatility': volatility,
                        'prediction_time': datetime.now() + timedelta(hours=hours_ahead)
                    }
                
                return predictions
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                return None

    # ============================================
    # 3. INITIALIZE SYSTEM
    # ============================================

    # Initialize session state
    if 'soil_identifier' not in st.session_state:
        st.session_state.soil_identifier = AdvancedSoilIdentifier()
        
    if 'prediction_system' not in st.session_state:
        st.session_state.prediction_system = ContinuousPredictionSystem(buffer_size=1000)
        
    if 'soil_dataset' not in st.session_state:
        # Generate and train on initial soil dataset
        with st.spinner("Generating soil dataset and training models..."):
            soil_dataset = generate_comprehensive_soil_dataset(2500)
            st.session_state.soil_dataset = soil_dataset
            st.session_state.soil_identifier.train(soil_dataset, auto_train=True)
        
    if 'csv_loaded' not in st.session_state:
        st.session_state.csv_loaded = False
        
    if 'csv_monitoring' not in st.session_state:
        st.session_state.csv_monitoring = False
        
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now()

    # Initialize models
    soil_identifier = st.session_state.soil_identifier
    prediction_system = st.session_state.prediction_system

    # ============================================
    # 4. FULL-PAGE APPLICATION UI
    # ============================================

    # MAIN HEADER (Full Width)
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
            
        <h1 class="minimal-title">SMART SOIL MANUAL SYSTEM</h1>
    """, unsafe_allow_html=True)

    # Create full-width tabs
    tab1, tab2 = st.tabs(["SOIL TYPES", "MANUAL SOIL ANALYSIS"])

    # TAB 1: SOIL ANALYSIS
    with tab1:
        st.markdown("## SOIL TYPE ANALYSIS & IDENTIFICATION")
        
        # Row 1: Soil Types Overview
        st.markdown("### SOIL TYPES DATABASE")
        
        soil_cols = st.columns(5)
        for idx, (soil_name, properties) in enumerate(SOIL_TYPES.items()):
            with soil_cols[idx]:
                st.markdown(f'<div style="background-color: {properties["color"]}20; padding: 15px; border-radius: 10px; border-left: 5px solid {properties["color"]};">', unsafe_allow_html=True)
                st.markdown(f'<h4 style="color: {properties["color"]};">{soil_name}</h4>', unsafe_allow_html=True)
                st.write(f"**{properties['description']}**")
                st.write(f"pH: {properties['ideal_pH'][0]}-{properties['ideal_pH'][1]}")
                st.write(f"Moisture: {properties['ideal_moisture'][0]}-{properties['ideal_moisture'][1]}%")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Row 2: Model Performance
        st.markdown("### ML MODEL PERFORMANCE")
        
        if soil_identifier.is_trained:
            perf_cols = st.columns(3)
            

            
            with perf_cols[0]:
                
                st.markdown('<div class="metric-label">TRAINING SAMPLES</div>', unsafe_allow_html=True)
                n_samples = soil_identifier.training_history[-1]['n_samples']
                st.markdown(f'<div class="metric-value">{n_samples:,}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with perf_cols[1]:
                
                st.markdown('<div class="metric-label">SOIL TYPES</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{len(SOIL_TYPES)}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with perf_cols[2]:
                
                st.markdown('<div class="metric-label">LAST TRAINED</div>', unsafe_allow_html=True)
                last_train = soil_identifier.last_training_time.strftime('%m/%d %H:%M')
                st.markdown(f'<div class="metric-value">{last_train}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                    # Feature Importance
        st.markdown("### FEATURE IMPORTANCE")
        
        fig_importance = go.Figure(data=[
            go.Bar(
                x=list(soil_identifier.training_history[-1]['feature_importance'].keys()),
                y=list(soil_identifier.training_history[-1]['feature_importance'].values()),
                marker_color=['#2E8B57', '#4682B4', '#DAA520', '#1E90FF', '#DC143C', '#9370DB', '#FF6347', '#40E0D0', '#FF69B4']
            )
        ])
        fig_importance.update_layout(
            title="Which Factors Most Influence Soil Type Identification?",
            xaxis_title="Features",
            yaxis_title="Importance",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    
    with tab2:
        st.markdown("### MANUAL SOIL TESTING")
        
        test_cols = st.columns(2)
        
        with test_cols[0]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### ENTER SOIL PARAMETERS")
            
            test_N = st.slider("Nitrogen (N) ppm", 0, 100, 55, key="test_n")
            test_P = st.slider("Phosphorus (P) ppm", 0, 100, 32, key="test_p")
            test_K = st.slider("Potassium (K) ppm", 0, 100, 42, key="test_k")
            test_moisture = st.slider("Moisture %", 0, 100, 52, key="test_m")
            test_temp = st.slider("Temperature °C", 10, 40, 26, key="test_t")
            test_pH = st.slider("pH Level", 4.0, 9.0, 6.8, key="test_ph")
            
            if st.button(" ANALYZE THIS SOIL", use_container_width=True):
                test_conditions = [test_N, test_P, test_K, test_moisture, test_temp, test_pH]
                predicted_soil, confidence = soil_identifier.predict_soil_type(test_conditions)
                
                st.session_state.test_result = {
                    'soil_type': predicted_soil,
                    'confidence': confidence,
                    'conditions': test_conditions
                }
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        with test_cols[1]:
            st.markdown('<div class="soil-card">', unsafe_allow_html=True)
            st.markdown("#### TEST RESULTS")
            
            if 'test_result' in st.session_state:
                result = st.session_state.test_result
                
                st.markdown(f"### {result['soil_type']}")
                st.write(f"**Description:** {SOIL_TYPES[result['soil_type']]['description']}")
                
                st.markdown("##### Soil Characteristics:")
                soil_props = SOIL_TYPES[result['soil_type']]
                st.write(f"**Texture:** {soil_props['texture']}")
                st.write(f"**Drainage:** {soil_props['drainage']}")
                st.write(f"**Water Holding:** {soil_props['water_holding']}")
                st.write(f"**Ideal pH:** {soil_props['ideal_pH'][0]} - {soil_props['ideal_pH'][1]}")
            else:
                st.info("Enter parameters and click 'Analyze This Soil' to see results")
            st.markdown('</div>', unsafe_allow_html=True)

 
    # ============================================
    # 5. CONTINUOUS MONITORING WITH AUTO-REFRESH
    # ============================================

    # Add Streamlit auto-refresh mechanism for CSV monitoring only
    if st.session_state.csv_loaded and st.session_state.csv_monitoring:
        # Calculate time since last update
        current_time = datetime.now()
        time_since_update = (current_time - st.session_state.last_update).total_seconds()
        
        # If it's been more than 2 seconds, trigger a rerun to check for updates
        if time_since_update >= 2:
            # This will cause Streamlit to rerun the script
            st.session_state.last_update = current_time
            st.rerun()



