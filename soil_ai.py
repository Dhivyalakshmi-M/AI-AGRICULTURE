import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import glob
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class SoilHealthPredictor:
    def __init__(self):
        self.data_history = pd.DataFrame()
        self.predictions_history = pd.DataFrame()
        self.models = None
        self.scaler = None
        self.label_encoders = {}
        self.watch_folder = "soil_data"
        self.processed_files = set()  # Track processed files
        
        # Create watch folder if it doesn't exist
        if not os.path.exists(self.watch_folder):
            os.makedirs(self.watch_folder)
            st.info(f"Created folder: {self.watch_folder}")
        
        # Train or load model
        self.initialize_model()
    
    def generate_training_data(self):
        """Generate comprehensive training data"""
        np.random.seed(42)
        n_samples = 5000
        
        data = {
            'N': np.random.randint(10, 100, n_samples),
            'P': np.random.randint(10, 80, n_samples),
            'K': np.random.randint(10, 90, n_samples),
            'moisture': np.random.randint(20, 90, n_samples),
            'temperature': np.random.uniform(15, 35, n_samples),
            'pH': np.random.uniform(4.5, 9.0, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Advanced soil health calculation
        def calculate_soil_score(row):
            score = 0
            
            # Nutrient optimization (ideal ranges)
            if 45 <= row['N'] <= 65: score += 2
            elif 35 <= row['N'] <= 75: score += 1
            
            if 25 <= row['P'] <= 45: score += 2
            elif 20 <= row['P'] <= 50: score += 1
            
            if 35 <= row['K'] <= 55: score += 2
            elif 30 <= row['K'] <= 60: score += 1
            
            # Environmental factors
            if 6.0 <= row['pH'] <= 7.5: score += 2
            elif 5.5 <= row['pH'] <= 8.0: score += 1
            
            if 40 <= row['moisture'] <= 70: score += 2
            elif 30 <= row['moisture'] <= 80: score += 1
            
            if 20 <= row['temperature'] <= 30: score += 2
            elif 18 <= row['temperature'] <= 32: score += 1
            
            return score
        
        def calculate_nutrient_balance(row):
            balance = []
            
            # N deficiency
            n_ratio = row['N'] / (row['P'] + row['K'] + 1)
            if n_ratio < 0.3:
                balance.append('Low_N')
            elif n_ratio > 0.7:
                balance.append('High_N')
            
            # P deficiency
            p_ratio = row['P'] / (row['N'] + row['K'] + 1)
            if p_ratio < 0.2:
                balance.append('Low_P')
            elif p_ratio > 0.5:
                balance.append('High_P')
            
            # K deficiency
            k_ratio = row['K'] / (row['N'] + row['P'] + 1)
            if k_ratio < 0.25:
                balance.append('Low_K')
            elif k_ratio > 0.6:
                balance.append('High_K')
            
            # pH issues
            if row['pH'] < 5.5:
                balance.append('Acidic')
            elif row['pH'] > 7.8:
                balance.append('Alkaline')
            
            # Temperature stress
            if row['temperature'] > 32:
                balance.append('Heat_Stress')
            elif row['temperature'] < 18:
                balance.append('Cold_Stress')
            
            # Moisture stress
            if row['moisture'] < 35:
                balance.append('Dry')
            elif row['moisture'] > 75:
                balance.append('Waterlogged')
            
            return ','.join(balance) if balance else 'Optimal'
        
        # Calculate scores and labels
        df['soil_score'] = df.apply(calculate_soil_score, axis=1)
        df['nutrient_balance'] = df.apply(calculate_nutrient_balance, axis=1)
        
        # Classify soil health based on score
        df['soil_health'] = 'Poor'  # Default value
        df.loc[df['soil_score'] >= 10, 'soil_health'] = 'Excellent'
        df.loc[(df['soil_score'] >= 7) & (df['soil_score'] < 10), 'soil_health'] = 'Good'
        df.loc[(df['soil_score'] >= 4) & (df['soil_score'] < 7), 'soil_health'] = 'Fair'
        df.loc[df['soil_score'] < 4, 'soil_health'] = 'Poor'
        
        return df
    
    def initialize_model(self):
        """Initialize or train the ML model"""
        model_path = 'soil_model.pkl'
        scaler_path = 'scaler.pkl'
        encoder_path = 'encoders.pkl'
        
        if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoder_path):
            try:
                self.models = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.label_encoders = joblib.load(encoder_path)
                return True
            except:
                st.warning("Error loading models, retraining...")
        
        # Train new models
        with st.spinner("Training models... This may take a moment."):
            df = self.generate_training_data()
            
            # Features and targets
            features = ['N', 'P', 'K', 'moisture', 'temperature', 'pH']
            X = df[features]
            
            # Encode labels
            le_health = LabelEncoder()
            le_nutrient = LabelEncoder()
            
            y_health = le_health.fit_transform(df['soil_health'])
            y_nutrient = le_nutrient.fit_transform(df['nutrient_balance'])
            
            # Split data
            X_train, X_test, y_health_train, y_health_test, y_nutrient_train, y_nutrient_test = train_test_split(
                X, y_health, y_nutrient, test_size=0.2, random_state=42
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train ensemble model for health prediction
            health_rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
            health_gb = GradientBoostingClassifier(n_estimators=150, random_state=42)
            
            # Train nutrient model
            nutrient_rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
            
            health_rf.fit(X_train_scaled, y_health_train)
            health_gb.fit(X_train_scaled, y_health_train)
            nutrient_rf.fit(X_train_scaled, y_nutrient_train)
            
            # Store models
            self.models = {
                'health_rf': health_rf,
                'health_gb': health_gb,
                'nutrient_rf': nutrient_rf
            }
            
            # Store encoders
            self.label_encoders = {
                'health': le_health,
                'nutrient': le_nutrient
            }
            
            # Save models
            joblib.dump(self.models, model_path)
            joblib.dump(self.scaler, scaler_path)
            joblib.dump(self.label_encoders, encoder_path)
            
            # Calculate accuracy
            health_rf_acc = health_rf.score(X_test_scaled, y_health_test)
            health_gb_acc = health_gb.score(X_test_scaled, y_health_test)
            nutrient_acc = nutrient_rf.score(X_test_scaled, y_nutrient_test)
            
            st.success(f"Model training complete. Accuracies - Health RF: {health_rf_acc:.3f}, Health GB: {health_gb_acc:.3f}, Nutrient: {nutrient_acc:.3f}")
            return True
    
    def predict_soil_health(self, features):
        """Make predictions with confidence scores"""
        features_scaled = self.scaler.transform(features)
        
        # Health prediction with ensemble
        health_rf_proba = self.models['health_rf'].predict_proba(features_scaled)
        health_gb_proba = self.models['health_gb'].predict_proba(features_scaled)
        
        # Ensemble prediction (weighted average)
        ensemble_proba = (health_rf_proba + health_gb_proba) / 2
        health_pred = np.argmax(ensemble_proba, axis=1)
        
        # Nutrient prediction
        nutrient_pred = self.models['nutrient_rf'].predict(features_scaled)
        nutrient_proba = self.models['nutrient_rf'].predict_proba(features_scaled)
        
        # Decode predictions
        health_label = self.label_encoders['health'].inverse_transform(health_pred)[0]
        nutrient_label = self.label_encoders['nutrient'].inverse_transform(nutrient_pred)[0]
        
        # Calculate confidence scores
        health_confidence = np.max(ensemble_proba) * 100
        nutrient_confidence = np.max(nutrient_proba) * 100
        
        # Get individual class probabilities
        health_classes = self.label_encoders['health'].classes_
        health_probabilities = {cls: float(prob) for cls, prob in zip(health_classes, ensemble_proba[0])}
        
        return {
            'soil_health': health_label,
            'nutrient_status': nutrient_label,
            'health_confidence': round(float(health_confidence), 2),
            'nutrient_confidence': round(float(nutrient_confidence), 2),
            'health_probabilities': health_probabilities
        }
    
    def process_csv_file(self, filepath):
        """Process a CSV file and make predictions"""
        try:
            df = pd.read_csv(filepath)
            
            # Check required columns
            required_cols = ['N', 'P', 'K', 'moisture', 'temperature', 'pH']
            if not all(col in df.columns for col in required_cols):
                st.error(f"Missing required columns in {filepath}. Required: {required_cols}")
                return None
            
            predictions = []
            for idx, row in df.iterrows():
                # Prepare features - handle both int and float
                features = np.array([[
                    float(row['N']), float(row['P']), float(row['K']),
                    float(row['moisture']), float(row['temperature']), float(row['pH'])
                ]])
                
                # Make prediction
                prediction = self.predict_soil_health(features)
                
                # Create result dictionary
                result = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'filename': os.path.basename(filepath),
                    'row_index': idx,
                    'N': float(row['N']),
                    'P': float(row['P']),
                    'K': float(row['K']),
                    'moisture': float(row['moisture']),
                    'temperature': float(row['temperature']),
                    'pH': float(row['pH'])
                }
                result.update(prediction)
                
                predictions.append(result)
            
            return predictions
            
        except Exception as e:
            st.error(f"Error processing {filepath}: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def scan_folder(self):
        """Scan folder for new CSV files and process them"""
        try:
            # Get all CSV files in the folder
            csv_files = glob.glob(os.path.join(self.watch_folder, '*.csv'))
            
            if not csv_files:
                return 0
            
            all_predictions = []
            new_files_found = 0
            
            for filepath in sorted(csv_files):
                # Get absolute path
                filepath = os.path.abspath(filepath)
                
                # Check if file is already processed
                if filepath in self.processed_files:
                    continue
                
                # Check if file exists and is readable
                if not os.path.exists(filepath):
                    continue
                
                # Try to read the file to check if it's valid
                try:
                    with open(filepath, 'r') as f:
                        # Just check if we can read it
                        pass
                except:
                    continue
                
                # Process the file
                predictions = self.process_csv_file(filepath)
                if predictions:
                    all_predictions.extend(predictions)
                    self.processed_files.add(filepath)
                    new_files_found += 1
                    st.info(f"Processed: {os.path.basename(filepath)}")
            
            # Update history
            if all_predictions:
                new_df = pd.DataFrame(all_predictions)
                self.predictions_history = pd.concat([self.predictions_history, new_df], ignore_index=True)
                
                # Remove duplicates
                self.predictions_history = self.predictions_history.drop_duplicates(
                    subset=['filename', 'row_index'], keep='last'
                )
                
                # Limit history size
                if len(self.predictions_history) > 1000:
                    self.predictions_history = self.predictions_history.tail(1000)
            
            return new_files_found
            
        except Exception as e:
            st.error(f"Error scanning folder: {str(e)}")
            return 0
def main():
    
    
    # ===== TOP BACK BUTTON TO SOIL FERTIGATION =====
    col_back, _ = st.columns([1, 6])
    with col_back:
        if st.button("Back", key="back_to_soil_menu_ai"):
            st.session_state.page = "soil"
            st.rerun()

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
        
        <h1 class="minimal-title">SOIL AI HEALTH PREDICTION SYSTEM</h1>
        <p class="minimal-subtitle">
            <strong>AI-Powered</strong>Continuously monitors CSV files for soil data and makes predictions
        </p>
    """, unsafe_allow_html=True)
    
    # Initialize predictor
# ===== FIX: Use a SEPARATE predictor for Soil AI =====
    if 'soil_ai_predictor' not in st.session_state or not isinstance(st.session_state.soil_ai_predictor, SoilHealthPredictor):
        st.session_state.soil_ai_predictor = SoilHealthPredictor()

    predictor = st.session_state.soil_ai_predictor

    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Live Monitoring", "Predictions History", "Upload CSV"])
    
    with tab1:
        st.header("Live Monitoring")
        
        # Refresh button and status
        col1, col2, col3 = st.columns(3)
        with col1:
            refresh_btn = st.button("Scan Folder Now")
        with col2:
            interval = st.selectbox("Auto-refresh (seconds)", [2, 5, 10, 30, 60], index=0)
        with col3:
            if st.button("Clear History"):
                predictor.predictions_history = pd.DataFrame()
                predictor.processed_files.clear()
                st.success("History cleared")
                st.rerun()
        
        # Folder info
        st.write(f"**Monitoring folder:** `{predictor.watch_folder}`")
        
        # Show folder contents
        if os.path.exists(predictor.watch_folder):
            files = os.listdir(predictor.watch_folder)
            csv_files = [f for f in files if f.lower().endswith('.csv')]
            st.write(f"**CSV files in folder:** {len(csv_files)}")
            if csv_files:
                st.write("Files found:", ", ".join(csv_files[:5]))
                if len(csv_files) > 5:
                    st.write(f"... and {len(csv_files) - 5} more")
        
        st.write(f"**Total predictions:** {len(predictor.predictions_history)}")
        st.write(f"**Processed files:** {len(predictor.processed_files)}")
        
        # ALWAYS scan folder automatically - this is the key change
        # Use a session state to track last scan time
        if 'last_scan_time' not in st.session_state:
            st.session_state.last_scan_time = 0
        
        current_time = time.time()
        time_since_last_scan = current_time - st.session_state.last_scan_time
        
        # Auto-scan if interval has passed or button clicked
        if time_since_last_scan > int(interval) or refresh_btn:
            with st.spinner("Scanning for new CSV files..."):
                num_new = predictor.scan_folder()
                st.session_state.last_scan_time = current_time
                if num_new > 0:
                    st.success(f"Processed {num_new} new CSV files")
                elif refresh_btn:
                    st.info("No new CSV files found")
        

            
            # Statistics
            st.subheader("Prediction Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'health_confidence' in predictor.predictions_history.columns:
                    avg_confidence = predictor.predictions_history['health_confidence'].mean()
                    st.metric("Avg Health Confidence", f"{avg_confidence:.1f}%")
                else:
                    st.metric("Avg Health Confidence", "N/A")
            
            with col2:
                if 'soil_health' in predictor.predictions_history.columns:
                    most_common = predictor.predictions_history['soil_health'].mode()
                    if len(most_common) > 0:
                        st.metric("Most Common Health", most_common[0])
                    else:
                        st.metric("Most Common Health", "N/A")
                else:
                    st.metric("Most Common Health", "N/A")
            
            with col3:
                if 'filename' in predictor.predictions_history.columns:
                    unique_files = predictor.predictions_history['filename'].nunique()
                    st.metric("Files Processed", unique_files)
                else:
                    st.metric("Files Processed", 0)
            
            with col4:
                total_rows = len(predictor.predictions_history)
                st.metric("Total Predictions", total_rows)
        
        else:
            st.info("No predictions yet. Add CSV files to the soil_data folder or upload below.")
            st.info("To add files: Save CSV files in the 'soil_data' folder, then click 'Scan Folder Now'")
    
   
    with tab2:
        st.header("Predictions History")
        
        if not predictor.predictions_history.empty:
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                if 'soil_health' in predictor.predictions_history.columns:
                    health_filter = st.multiselect(
                        "Filter by Health Status",
                        options=predictor.predictions_history['soil_health'].unique(),
                        default=predictor.predictions_history['soil_health'].unique()
                    )
                else:
                    health_filter = []
                    st.info("No health data available")
            
            with col2:
                if 'health_confidence' in predictor.predictions_history.columns:
                    confidence_threshold = st.slider(
                        "Min Health Confidence (%)",
                        min_value=0, max_value=100, value=70
                    )
                else:
                    confidence_threshold = 0
                    st.info("No confidence data available")
            
            with col3:
                date_sort = st.selectbox(
                    "Sort by",
                    ["Newest First", "Oldest First"]
                )
            
            # Apply filters
            filtered_df = predictor.predictions_history.copy()
            if health_filter and 'soil_health' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['soil_health'].isin(health_filter)]
            
            if 'health_confidence' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['health_confidence'] >= confidence_threshold]
            
            # Sort
            if date_sort == "Newest First":
                filtered_df = filtered_df.sort_values('timestamp', ascending=False)
            else:
                filtered_df = filtered_df.sort_values('timestamp', ascending=True)
            
            # Display
            st.write(f"Showing {len(filtered_df)} of {len(predictor.predictions_history)} predictions")
            
            # Detailed view
            if st.checkbox("Show Detailed View"):
                st.dataframe(filtered_df, use_container_width=True, height=500)
            else:
                # Summary view
                summary_cols = ['timestamp', 'filename', 'soil_health', 'health_confidence', 
                              'nutrient_status', 'N', 'P', 'K']
                # Only show columns that exist
                summary_cols = [col for col in summary_cols if col in filtered_df.columns]
                st.dataframe(filtered_df[summary_cols], use_container_width=True, height=500)
            
            # Export option
            if st.button("Export to CSV"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"soil_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        else:
            st.info("No predictions history available.")
    
    with tab3:
        st.header("Upload CSV File")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Save to watch folder
                filename = f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                filepath = os.path.join(predictor.watch_folder, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                st.success(f"File saved to {filepath}")
                
                # Process immediately
                with st.spinner("Processing uploaded file..."):
                    predictions = predictor.process_csv_file(filepath)
                    
                    if predictions:
                        # Add to predictions history
                        new_df = pd.DataFrame(predictions)
                        predictor.predictions_history = pd.concat([predictor.predictions_history, new_df], ignore_index=True)
                        
                        st.success(f"Processed {len(predictions)} rows from uploaded file")
                        
                        # Show first prediction
                        if predictions:
                            pred = predictions[0]
                            st.subheader("Sample Prediction")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Input Values:**")
                                st.write(f"N: {pred['N']} ppm")
                                st.write(f"P: {pred['P']} ppm")
                                st.write(f"K: {pred['K']} ppm")
                                st.write(f"Moisture: {pred['moisture']}%")
                                st.write(f"Temperature: {pred['temperature']}°C")
                                st.write(f"pH: {pred['pH']}")
                            
                            with col2:
                                st.write("**Predictions:**")
                                st.write(f"Soil Health: **{pred['soil_health']}**")
                                st.write(f"Confidence: **{pred['health_confidence']}%**")
                                st.write(f"Nutrient Status: **{pred['nutrient_status']}**")
                                st.write(f"Nutrient Confidence: **{pred['nutrient_confidence']}%**")
            
            except Exception as e:
                st.error(f"Error processing uploaded file: {str(e)}")
        
        # CSV format guide
        st.subheader("CSV File Format")
        st.write("""
        CSV files should have the following columns (exact names required):
        - **N**: Nitrogen level (ppm)
        - **P**: Phosphorus level (ppm)
        - **K**: Potassium level (ppm)
        - **moisture**: Moisture content (%)
        - **temperature**: Temperature (°C)
        - **pH**: pH level
        
        **Example:**
        ```
        N,P,K,moisture,temperature,pH
        55,32,42,52,26.5,6.8
        54,31,43,51,26.3,6.7
        56,33,44,53,26.8,6.9
        ```
        """)
        
        # Create sample CSV
        if st.button("Download Sample CSV"):
            sample_data = pd.DataFrame({
                'N': [55, 54, 56],
                'P': [32, 31, 33],
                'K': [42, 43, 44],
                'moisture': [52, 51, 53],
                'temperature': [26.5, 26.3, 26.8],
                'pH': [6.8, 6.7, 6.9]
            })
            csv = sample_data.to_csv(index=False)
            st.download_button(
                label="Download Sample CSV",
                data=csv,
                file_name="sample_soil_data.csv",
                mime="text/csv"
            )
    
    # Auto-refresh
    time.sleep(int(interval))
    st.rerun()

if __name__ == "__main__":
    main()
