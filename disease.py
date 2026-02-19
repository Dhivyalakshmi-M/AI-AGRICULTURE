  
import os
import cv2
import numpy as np
import streamlit as st
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def main():
    # Custom CSS for beautiful UI
    st.markdown("""
        <style>
        .main {
            padding: 0rem 1rem;
        }
        
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        .card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
            border-left: 5px solid #4CAF50;
        }
        
        .highlight-box {
            background-color: #f8fff9;
            padding: 1rem;
            border-radius: 10px;
            border: 2px solid #4CAF50;
            margin: 1rem 0;
        }
        
        .warning-box {
            background-color: #fff8f8;
            padding: 1rem;
            border-radius: 10px;
            border: 2px solid #ff6b6b;
            margin: 1rem 0;
        }
        
        .info-box {
            background-color: #f0f8ff;
            padding: 1rem;
            border-radius: 10px;
            border: 2px solid #2196F3;
            margin: 1rem 0;
        }
        
        .metric-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            margin: 0.5rem;
        }
        
        .stButton>button {
            background: linear-gradient(90deg, #4CAF50 0%, #2E7D32 100%);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 25px;
            font-weight: bold;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(76, 175, 80, 0.3);
        }
        
        .uploadedImage {
            border-radius: 15px;
            border: 3px solid #4CAF50;
            padding: 5px;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .healthy { background-color: #4CAF50; }
        .moderate { background-color: #FF9800; }
        .critical { background-color: #f44336; }
        
        .footer {
            text-align: center;
            margin-top: 3rem;
            padding: 1rem;
            color: #666;
            font-size: 0.9rem;
        }
        </style>
    """, unsafe_allow_html=True)

    DATASET_PATH = r'C:\Users\varma\Desktop\agri\dataset'
    MODEL_PATH = "model.pkl"
    IMG_SIZE = 128
    CONF_THRESHOLD = 60  # percent

    classes = ["healthy", "leaf_spot", "leaf_blight"]

    disease_info = {
        0: ("Healthy Leaf", "No treatment needed", "Balanced NPK fertilizer", "#4CAF50", "Low"),
        1: ("Leaf Spot Disease", "Spray Mancozeb / Copper fungicide", "Potassium rich fertilizer", "#FF9800", "Medium"),
        2: ("Leaf Blight Disease", "Spray Carbendazim", "Organic compost + controlled Nitrogen", "#f44336", "High")
    }

    def extract_hog(image):
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return hog(gray, orientations=9, pixels_per_cell=(8,8),
                   cells_per_block=(2,2), block_norm="L2-Hys")

    def estimate_growth_stage(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        leaf_area = cv2.countNonZero(thresh)

        if leaf_area < 2000:
            return "Seedling", 25
        elif leaf_area < 5000:
            return "Vegetative", 50
        else:
            return "Flowering", 85

    def estimate_yield(stage, disease):
        base_yield = {
            "Seedling": 30,
            "Vegetative": 70,
            "Flowering": 100
        }
        reduction = 0 if disease == "Healthy Leaf" else (40 if disease == "Leaf Blight Disease" else 25)
        return max(base_yield[stage] - reduction, 20)

    def calculate_health_score(confidence, disease_idx):
        base_score = confidence
        if disease_idx == 0:  # Healthy
            return min(100, base_score + 20)
        elif disease_idx == 1:  # Leaf Spot
            return max(30, base_score - 20)
        else:  # Leaf Blight
            return max(20, base_score - 30)

    def load_sample_images():
        return {
            "Healthy": "https://images.unsplash.com/photo-1518837695005-2083093ee35b?w=400&h=300&fit=crop",
            "Leaf Spot": "https://images.unsplash.com/photo-1593482892290-5d188b9e56dc?w-400&h=300&fit=crop",
            "Leaf Blight": "https://images.unsplash.com/photo-1597059194406-bbde4e025d9b?w=400&h=300&fit=crop"
        }

    # -------- LOAD OR TRAIN MODEL --------
    @st.cache_resource
    def load_model():
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        else:
            X, y = [], []
            for label, cls in enumerate(classes):
                folder = os.path.join(DATASET_PATH, cls)
                if os.path.exists(folder):
                    for img_name in os.listdir(folder)[:100]:  # Limit for demo
                        img = cv2.imread(os.path.join(folder, img_name))
                        if img is not None:
                            X.append(extract_hog(img))
                            y.append(label)

            if len(X) > 0:
                X = np.array(X)
                y = np.array(y)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                model = SVC(kernel="linear", probability=True)
                model.fit(X_train, y_train)
                joblib.dump(model, MODEL_PATH)
            else:
                # Create a dummy model for demo
                model = SVC(kernel="linear", probability=True)
                # Fit with dummy data
                dummy_X = np.random.rand(10, 3249)
                dummy_y = np.random.randint(0, 3, 10)
                model.fit(dummy_X, dummy_y)
        return model

    model = load_model()

    # -------- HEADER SECTION --------
    # Custom CSS with enhanced title styling
    st.markdown("""
        <style>
        .main {
            padding: 0rem 1rem;
        }
        
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        /* Enhanced Title Container with Glow Effect */
        .title-container {
            background: linear-gradient(135deg, 
                rgba(26, 107, 45, 0.95) 0%, 
                rgba(76, 175, 80, 0.95) 50%, 
                rgba(139, 195, 74, 0.95) 100%);
            padding: 3rem 2rem;
            border-radius: 20px;
            margin-bottom: 2.5rem;
            box-shadow: 
                0 10px 30px rgba(26, 107, 45, 0.3),
                0 4px 6px rgba(0,0,0,0.1),
                inset 0 1px 0 rgba(255,255,255,0.2);
            color: white;
            text-align: center;
            position: relative;
            overflow: hidden;
            border: 2px solid rgba(255, 255, 255, 0.1);
        }
        
        .title-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 5px;
            background: linear-gradient(90deg, 
                #FFD700 0%, 
                #FFA500 50%, 
                #FFD700 100%);
            z-index: 2;
        }
        
        .title-container::after {
            content: '🌱';
            position: absolute;
            top: -25px;
            right: -25px;
            font-size: 150px;
            opacity: 0.1;
            transform: rotate(15deg);
        }
        
        /* Main Title with Highlight Effect */
        .main-title {
            font-size: 3.2rem !important;
            font-weight: 800 !important;
            margin-bottom: 1rem !important;
            background: linear-gradient(to right, 
                #FFFFFF 0%, 
                #F8FFE8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: 
                0 2px 4px rgba(0, 0, 0, 0.2),
                0 0 30px rgba(255, 255, 255, 0.3);
            position: relative;
            display: inline-block;
            letter-spacing: 0.5px;
        }
        
        .main-title::after {
            content: '';
            position: absolute;
            bottom: -10px;
            left: 25%;
            width: 50%;
            height: 4px;
            background: linear-gradient(90deg, 
                transparent 0%, 
                #FFD700 50%, 
                transparent 100%);
            border-radius: 2px;
        }
        
        /* Subtitle with Icon */
        .subtitle {
            font-size: 1.4rem !important;
            font-weight: 400 !important;
            color: rgba(255, 255, 255, 0.95) !important;
            margin-top: 1.5rem !important;
            padding: 0 2rem;
            position: relative;
            display: inline-flex;
            align-items: center;
            gap: 10px;
        }
        
        .subtitle::before {
            content: '🤖';
            font-size: 1.6rem;
            filter: drop-shadow(0 2px 3px rgba(0,0,0,0.2));
        }
        
        .subtitle::after {
            content: '🌿';
            font-size: 1.6rem;
            filter: drop-shadow(0 2px 3px rgba(0,0,0,0.2));
        }
        
        /* Tagline */
        .tagline {
            font-size: 1.1rem;
            color: rgba(255, 255, 255, 0.8);
            margin-top: 1.5rem;
            font-style: italic;
            padding: 0 3rem;
            line-height: 1.6;
        }
        
        /* Stats Row in Title */
        .title-stats {
            display: flex;
            justify-content: center;
            gap: 3rem;
            margin-top: 2rem;
            padding-top: 1.5rem;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .stat-item {
            text-align: center;
        }
        
        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            color: #FFD700;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            display: block;
        }
        
        .stat-label {
            font-size: 0.9rem;
            color: rgba(255, 255, 255, 0.9);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 0.3rem;
        }
        
        /* Floating Leaves Animation */
        @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(10deg); }
        }
        
        .leaf-emoji {
            position: absolute;
            font-size: 2.5rem;
            animation: float 6s ease-in-out infinite;
            opacity: 0.7;
            z-index: 1;
        }
        
        .leaf1 {
            top: 20px;
            left: 30px;
            animation-delay: 0s;
        }
        
        .leaf2 {
            bottom: 30px;
            right: 40px;
            animation-delay: 2s;
        }
        
        .leaf3 {
            top: 60px;
            right: 80px;
            animation-delay: 4s;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .main-title {
                font-size: 2.5rem !important;
            }
            .subtitle {
                font-size: 1.2rem !important;
                padding: 0 1rem;
            }
            .title-stats {
                gap: 1.5rem;
                flex-wrap: wrap;
            }
            .stat-value {
                font-size: 1.8rem;
            }
        }
        
        /* Rest of your existing CSS... */
        .card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            margin-bottom: 1.5rem;
            border-left: 5px solid #4CAF50;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.12);
        }
        
        .highlight-box {
            background: linear-gradient(135deg, #f8fff9 0%, #e8f5e9 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border: 2px solid #4CAF50;
            margin: 1rem 0;
            box-shadow: 0 3px 10px rgba(76, 175, 80, 0.1);
        }
        
        .stButton>button {
            background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
            color: white;
            border: none;
            padding: 0.85rem 2.5rem;
            border-radius: 30px;
            font-weight: 600;
            font-size: 1.1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        }
        
        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(76, 175, 80, 0.4);
        }
        
        .uploadedImage {
            border-radius: 15px;
            border: 3px solid #4CAF50;
            padding: 5px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    # -------- ENHANCED TITLE SECTION --------
    # Minimal highlighted title
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
        
        <h1 class="minimal-title">Plant Disease Detection System</h1>
        <p class="minimal-subtitle">
            <strong>AI-Powered</strong> Crop Health Analysis & Treatment Recommendations
        </p>
    """, unsafe_allow_html=True)

    # -------- MAIN LAYOUT --------
    col1, col2 = st.columns([2, 1])

    with col1:
        
        st.subheader("Upload Leaf Image")
        
        # Upload section
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"], 
                                        label_visibility="collapsed")
        
        if uploaded_file:
            img_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            
            # Display uploaded image
            st.image(image, channels="BGR", use_container_width=True, caption="Uploaded Leaf Image")
            
            # Process image
            features = extract_hog(image).reshape(1, -1)
            probs = model.predict_proba(features)[0]
            confidence = np.max(probs) * 100
            prediction = np.argmax(probs)
            
            disease, remedy, fertilizer, color, severity = disease_info[prediction]
            stage, stage_progress = estimate_growth_stage(image)
            yield_est = estimate_yield(stage, disease)
            health_score = calculate_health_score(confidence, prediction)
            
            # Confidence indicator
            st.markdown("###  Analysis Results")
            
            if confidence < CONF_THRESHOLD:
                st.markdown(f'''
                <div class="warning-box">
                    <h4>Low Confidence Prediction ({confidence:.1f}%)</h4>
                    <p>Please upload a clearer image of the leaf for more accurate diagnosis.</p>
                </div>
                ''', unsafe_allow_html=True)
            else:
                # Metrics in columns
                col_metric1, col_metric2, col_metric3 = st.columns(3)
                
                with col_metric1:
                    st.markdown(f'''
                    <div class="metric-box">
                        <h3>{confidence:.1f}%</h3>
                        <p>Confidence</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col_metric2:
                    st.markdown(f'''
                    <div class="metric-box" style="background: linear-gradient(135deg, {color}99 0%, {color} 100%);">
                        <h3>{disease}</h3>
                        <p>Diagnosis</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col_metric3:
                    st.markdown(f'''
                    <div class="metric-box">
                        <h3>{health_score:.0f}/100</h3>
                        <p>Health Score</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Disease severity indicator
                st.markdown(f"### 📊 Disease Severity: **{severity}**")
                severity_map = {"Low": 25, "Medium": 60, "High": 90}
                st.progress(severity_map[severity] / 100)
                
                # Detailed Information
                col_info1, col_info2 = st.columns(2)
                
                with col_info1:
                    st.markdown(f'''
                    <div class="highlight-box">
                        <h4>Treatment Recommendation</h4>
                        <p><strong>Primary Treatment:</strong> {remedy}</p>
                        <p><strong>Fertilizer:</strong> {fertilizer}</p>
                        <p><strong>Application:</strong> Apply every 7-10 days until symptoms disappear</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col_info2:
                    st.markdown(f'''
                    <div class="info-box">
                        <h4>Plant Growth Info</h4>
                        <p><strong>Growth Stage:</strong> {stage}</p>
                        <p><strong>Estimated Yield:</strong> {yield_est}% of potential</p>
                        <p><strong>Leaf Area:</strong> {cv2.countNonZero(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))} pixels</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Visualization
                st.markdown("###Health Analysis")
                
                # Create health chart
                fig = go.Figure(data=[
                    go.Bar(name='Metrics', 
                          x=['Health Score', 'Yield Potential', 'Disease Risk'], 
                          y=[health_score, yield_est, 100-health_score],
                          marker_color=[color, '#4CAF50', '#f44336'])
                ])
                fig.update_layout(title="Plant Health Metrics", yaxis_range=[0,100])
                st.plotly_chart(fig, use_container_width=True)
                
                # Probability distribution
                st.markdown("### Disease Probability Distribution")
                prob_data = pd.DataFrame({
                    'Disease': ['Healthy', 'Leaf Spot', 'Leaf Blight'],
                    'Probability': probs * 100
                })
                fig2 = px.pie(prob_data, values='Probability', names='Disease', 
                             color_discrete_sequence=['#4CAF50', '#FF9800', '#f44336'])
                st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Quick Tips Card
        st.subheader("Quick Tips")
        tips = [
            "Upload clear, well-lit images of leaves for best results",
            "Capture both sides of the leaf if possible",
            "Include a scale reference when possible",
            "Regular monitoring prevents disease spread",
            "Maintain proper spacing between plants"
        ]
        for tip in tips:
            st.markdown(f"• {tip}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Disease Examples
        st.subheader("Common Diseases")
        
        sample_images = load_sample_images()
        
        for i, (disease_name, (full_name, remedy, fertilizer, color, severity)) in enumerate(disease_info.items()):
            with st.expander(f"{full_name} - {severity} Risk"):
                st.markdown(f"**Symptoms:** {disease_name}")
                st.markdown(f"**Treatment:** {remedy}")
                st.markdown(f"**Prevention:** {fertilizer}")
                st.markdown(f'<span class="status-indicator {severity.lower()}"></span> Risk Level: {severity}', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


    # -------- FOOTER --------
    st.markdown("---")
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown("**Plant Disease Detection System v2.0** • Using Machine Learning for Agricultural Health Monitoring")
    st.markdown("*For accurate results, ensure good lighting and clear focus when capturing leaf images*")
    st.markdown("</div>", unsafe_allow_html=True)
