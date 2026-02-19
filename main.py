import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import base64

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="AgriAI Platform",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===================== CUSTOM CSS =====================
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    div[data-testid="stToolbar"] {display:none;}
    section[data-testid="stSidebar"] {display:none;}
    
    /* Main app styling */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Login container */
    .login-container {
        background: white;
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        width: 100%;
        max-width: 450px;
        margin: 0 auto;
        text-align: center;
    }
    
    .login-title {
        color: #2E7D32;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 10px;
    }
    
    .login-subtitle {
        color: #666;
        margin-bottom: 30px;
        font-size: 16px;
    }
    
    .input-group {
        margin-bottom: 20px;
        text-align: left;
    }
    
    .input-label {
        color: #555;
        font-weight: 500;
        margin-bottom: 8px;
        display: block;
    }
    
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 12px 16px;
        font-size: 15px;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #4CAF50;
        box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.1);
        outline: none;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        color: white;
        border: none;
        padding: 14px;
        border-radius: 10px;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        margin-top: 10px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(76, 175, 80, 0.3);
    }
    
    /* Dashboard styling */
    .dashboard-header {
        background: linear-gradient(135deg, #2E7D32 0%, #388E3C 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin-bottom: 30px;
        text-align: center;
    }
    
    .card-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 25px;
        margin-top: 30px;
    }
    
    .feature-card {
        background: white;
        border-radius: 15px;
        padding: 0;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        border: 1px solid #e0e0e0;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    
    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.15);
        border-color: #4CAF50;
    }
    
    .card-image {
        width: 100%;
        height: 200px;
        object-fit: cover;
    }
    
    .card-content {
        padding: 20px;
        flex-grow: 1;
        display: flex;
        flex-direction: column;
    }
    
    .card-title {
        color: #2E7D32;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 10px;
    }
    
    .card-description {
        color: #666;
        font-size: 14px;
        line-height: 1.5;
        margin-bottom: 20px;
        flex-grow: 1;
    }
    
    /* Back button styling */
    .back-button {
        background: #f8f9fa;
        border: 2px solid #4CAF50;
        color: #2E7D32;
        padding: 10px 25px;
        border-radius: 8px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        display: inline-block;
        text-align: center;
        text-decoration: none;
    }
    
    .back-button:hover {
        background: #4CAF50;
        color: white;
        transform: translateY(-2px);
    }
    
    /* Bottom buttons container */
    .bottom-buttons {
        position: fixed;
        bottom: 20px;
        left: 0;
        right: 0;
        display: flex;
        justify-content: center;
        gap: 20px;
        padding: 0 20px;
        z-index: 100;
    }
    
    .bottom-button {
        background: white;
        border: 2px solid #4CAF50;
        color: #2E7D32;
        padding: 10px 25px;
        border-radius: 8px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        min-width: 150px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .bottom-button:hover {
        background: #4CAF50;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(76, 175, 80, 0.3);
    }
    
    /* Logout button specific */
    .logout-button {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        border: none;
        color: white;
    }
    
    .logout-button:hover {
        background: linear-gradient(135deg, #d32f2f 0%, #b71c1c 100%);
    }
    
    /* Error message */
    .error-message {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Ensure content doesn't hide behind bottom buttons */
    .main-content {
        margin-bottom: 100px;
    }
</style>
""", unsafe_allow_html=True)

# ===================== SESSION STATE =====================
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'page' not in st.session_state:
    st.session_state.page = 'home'

def home_page():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), 
                    url('https://images.unsplash.com/photo-1500382017468-9049fed747ef?ixlib=rb-4.0.3&auto=format&fit=crop&w=2000&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        min-height: 100vh;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align:center; margin-top:100px; color:white;">
        <h1>🌾 AgriAI Platform</h1>
        <h3>Empowering Farmers with AI Technology</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("Get Started", use_container_width=True):
            st.session_state.page = "login"
            st.rerun()

# ===================== LOGIN FUNCTION =====================
def login(username, password):
    # Simple hardcoded credentials (in real app, use database)
    valid_credentials = {
        "admin": "admin123"
    }
    
    if username in valid_credentials and valid_credentials[username] == password:
        return True
    return False

# ===================== LOGIN PAGE =====================
def show_login_page():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), 
                    url('https://images.unsplash.com/photo-1550745165-9bc0b252726f?ixlib=rb-4.0.3&auto=format&fit=crop&w=2000&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        min-height: 100vh;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="login-container">
            <h1 class="login-title">AI Agri Platform</h1>
            <p class="login-subtitle">Sign in to access AI-powered agricultural tools</p>
        """, unsafe_allow_html=True)
        
        # Login form
        with st.form(key='login_form'):
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown('<label class="input-label" style="color: white; font-weight: bold;">Username</label>', unsafe_allow_html=True)
            username = st.text_input("", placeholder="Enter your username", label_visibility="collapsed")
            
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown('<label class="input-label" style="color: white; font-weight: bold;">Password</label>', unsafe_allow_html=True)

            password = st.text_input("", placeholder="Enter your password", 
                                    type="password", label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
            
            submit_button = st.form_submit_button(label="LOGIN", use_container_width=True)
            
            if submit_button:
                if username and password:
                    if login(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.page = 'dashboard'
                        st.rerun()
                    else:
                        st.markdown("""
                        <div class="error-message">
                            Invalid username or password. Try: admin/admin123
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="error-message">
                        Please enter both username and password
                    </div>
                    """, unsafe_allow_html=True)
        

    
    # Back button
    if st.button("← Back to Home", key="login_back_home"):
        st.session_state['page'] = "home"
        st.rerun()  # reloads the app and shows homepage


# ===================== DASHBOARD PAGE =====================
def show_dashboard():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0.2)), 
                    url('https://images.unsplash.com/photo-1620200423727-8127f75d7f53?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8N3x8YWdyaWN1bHR1cmV8ZW58MHx8MHx8fDA%3D');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        min-height: 100vh;
    }
    </style>
    """, unsafe_allow_html=True)

   
    # Header
    st.markdown(f"""
    <div class="dashboard-header">
        <h1>AI-Powered Agricultural Assistant Platform</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Grid (2 rows x 3 columns)
    features = [
        {
            "title": "AI Chat Assistant",
            "description": "Get instant answers to agricultural questions in English & Tamil",
            "image": "https://images.unsplash.com/photo-1581094794329-c8112a89af12?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
            "page": "chat"
        },
        {
            "title": "Crop Prediction",
            "description": "AI-powered crop recommendation based on soil & weather conditions",
            "image": "https://images.unsplash.com/photo-1591117610713-0632efed0f93?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTR8fGNyb3AlMjBwcmVkaWN0aW9ufGVufDB8fDB8fHww",
            "page": "crop"
        },
        {
            "title": "Disease Detection",
            "description": "Identify plant diseases from images using AI",
            "image": "https://images.unsplash.com/photo-1692481061201-5767c19396af?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NHx8cGxhbnQlMjBkaXNlYXNlJTIwZGV0ZWN0aW9ufGVufDB8fDB8fHww",
            "page": "disease"
        },
        {
            "title": "Growth Monitoring",
            "description": "Track crop growth stages and irrigation scheduling",
            "image": "https://images.unsplash.com/photo-1574943320219-553eb213f72d?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
            "page": "growth"
        },
        {
            "title": "Market Analysis",
            "description": "Price trends, best time to sell, and market insights",
            "image": "https://plus.unsplash.com/premium_photo-1683980578016-a1f980719ec2?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NXx8bWFya2V0JTIwYW5hbHlzaXN8ZW58MHx8MHx8fDA%3D",
            "page": "market"
        },
        {
            "title": "Soil Fertigation",
            "description": "Smart soil monitoring and automated fertilization",
            "image": "https://images.unsplash.com/photo-1583088516070-4668f15a4c4c?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTZ8fHNvaWwlMjBmZXJ0aWdhdGlvbnxlbnwwfHwwfHx8MA%3D%3D",
            "page": "soil"
        }
    ]
    
    # Create 2x3 grid
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    
    for i in range(0, len(features), 3):
        cols = st.columns(3)
        
        for j in range(3):
            if i + j < len(features):
                feature = features[i + j]
                with cols[j]:
                    # Card with image and content
                    st.markdown(f"""
                    <div class="feature-card">
                        <img src="{feature['image']}" class="card-image" alt="{feature['title']}">
                        <div class="card-content">
                            <h3 class="card-title">{feature['title']}</h3>
                            <p class="card-description">{feature['description']}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Button for navigation
                    if st.button(f"Open {feature['title']}", key=f"btn_{feature['page']}"):
                        st.session_state.page = feature['page']
                        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    

# ===================== FEATURE PAGES =====================
def run_feature_page(page_name):
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    # Import and run the specific feature page
    try:
        if page_name == "chat":
            import chat
            chat.main()
        elif page_name == "crop":
            import croppred
            croppred.main()
        elif page_name == "disease":
            import disease
            disease.main()
        elif page_name == "growth":
            import growth
            growth.main()
        elif page_name == "market":
            import market
            market.main()
        elif page_name == "soil":
            st.markdown("""
            <style>
            .soil-option-card {
                background: white;
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 5px 15px rgba(0,0,0,0.08);
                transition: all 0.3s ease;
                border: 1px solid #e0e0e0;
                height: 100%;
                display: flex;
                flex-direction: column;
            }

            .soil-option-card:hover {
                transform: translateY(-8px);
                box-shadow: 0 12px 25px rgba(0,0,0,0.15);
                border-color: #4CAF50;
            }

            .soil-image {
                width: 100%;
                height: 220px;
                object-fit: cover;
            }

            .soil-content {
                padding: 20px;
                text-align: center;
            }

            .soil-title {
                color: #2E7D32;
                font-size: 20px;
                font-weight: 600;
                margin-bottom: 8px;
            }

            .soil-desc {
                color: #666;
                font-size: 14px;
                margin-bottom: 15px;
            }
            </style>

            """, unsafe_allow_html=True)
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
                
            <h1 class="minimal-title">SMART SOIL FERTIGATION SYSTEM</h1>
            <p class="minimal-subtitle">
                <strong>Select a module below</strong> 
            </p>
        """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                <div class="soil-option-card">
                    <img src="https://images.unsplash.com/photo-1510844355160-2fb07bf9af75?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTJ8fHNvaWx8ZW58MHx8MHx8fDA%3D"
                         class="soil-image">
                    <div class="soil-content">
                        <div class="soil-title">Soil Manual Analysis</div>
                        <div class="soil-desc">
                            Real-time soil type identification, ML analysis, and CSV monitoring.
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if st.button("Open Soil Analysis", use_container_width=True, key="soil_analysis_btn"):
                    st.session_state.page = "soil_analysis"
                    st.rerun()

            with col2:
                st.markdown("""
                <div class="soil-option-card">
                    <img src="https://images.unsplash.com/photo-1583088516070-4668f15a4c4c?auto=format&fit=crop&w=800&q=80"
                         class="soil-image">
                    <div class="soil-content">
                        <div class="soil-title">Soil AI Analysis</div>
                        <div class="soil-desc">
                            AI-based soil health prediction and automated CSV monitoring.
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if st.button("Open Soil AI", use_container_width=True, key="soil_ai_btn"):
                    st.session_state.page = "soil_ai"
                    st.rerun()

        elif page_name == "soil_analysis":
            import soil
            soil.main()

        elif page_name == "soil_ai":
            import soil_ai
            soil_ai.main()

    except Exception as e:
        st.error(f"Error loading {page_name}: {str(e)}")
        st.info("Make sure the feature files are in the same directory.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Bottom back button only for feature pages
    st.markdown('<div class="bottom-buttons">', unsafe_allow_html=True)

    if st.button("← Back to Dashboard", key="back_dashboard"):
        st.session_state.page = "dashboard"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ===================== MAIN APP =====================
def main():
    # Check query parameters for navigation
    query_params = st.query_params
    
    if "back_to_dashboard" in query_params:
        st.session_state.page = "dashboard"
        st.rerun()
    elif "back_to_login" in query_params:
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.session_state.page = "login"
        st.rerun()
    elif "logout" in query_params:
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.session_state.page = "home"
        st.rerun()
    
    # Default first page should be HOME if not set
    if 'page' not in st.session_state:
        st.session_state.page = "home"

    # -------- PAGE ROUTING --------
    if st.session_state.page == "home":
        home_page()
        return

    if st.session_state.page == "login":
        # If already authenticated, skip login
        if st.session_state.authenticated:
            st.session_state.page = "dashboard"
            st.rerun()
        else:
            show_login_page()
        return

    if st.session_state.page == "dashboard":
        # If not authenticated, force back to login
        if not st.session_state.authenticated:
            st.session_state.page = "login"
            st.rerun()
        else:
            show_dashboard()
        return

    # -------- FEATURE PAGES --------
    # Anything else is treated as a feature page
    run_feature_page(st.session_state.page)

if __name__ == "__main__":
    main()
