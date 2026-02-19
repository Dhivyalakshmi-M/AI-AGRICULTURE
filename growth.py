"""
SMART FARM ML - OPTIMIZED FOR FAST LOADING
Crop Growth Prediction, and Irrigation Advisor
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ========== LIGHTWEIGHT ML IMPORTS ==========
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import cv2
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import datetime
import tempfile
from datetime import timedelta

# ========== PAGE CONFIGURATION ==========
# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stage-card {
        background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid;
    }
    .current-stage {
        border-left-color: #FF9800;
        background: linear-gradient(135deg, #FFF3E0, #FFE0B2);
    }
</style>
""", unsafe_allow_html=True)

# ========== CROP GROWTH PREDICTOR ==========
class CropGrowthPredictor:
    """Predict crop growth stages and harvest dates"""
    
    def __init__(self):
        # Crop lifecycle data (days)
        self.crop_data = {
            'Tomato': {
                'stages': ['Germination', 'Seedling', 'Vegetative', 'Flowering', 'Fruiting', 'Ripening', 'Harvest'],
                'durations': [7, 14, 30, 20, 30, 15, 0],
                'total_days': 116
            },
            'Potato': {
                'stages': ['Planting', 'Sprouting', 'Vegetative', 'Tuber Initiation', 'Tuber Bulking', 'Maturation', 'Harvest'],
                'durations': [0, 15, 25, 20, 40, 25, 0],
                'total_days': 125
            },
            'Rice': {
                'stages': ['Germination', 'Seedling', 'Tillering', 'Stem Elongation', 'Booting', 'Heading', 'Flowering', 'Milk', 'Dough', 'Maturity', 'Harvest'],
                'durations': [5, 25, 30, 25, 10, 5, 7, 10, 10, 15, 0],
                'total_days': 142
            },
            'Wheat': {
                'stages': ['Germination', 'Seedling', 'Tillering', 'Stem Elongation', 'Booting', 'Heading', 'Flowering', 'Milk', 'Dough', 'Ripening', 'Harvest'],
                'durations': [7, 20, 35, 25, 10, 5, 7, 12, 15, 10, 0],
                'total_days': 146
            }
        }
        
        # Growth stage indicators (simplified)
        self.stage_indicators = {
            'Germination': 'Small shoot emerging from seed',
            'Seedling': 'First true leaves appearing',
            'Vegetative': 'Rapid leaf growth, no flowers',
            'Flowering': 'Flower buds and blossoms visible',
            'Fruiting': 'Small fruits forming',
            'Ripening': 'Fruits changing color',
            'Harvest': 'Ready for picking'
        }
    
    def predict_growth_stage(self, image_path, crop_type='Tomato', days_planted=30):
        """Predict current growth stage based on days planted"""
        if crop_type not in self.crop_data:
            crop_type = 'Tomato'
        
        data = self.crop_data[crop_type]
        durations = data['durations']
        stages = data['stages']
        
        # Calculate current stage based on days
        cumulative = 0
        current_stage = 0
        for i, duration in enumerate(durations):
            cumulative += duration
            if days_planted <= cumulative or i == len(durations) - 1:
                current_stage = i
                break
        
        # Calculate progress
        days_in_stage = days_planted - (cumulative - durations[current_stage])
        stage_progress = (days_in_stage / durations[current_stage]) * 100 if durations[current_stage] > 0 else 100
        
        # Days to harvest
        remaining_days = data['total_days'] - days_planted
        harvest_date = datetime.datetime.now() + timedelta(days=remaining_days)
        
        return {
            'crop': crop_type,
            'current_stage': stages[current_stage],
            'stage_index': current_stage,
            'stage_description': self.stage_indicators.get(stages[current_stage], 'Growing stage'),
            'days_planted': days_planted,
            'days_in_stage': days_in_stage,
            'stage_progress': min(100, stage_progress),
            'remaining_days': max(0, remaining_days),
            'harvest_date': harvest_date.strftime('%B %d, %Y'),
            'total_stages': len(stages),
            'all_stages': stages
        }
    
    def get_lifecycle_chart(self, crop_type='Tomato', current_stage_idx=2):
        """Generate lifecycle visualization"""
        data = self.crop_data.get(crop_type, self.crop_data['Tomato'])
        
        fig = go.Figure()
        
        for i, (stage, duration) in enumerate(zip(data['stages'], data['durations'])):
            if i < current_stage_idx:
                color = '#4CAF50'  # Completed
            elif i == current_stage_idx:
                color = '#FF9800'  # Current
            else:
                color = '#E0E0E0'  # Future
            
            fig.add_trace(go.Bar(
                x=[duration],
                y=[stage],
                orientation='h',
                marker_color=color,
                name=stage,
                text=[f"{duration} days"],
                textposition='auto'
            ))
        
        fig.update_layout(
            title=f'{crop_type} Growth Lifecycle',
            xaxis_title='Duration (days)',
            yaxis_title='Growth Stage',
            showlegend=False,
            height=400
        )
        
        return fig

# ========== IRRIGATION ADVISOR ==========
class IrrigationAdvisor:
    """Simple irrigation recommendations based on weather"""
    
    def __init__(self):
        pass
    
    def get_weekly_forecast(self):
        """Generate 7-day weather forecast"""
        dates = [datetime.datetime.now() + timedelta(days=i) for i in range(7)]
        forecast = []
        
        for i, date in enumerate(dates):
            # Simulated weather data
            forecast.append({
                'date': date.strftime('%Y-%m-%d'),
                'day': date.strftime('%A'),
                'temperature': round(25 + np.random.uniform(-5, 5), 1),
                'humidity': np.random.randint(50, 85),
                'rainfall': round(np.random.uniform(0, 10), 1),
                'rain_chance': np.random.randint(0, 100)
            })
        
        return forecast
    
    def recommend_irrigation(self, soil_moisture, crop_stage, forecast):
        """Generate irrigation recommendations"""
        recommendations = []
        
        for day in forecast:
            rain_sufficient = day['rainfall'] > 5
            temp_high = day['temperature'] > 30
            humidity_low = day['humidity'] < 60
            
            if rain_sufficient:
                action = "Skip irrigation"
                reason = f"Expected rainfall: {day['rainfall']}mm"
                water_needed = 0
            elif temp_high and humidity_low:
                action = "Irrigate in morning"
                reason = "High temperature & low humidity"
                water_needed = 20
            elif day['rain_chance'] > 70:
                action = "Delay irrigation"
                reason = f"High rain chance: {day['rain_chance']}%"
                water_needed = 0
            else:
                action = "Normal irrigation"
                reason = "Standard conditions"
                water_needed = 15
            
            recommendations.append({
                'day': day['day'],
                'date': day['date'],
                'temperature': day['temperature'],
                'rainfall': day['rainfall'],
                'action': action,
                'reason': reason,
                'water_needed': water_needed
            })
        
        return recommendations

# ========== MAIN APPLICATION ==========
def main():
    # Title
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
        
        <h1 class="minimal-title">SMART FARM ML SYSTEM</h1>
        <p class="minimal-subtitle">
            <strong>AI-Powered</strong> Crop Growth Predictor & Irrigation Advisor
        </p>
    """, unsafe_allow_html=True)
    
    growth_predictor = CropGrowthPredictor()
    irrigation_advisor = IrrigationAdvisor()
    
   
    # Create tabs
    tab1, tab2 = st.tabs(["Growth Prediction", "Irrigation Advisor"])
    

    # TAB 1: GROWTH PREDICTION
    with tab1:
        st.markdown("### Crop Growth Stage Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            crop_type = st.selectbox(
                "Select Crop Type",
                ['Tomato', 'Potato', 'Rice', 'Wheat']
            )
            
            days_planted = st.slider(
                "Days since planting",
                min_value=0,
                max_value=200,
                value=45,
                help="How many days ago was the crop planted?"
            )
            
            if st.button("📈 Analyze Growth", type="primary", use_container_width=True):
                # Get growth prediction
                growth_info = growth_predictor.predict_growth_stage(
                    "",  # No image needed for simple prediction
                    crop_type,
                    days_planted
                )
                
                with col2:
                    st.markdown(f"### Current Stage: **{growth_info['current_stage']}**")
                    
                    # Progress bar
                    progress_bar = st.progress(growth_info['stage_progress'] / 100)
                    st.caption(f"Stage progress: {growth_info['stage_progress']:.1f}%")
                    
                    # Metrics
                    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                    with col_metrics1:
                        st.metric("Days Planted", growth_info['days_planted'])
                    with col_metrics2:
                        st.metric("Days to Harvest", growth_info['remaining_days'])
                    with col_metrics3:
                        st.metric("Harvest Date", growth_info['harvest_date'])
                    
                    st.markdown("####Stage Description")
                    st.info(growth_info['stage_description'])
                    
                    # Store for chart display
                    st.session_state['growth_info'] = growth_info
        
        # Display lifecycle chart if growth info exists
        if 'growth_info' in st.session_state:
            growth_info = st.session_state['growth_info']
            st.markdown("#### 📊 Crop Lifecycle Timeline")
            fig = growth_predictor.get_lifecycle_chart(crop_type, growth_info['stage_index'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Stage details
            st.markdown("#### Growth Stage Details")
            cols = st.columns(3)
            for i, stage in enumerate(growth_info['all_stages']):
                col_idx = i % 3
                with cols[col_idx]:
                    if i < growth_info['stage_index']:
                        status = "Completed"
                        color = "green"
                    elif i == growth_info['stage_index']:
                        status = "Current"
                        color = "orange"
                    else:
                        status = "Upcoming"
                        color = "gray"
                    
                    st.markdown(f"""
                    <div style="padding: 10px; border-left: 3px solid {color}; background: #f9f9f9; margin: 5px 0; border-radius: 5px;">
                        <strong>{stage}</strong><br>
                        <small>{status}</small>
                    </div>
                    """, unsafe_allow_html=True)
    
    # TAB 2: IRRIGATION ADVISOR
    with tab2:
        st.markdown("### Smart Irrigation Advisor")
        
        # Get weather forecast
        forecast = irrigation_advisor.get_weekly_forecast()
        
        # Display forecast
        st.markdown("#### 7-Day Weather Forecast")
        cols = st.columns(7)
        for i, day in enumerate(forecast):
            with cols[i]:
                temp_color = "#FF6B6B" if day['temperature'] > 30 else "#4ECDC4"
                bg_color = "#FFF5F5" if day['temperature'] > 30 else "#F0F8FF"
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; background: {bg_color}; border-radius: 8px; border: 1px solid #ddd;">
                    <strong>{day['day'][:3]}</strong><br>
                    <span style="color:{temp_color}; font-size: 20px;">{day['temperature']}°C</span><br>
                    💧 {day['rainfall']}mm<br>
                    📊 {day['rain_chance']}%
                </div>
                """, unsafe_allow_html=True)
        
        # User inputs
        col1, col2 = st.columns(2)
        with col1:
            soil_moisture = st.slider("Soil Moisture Level", 0.1, 1.0, 0.6, 0.1, 
                                      help="0.1 = Very dry, 1.0 = Saturated")
            crop_type = st.selectbox("Crop Type", ['Tomato', 'Potato', 'Rice', 'Wheat'], key="irrigation_crop")
        with col2:
            crop_stage = st.selectbox("Current Growth Stage", 
                                     ['Germination', 'Vegetative', 'Flowering', 'Fruiting', 'Maturation'])
        
        if st.button("💧 Get Irrigation Plan", type="primary", use_container_width=True):
            recommendations = irrigation_advisor.recommend_irrigation(
                soil_moisture, crop_stage, forecast
            )
            
            st.markdown("####Daily Irrigation Recommendations")
            
            total_water = 0
            irrigation_days = 0
            
            for rec in recommendations:
                total_water += rec['water_needed']
                if rec['water_needed'] > 0:
                    irrigation_days += 1
                
                # Card styling based on action
                if rec['action'] == "Skip irrigation":
                    icon = "⏸️"
                    border_color = "#4CAF50"  # Green
                elif rec['action'] == "Irrigate in morning":
                    icon = "🌅"
                    border_color = "#2196F3"  # Blue
                elif rec['action'] == "Delay irrigation":
                    icon = "⏳"
                    border_color = "#FF9800"  # Orange
                else:
                    icon = "💧"
                    border_color = "#9C27B0"  # Purple
                
                st.markdown(f"""
                <div style="padding: 15px; margin: 10px 0; border-left: 4px solid {border_color}; 
                            background: linear-gradient(135deg, #FFFFFF, #F5F5F5); border-radius: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>{icon} {rec['day']} ({rec['date']})</strong><br>
                            <span style="font-size: 14px;">🌡️ {rec['temperature']}°C | 💧 {rec['rainfall']}mm rain</span><br>
                            <span style="color: #666;">{rec['reason']}</span>
                        </div>
                        <div style="text-align: right;">
                            <h3 style="margin: 0; color: #2196F3;">{rec['water_needed']} mm</h3>
                            <strong>{rec['action']}</strong>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Summary
            st.markdown("####Weekly Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Water Needed", f"{total_water} mm")
            with col2:
                st.metric("Irrigation Days", irrigation_days)
            with col3:
                water_saved = (7 - irrigation_days) * 15  # Estimate
                st.metric("Water Saved", f"{water_saved} mm")
    

if __name__ == "__main__":
    main()
