import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import json
import random
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')
from prophet import Prophet

def main():
    
    # Custom CSS
    st.markdown("""
        <style>
            .stApp {
                max-width: 100%;
                padding: 0px;
            }
            
            .card {
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                background-color: #f9f9f9;
            }
            
            .price-up {
                color: green;
            }
            
            .price-down {
                color: red;
            }
            
            .news-card {
                background-color: #e8f5e8;
                border-left: 4px solid #4CAF50;
                padding: 15px;
                margin: 10px 0;
            }
            
            .forum-post {
                background-color: white;
                border: 1px solid #ddd;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
            }
            
            .video-container {
                position: relative;
                padding-bottom: 56.25%;
                height: 0;
                overflow: hidden;
                max-width: 100%;
                background: #000;
                margin: 10px 0;
            }
            
            .video-container iframe {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
            }
            
            .stTabs [data-baseweb="tab-list"] {
                gap: 10px;
            }
            
            .stTabs [data-baseweb="tab"] {
                height: 50px;
                padding: 0 20px;
                font-weight: 600;
            }
            
            .minimal-title {
                background: linear-gradient(90deg, 
                    #1B5E20 0%, 
                    #43A047 50%, 
                    #81C784 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-size: 3.5rem !important;
                font-weight: 900 !important;
                text-align: center !important;
                margin-bottom: 1rem !important;
                text-shadow: 0 2px 10px rgba(67, 160, 71, 0.2);
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
    """, unsafe_allow_html=True)

    # Title and Header
    st.markdown("""
        <h1 class="minimal-title">SMART FARMING HUB</h1>
        <p class="minimal-subtitle">
            <strong>AI-Powered Agriculture</strong> • Crop Price Predictions • Soil & Weather Insights • Market Trends • Farming Recommendations
        </p>
    """, unsafe_allow_html=True)

    # Create main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Market Prices & Predictions",
        "🌾 Farm Guide & Community",
        "🛒 Agri E-Commerce",
        "📰 News & Updates",
        "🌤️ Weather & Videos"
    ])

    # ============================================================================
    # TAB 1: MARKET PRICES & PREDICTIONS
    # ============================================================================
    with tab1:
        st.header("Market Trends & Price Predictions")
        
        # Simplified data loading function without external API calls
        @st.cache_data
        def load_market_data():
            """Load simulated market data"""
            crops = ["Tomato", "Potato", "Onion", "Rice", "Wheat", "Sugarcane", "Cotton", "Soybean"]
            states = ["Maharashtra", "Punjab", "Uttar Pradesh", "Karnataka", "Andhra Pradesh"]
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            base_prices = {
                "Tomato": 2500,
                "Potato": 1500,
                "Onion": 2000,
                "Rice": 3500,
                "Wheat": 2800,
                "Sugarcane": 3200,
                "Cotton": 5500,
                "Soybean": 4200
            }
            
            seasonal_patterns = {
                "Tomato": [1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.8, 0.9, 1.0, 1.2, 1.3],
                "Potato": [1.0, 1.0, 1.1, 1.2, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.9, 1.0],
                "Onion": [1.4, 1.2, 1.0, 0.8, 0.7, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.5],
                "Rice": [1.1, 1.0, 0.9, 0.9, 0.8, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.2],
                "Wheat": [1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
            }
            
            data = []
            for crop in crops:
                base_price = base_prices.get(crop, 2000)
                seasonal = seasonal_patterns.get(crop, [1.0]*12)
                
                for date in dates:
                    month_idx = date.month - 1
                    seasonal_factor = seasonal[month_idx]
                    
                    # Add some randomness
                    noise = random.uniform(-0.05, 0.05)
                    
                    price = base_price * seasonal_factor * (1 + noise)
                    price = round(price, 2)
                    
                    data.append({
                        "date": date.date(),
                        "crop": crop,
                        "price": price,
                        "state": random.choice(states),
                        "market": random.choice(["Mumbai APMC", "Azadpur Delhi", "Chennai", "Kolkata", "Hyderabad"])
                    })
            
            return pd.DataFrame(data)
        
        # Load data
        df = load_market_data()
        
        # Crop selection interface
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_crop = st.selectbox("Select Crop", sorted(df['crop'].unique()), key="crop_select")
        with col2:
            states = df[df['crop'] == selected_crop]['state'].unique()
            selected_state = st.selectbox("Select State", states, key="state_select")
        with col3:
            forecast_days = st.slider("Forecast Days", 7, 90, 30, key="forecast_days")
        
        # Filter and prepare data
        crop_df = df[(df['crop'] == selected_crop) & (df['state'] == selected_state)]
        if len(crop_df) == 0:
            crop_df = df[df['crop'] == selected_crop].iloc[:100]
        
        crop_df = crop_df.groupby("date")["price"].mean().reset_index()
        crop_df['date'] = pd.to_datetime(crop_df['date'])
        crop_df = crop_df.sort_values('date')
        crop_df.set_index("date", inplace=True)
        
        # Ensure we have enough data
        if len(crop_df) < 30:
            st.info(f"Generating additional data for analysis...")
            base_price = crop_df['price'].mean() if len(crop_df) > 0 else 2000
            dates = pd.date_range(start=crop_df.index.min(), periods=100, freq='D')
            prices = []
            for i, date in enumerate(dates):
                seasonal = 0.2 * np.sin(2 * np.pi * i/30)  # Monthly seasonality
                trend = 0.001 * i  # Slight upward trend
                noise = random.uniform(-0.05, 0.05)
                price = base_price * (1 + seasonal + trend + noise)
                prices.append(price)
            
            crop_df = pd.DataFrame({'price': prices}, index=dates)
        
        st.subheader(f"Price Trend & Prediction for {selected_crop}")
        
        # Prepare data for Prophet
        def prepare_time_series(data):
            """Prepare time series data for forecasting"""
            ts = data['price'].resample('D').mean()
            ts = ts.fillna(method='ffill').fillna(method='bfill')
            return ts
        
        try:
            ts_data = prepare_time_series(crop_df)
            prophet_df = ts_data.reset_index()
            prophet_df.columns = ['ds', 'y']
            
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            model.fit(prophet_df)
            
            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)
            
            # Create Plotly figure
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=prophet_df['ds'],
                y=prophet_df['y'],
                mode='lines',
                name='Historical Prices',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                mode='lines',
                name='Price Forecast',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_lower'],
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(width=0),
                name='Confidence Interval'
            ))
            
            fig.update_layout(
                title=f"{selected_crop} Price Prediction for Next {forecast_days} Days",
                xaxis_title="Date",
                yaxis_title="Price (₹/Quintal)",
                hovermode="x unified",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Analysis and recommendations
            recent_data = ts_data[-30:]
            forecast_values = forecast['yhat'].iloc[-forecast_days:].values
            
            current_price = recent_data.iloc[-1]
            forecast_avg = forecast_values.mean()
            price_change = ((forecast_avg - current_price) / current_price) * 100
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Current Price",
                    value=f"₹{current_price:.2f}",
                    delta=f"{price_change:.1f}%" if abs(price_change) > 2 else None,
                    delta_color="inverse"
                )
            
            with col2:
                max_idx = np.argmax(forecast_values)
                best_day_price = forecast_values[max_idx]
                best_day = forecast['ds'].iloc[-forecast_days + max_idx]
                
                st.metric(
                    label="Best Selling Day",
                    value=best_day.strftime("%d %b %Y"),
                    delta=f"₹{best_day_price:.2f}"
                )
            
            with col3:
                if price_change > 5:
                    trend = "STRONG UP"
                    color = "green"
                    action = "Consider delaying sale"
                elif price_change > 2:
                    trend = "UP"
                    color = "lightgreen"
                    action = "Good time to sell"
                elif price_change < -5:
                    trend = "STRONG DOWN"
                    color = "red"
                    action = "Sell immediately"
                elif price_change < -2:
                    trend = "DOWN"
                    color = "orange"
                    action = "Consider selling soon"
                else:
                    trend = "STABLE"
                    color = "gray"
                    action = "Monitor daily"
                
                st.markdown(f"""
                    <div style="padding: 15px; border-radius: 10px; background-color: {color}20; border-left: 4px solid {color};">
                        <strong>Market Trend:</strong> {trend}<br>
                        <small>Recommendation: {action}</small>
                    </div>
                """, unsafe_allow_html=True)
            
            # Detailed recommendations
            st.subheader("Trading Recommendations")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("** Storage Advice**")
                if price_change > 5:
                    st.write("• If you have storage facilities, consider holding")
                    st.write("• Monitor for market news")
                    st.write("• Check storage conditions regularly")
                elif price_change < -5:
                    st.write("• Sell immediately if no storage")
                    st.write("• Avoid storing for long periods")
                    st.write("• Consider local markets")
                else:
                    st.write("• Normal storage practices apply")
                    st.write("• Check quality regularly")
                    st.write("• Plan according to weather")
            
            with col2:
                st.markdown("** Market Timing**")
                if price_change > 5:
                    st.write(f"• Expected peak in {max_idx+1} days")
                    st.write("• Watch for weekly patterns")
                    st.write("• Consider partial selling")
                elif price_change < -5:
                    st.write("• Sell as soon as possible")
                    st.write("• Avoid weekend sales")
                    st.write("• Check morning prices")
                else:
                    st.write("• Plan based on cash flow")
                    st.write("• Monitor daily trends")
                    st.write("• Consider contract farming")
            
            with col3:
                st.markdown("** Risk Management**")
                if price_change > 5:
                    st.write("• Consider forward contracts")
                    st.write("• Diversify selling days")
                    st.write("• Monitor competitor prices")
                elif price_change < -5:
                    st.write("• Immediate sale recommended")
                    st.write("• Check government MSP")
                    st.write("• Consider value addition")
                else:
                    st.write("• Diversify crops")
                    st.write("• Insurance recommended")
                    st.write("• Build market relationships")
                    
        except Exception as e:
            st.error(f"Prediction model error: {str(e)}")
            st.info("Showing historical trend analysis")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=crop_df.index,
                y=crop_df['price'],
                mode='lines',
                name='Historical Prices'
            ))
            
            if len(crop_df) > 7:
                ma_7 = crop_df['price'].rolling(window=7).mean()
                fig.add_trace(go.Scatter(
                    x=crop_df.index,
                    y=ma_7,
                    mode='lines',
                    name='7-Day Average',
                    line=dict(color='orange', dash='dash')
                ))
            
            fig.update_layout(
                title=f"{selected_crop} Historical Price Trend",
                xaxis_title="Date",
                yaxis_title="Price (₹/Quintal)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    # ============================================================================
    # TAB 2: FARM GUIDE & COMMUNITY
    # ============================================================================
    with tab2:
        st.header("Farm Guide & Community Forum")
        
        st.subheader("Community Forum - Ask & Share Farming Knowledge")
        
        # Sample forum posts
        forum_posts = [
            {
                "user": "Ramesh Patel (Gujarat)",
                "date": "2 days ago",
                "question": "How to control whitefly in cotton crops organically without pesticides?",
                "answers": [
                    "Use yellow sticky traps (15 traps per acre) to catch adult whiteflies",
                    "Spray neem oil solution (5ml neem oil + 2ml soap per liter water) weekly",
                    "Intercrop with marigold or sunflower as trap crops",
                    "Release Encarsia formosa parasitic wasps for biological control"
                ],
                "upvotes": 24,
                "category": "Pest Control"
            },
            {
                "user": "Sunita Devi (Uttar Pradesh)",
                "date": "5 days ago",
                "question": "Best time to sow wheat and optimal fertilizer schedule?",
                "answers": [
                    "Optimal sowing time: Last week of October to mid-November",
                    "Recommended variety: HD-2967 for irrigated conditions",
                    "Fertilizer schedule: 120kg N, 60kg P2O5, 40kg K2O per hectare",
                    "Apply 1/3 nitrogen at sowing, 1/3 at crown root initiation, 1/3 at flowering"
                ],
                "upvotes": 18,
                "category": "Crop Management"
            },
            {
                "user": "Kumar Swamy (Karnataka)",
                "date": "1 week ago",
                "question": "Drip irrigation system cost and government subsidy available?",
                "answers": [
                    "Average cost: ₹50,000-₹70,000 per acre for complete system",
                    "Government subsidy: 50-90% under PMKSY scheme",
                    "Contact your district agriculture office for application",
                    "Required documents: Land papers, Aadhar, bank account details"
                ],
                "upvotes": 32,
                "category": "Irrigation"
            }
        ]
        
        # Display existing posts
        for post in forum_posts:
            with st.container():
                st.markdown(f"""
                <div class="forum-post">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="font-size: 1.1em;">{post['question']}</strong><br>
                            <small>Asked by {post['user']} • {post['date']} • {post['category']}</small>
                        </div>
                        <div style="background-color: #4CAF50; color: white; padding: 2px 10px; border-radius: 10px;">
                            {post['upvotes']}
                        </div>
                    </div>
                    <hr style="margin: 10px 0;">
                    <strong>Expert Answers:</strong>
                </div>
                """, unsafe_allow_html=True)
                
                for answer in post['answers']:
                    st.markdown(f"• {answer}")
                
                new_answer = st.text_area("Add your experience or answer:", 
                                         key=f"answer_{post['user']}", 
                                         placeholder="Share your farming experience or additional tips...",
                                         height=80)
                if st.button("Post Answer", key=f"btn_{post['user']}"):
                    if new_answer.strip():
                        st.success("Thank you for sharing your knowledge! Your answer will be reviewed and posted.")
                    else:
                        st.warning("Please enter your answer before posting.")
        
        # New question form
        st.markdown("### Ask a New Question")
        col1, col2 = st.columns(2)
        with col1:
            new_question = st.text_area("Your Question:", 
                                       height=100, 
                                       placeholder="Type your farming question here...",
                                       help="Be specific about crop, problem, and your location")
        with col2:
            your_name = st.text_input("Your Name (Optional):")
            your_state = st.selectbox("Your State:", 
                                    ["Select State", "Andhra Pradesh", "Assam", "Bihar", "Gujarat", "Haryana", 
                                     "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", 
                                     "Punjab", "Rajasthan", "Tamil Nadu", "Uttar Pradesh", 
                                     "West Bengal", "Other"])
            question_category = st.selectbox("Category:", 
                                           ["Select Category", "Pest Control", "Crop Management", "Soil Health", 
                                            "Irrigation", "Government Schemes", "Market Information", "Other"])
        
        if st.button("Submit Question", key="submit_question"):
            if new_question.strip() and your_state != "Select State" and question_category != "Select Category":
                st.success("Question submitted successfully! Our farming community will respond within 24 hours.")
            else:
                st.error("Please fill all required fields: Question, State, and Category")

    # ============================================================================
    # TAB 3: AGRI E-COMMERCE
    # ============================================================================
    with tab3:
        st.header("Agri E-Commerce")
        st.subheader("Purchase Farming Products")
        
        # E-commerce sites data
        ecommerce_sites = [
            {
                "name": "Amazon India",
                "base_url": "https://www.amazon.in/s?k=",
                "rating": 4.5,
                "delivery": "2-3 days",
                "trust_score": 9,
                
            },
            {
                "name": "Flipkart",
                "base_url": "https://www.flipkart.com/search?q=",
                "rating": 4.3,
                "delivery": "3-4 days",
                "trust_score": 8,

            },
            {
                "name": "BigBasket",
                "base_url": "https://www.bigbasket.com/ps/?q=",
                "rating": 4.6,
                "delivery": "1-2 days",
                "trust_score": 9,
                
            },
            {
                "name": "JioMart",
                "base_url": "https://www.jiomart.com/search/",
                "rating": 4.2,
                "delivery": "2-3 days",
                "trust_score": 8,
                
            },
            {
                "name": "AgriBegri",
                "base_url": "https://agribegri.com/search?q=",
                "rating": 4.4,
                "delivery": "3-5 days",
                "trust_score": 7,
                
            },
            {
                "name": "Ugaoo",
                "base_url": "https://www.ugaoo.com/search?q=",
                "rating": 4.7,
                "delivery": "1-3 days",
                "trust_score": 9,
                
            }
        ]
        
        # Product categories
        product_categories = {
            "Seeds": ["Tomato Seeds", "Wheat Seeds HD-2967", "Basmati Rice Seeds", "Bt Cotton Seeds", "Vegetable Seeds Kit"],
            "Fertilizers": ["Urea Fertilizer 50kg", "DAP Fertilizer", "NPK 19:19:19", "Organic Vermicompost", "Neem Cake"],
            "Pesticides": ["Neem Oil 1L", "Cypermethrin 250ml", "Mancozeb Fungicide", "Herbal Pesticide", "Monocrotophos"],
            "Equipment": ["Power Sprayer 20L", "Drip Irrigation Kit", "Soil Moisture Meter", "Crop Cutter", "Tractor Accessories"],
            "Tools": ["Garden Tool Set", "Pruning Shears", "Watering Can", "Wheelbarrow", "Gardening Gloves"]
        }
        
        # Product selection
        selected_category = st.selectbox("Select Product Category", list(product_categories.keys()), key="product_category")
        selected_product = st.selectbox("Select Product", product_categories[selected_category], key="product_select")
        
        # AI scoring algorithm
        def calculate_ai_score(site, product_name, category):
            """Calculate AI recommendation score based on multiple factors"""
            score = site['rating'] * 20
            
            if "1-2" in site['delivery']:
                score += 15
            elif "2-3" in site['delivery']:
                score += 10
            else:
                score += 5
            
            score += site['trust_score'] * 10
            
            if category == "Seeds":
                if "Agri" in site['name'] or "Ugaoo" in site['name']:
                    score += 20
            elif category == "Equipment":
                if "Amazon" in site['name'] or "Flipkart" in site['name']:
                    score += 15
            
            price_score = random.uniform(0.8, 1.2) * 20
            score += price_score
            
            return min(100, score)
        
        st.markdown(f"### AI Recommendations for: **{selected_product}**")
        
        # Calculate scores and get top recommendations
        scored_sites = []
        for site in ecommerce_sites:
            score = calculate_ai_score(site, selected_product, selected_category)
            search_query = selected_product.replace(" ", "+")
            full_url = f"{site['base_url']}{search_query}"
            
            scored_sites.append({
                **site,
                "score": round(score, 1),
                "url": full_url,
                "estimated_price": f"₹{random.randint(200, 2000)}"
            })
        
        scored_sites.sort(key=lambda x: x['score'], reverse=True)
        top_3_sites = scored_sites[:3]
        
        # Display top recommendations
        for idx, site in enumerate(top_3_sites, 1):
            col1, col2, col3, col4 = st.columns([1, 2, 2, 2])
            
            with col1:
                st.markdown(f"**#{idx}**")

            
            with col2:
                st.markdown(f"**{site['name']}**")
                st.markdown(f"**AI Score: {site['score']}/100**")
                progress = site['score'] / 100
                st.progress(progress)
            
            with col3:
                st.markdown(f"**{site['rating']}/5**")
                st.markdown(f"**{site['delivery']}**")
                
            
            with col4:
                st.markdown(f'<a href="{site["url"]}" target="_blank"><button style="background-color:#4CAF50;color:white;padding:10px 20px;border:none;border-radius:5px;cursor:pointer;">Visit Store →</button></a>', 
                           unsafe_allow_html=True)
        
        st.info("**AI Tip:** For seeds and specialized agricultural products, prefer AgriBegri or Ugaoo. For equipment and fertilizers, Amazon and Flipkart often have better deals.")

    # ============================================================================
    # TAB 4: NEWS & UPDATES
    # ============================================================================
    with tab4:
        st.header("Latest Agriculture News & Updates")
        
        # Simplified news fetching without external API calls
        @st.cache_data(ttl=3600)
        def fetch_agri_news():
            """Return curated agriculture news"""
            return [
                {
                    'title': 'Government Increases MSP for Kharif Crops 2023-24',
                    'description': 'Minimum Support Prices increased by 5-7% for major crops including paddy, pulses, and oilseeds.',
                    'date': 'Today',
                    'source': 'PIB Delhi',
                    'link': 'https://pib.gov.in'
                },
                {
                    'title': 'IMD Predicts Normal Monsoon for 2024',
                    'description': 'India Meteorological Department forecasts 102% of long-period average rainfall for June-September 2024.',
                    'date': '1 day ago',
                    'source': 'IMD Update',
                    'link': 'https://mausam.imd.gov.in'
                },
                {
                    'title': 'New Soil Health Card Scheme Launched',
                    'description': 'Government launches digital soil health cards with fertilizer recommendations for each farm plot.',
                    'date': '3 days ago',
                    'source': 'Agriculture Ministry',
                    'link': 'https://soilhealth.dac.gov.in'
                },
                {
                    'title': 'Record Wheat Production Expected',
                    'description': 'Wheat production expected to reach 112 million tonnes in 2023-24, 5% higher than last year.',
                    'date': '1 week ago',
                    'source': 'FAO Report',
                    'link': 'https://fao.org'
                },
                {
                    'title': 'Digital Agriculture Mission Launched',
                    'description': 'Government launches ₹500 crore mission to promote technology in agriculture including AI, drones, and IoT.',
                    'date': '2 weeks ago',
                    'source': 'Economic Times',
                    'link': 'https://economictimes.indiatimes.com'
                },
                {
                    'title': 'Organic Farming Exports Grow 25%',
                    'description': 'Indian organic product exports reach $1.5 billion, with major demand from US and EU markets.',
                    'date': '2 weeks ago',
                    'source': 'APEDA',
                    'link': 'https://apeda.gov.in'
                },
                {
                    'title': 'New Drought-Resistant Rice Variety',
                    'description': 'ICAR develops Sahbhagi Dhan variety that requires 30% less water and gives 20% higher yield.',
                    'date': '3 weeks ago',
                    'source': 'ICAR News',
                    'link': 'https://icar.org.in'
                },
                {
                    'title': 'Farmers Income Doubling Scheme Progress',
                    'description': '50% of target achieved under PM-KISAN scheme, ₹6000 annually transferred to 11 crore farmers.',
                    'date': '1 month ago',
                    'source': 'NITI Aayog',
                    'link': 'https://niti.gov.in'
                }
            ]
        
        # Fetch and display news
        news_items = fetch_agri_news()
        
        for news in news_items:
            st.markdown(f"""
            <div class="news-card">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <strong style="font-size: 1.1em;">{news['title']}</strong>
                    <small style="color: #666; white-space: nowrap;">{news.get('date', 'Recent')}</small>
                </div>
                <p style="margin: 10px 0; color: #333;">{news['description']}</p>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="background-color: #4CAF50; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.8em;">
                        {news['source']}
                    </span>
                    <a href="{news['link']}" target="_blank" style="color: #2196F3; text-decoration: none;">Read more →</a>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ============================================================================
    # TAB 5: WEATHER & VIDEOS
    # ============================================================================
    with tab5:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("Weather Forecast")
            st.subheader("Weather Forecast for Major Agricultural Zones")
            
            weather_data = {
                "North India (Punjab, Haryana)": {"temp": "18-28°C", "rain": "10%", "condition": "Sunny", "advice": "Good for wheat harvesting"},
                "West India (Maharashtra, Gujarat)": {"temp": "22-32°C", "rain": "30%", "condition": "Partly Cloudy", "advice": "Suitable for cotton picking"},
                "South India (Karnataka, Tamil Nadu)": {"temp": "24-34°C", "rain": "60%", "condition": "Rain likely", "advice": "Delay rice transplantation"},
                "East India (West Bengal, Odisha)": {"temp": "20-30°C", "rain": "40%", "condition": "Humid", "advice": "Good for vegetable crops"}
            }
            
            for region, info in weather_data.items():
                with st.container():
                    st.markdown(f"**{region}**")
                    col_temp, col_rain = st.columns(2)
                    with col_temp:
                        st.metric("Temperature", info["temp"])
                    with col_rain:
                        st.metric("Rain Chance", info["rain"])
                    st.caption(f"Condition: {info['condition']}")
                    st.info(info['advice'])
                    st.divider()
        
        with col2:
            st.header("Educational Videos")
            st.subheader("Farming Tutorials & Guides")
            
            farming_videos = [
                {
                    "title": "Modern Farming Techniques",
                    "video_id": "I6jBdX0l-Xk",
                    "channel": "Agriculture World",
                    "duration": "12:45",
                    "description": "Learn about modern farming methods including precision agriculture and drone technology."
                },
                {
                    "title": "Organic Farming Complete Guide",
                    "video_id": "WhOrIUlrnPo",
                    "channel": "Organic Farming Channel",
                    "duration": "18:30",
                    "description": "Step-by-step guide to start organic farming with natural pest control methods."
                },
                {
                    "title": "Drip Irrigation Installation",
                    "video_id": "0aoS_7d7zGU",
                    "channel": "FarmTech Solutions",
                    "duration": "15:20",
                    "description": "Practical demonstration of drip irrigation system installation for small farms."
                },
                {
                    "title": "Soil Health Management",
                    "video_id": "vROQ+Z3N19Y",
                    "channel": "Soil Science India",
                    "duration": "22:10",
                    "description": "Understanding soil nutrients, pH balance, and organic matter improvement."
                }
            ]
            
            for video in farming_videos:
                with st.expander(f"{video['title']} ({video['duration']})"):
                    st.markdown(f"**Channel:** {video['channel']}")
                    st.markdown(f"""
                    <div class="video-container">
                        <iframe src="https://www.youtube.com/embed/{video['video_id']}" 
                                frameborder="0" 
                                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                                allowfullscreen>
                        </iframe>
                    </div>
                    """, unsafe_allow_html=True)
                    st.write(video['description'])

    # ============================================================================
    # FOOTER
    # ============================================================================
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px; background-color: #f0f0f0; border-radius: 10px; margin-top: 20px;">
        <p><strong>AI Agriculture Platform</strong> • Empowering Farmers with Technology</p>
        <p>Data Sources: AGMARKNET • India Meteorological Department • Ministry of Agriculture</p>
        <p>Contact Support: support@aiagriplatform.gov.in • Toll Free: 1800-XXX-XXXX</p>
        <p style="font-size: 0.9em; margin-top: 10px;">Disclaimer: Market predictions are based on historical data and AI models. Actual prices may vary.</p>
    </div>
    """, unsafe_allow_html=True)
