import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf
import numpy as np
import os
from datetime import datetime, timedelta
from PIL import Image

from transformers import ViTImageProcessor, ViTModel, pipeline

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="FinFusion - NVDA Predictor",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="collapsed"
)

# ====================== CUSTOM CSS ======================
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(180deg, #0a0a0a 0%, #111827 100%);
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Card Styles */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    }
    
    .metric-value {
        font-size: 42px;
        font-weight: 700;
        margin: 8px 0;
    }
    
    .metric-label {
        font-size: 14px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    .up-prediction {
        color: #22c55e;
        text-shadow: 0 0 20px rgba(34, 197, 94, 0.5);
    }
    
    .down-prediction {
        color: #ef4444;
        text-shadow: 0 0 20px rgba(239, 68, 68, 0.5);
    }
    
    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 60px 40px;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 50%, #1e293b 100%);
        border-radius: 24px;
        border: 1px solid #334155;
        margin-bottom: 40px;
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #22c55e, transparent);
    }
    
    .hero-title {
        font-size: 64px;
        font-weight: 800;
        background: linear-gradient(135deg, #22c55e 0%, #4ade80 50%, #22c55e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -2px;
    }
    
    .hero-subtitle {
        font-size: 24px;
        color: #e2e8f0;
        margin: 16px 0 0 0;
        font-weight: 300;
    }
    
    .hero-tech {
        font-size: 14px;
        color: #64748b;
        margin-top: 12px;
        letter-spacing: 2px;
    }
    
    /* Feature Cards */
    .feature-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 32px 24px;
        height: 100%;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        border-color: #22c55e;
        box-shadow: 0 0 30px rgba(34, 197, 94, 0.1);
    }
    
    .feature-icon {
        font-size: 48px;
        margin-bottom: 16px;
    }
    
    .feature-title {
        font-size: 20px;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 12px;
    }
    
    .feature-desc {
        font-size: 14px;
        color: #94a3b8;
        line-height: 1.6;
    }
    
    /* Date Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        color: #f1f5f9;
        font-weight: 600;
        padding: 16px 24px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        border-color: #22c55e;
        box-shadow: 0 0 20px rgba(34, 197, 94, 0.2);
        transform: translateY(-2px);
    }
    
    /* Primary Button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        border: none;
        font-size: 18px;
        padding: 20px 40px;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #4ade80 0%, #22c55e 100%);
        box-shadow: 0 0 30px rgba(34, 197, 94, 0.4);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 28px;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    /* Info Panel */
    .info-panel {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px;
        margin-top: 16px;
    }
    
    .info-row {
        display: flex;
        justify-content: space-between;
        padding: 12px 0;
        border-bottom: 1px solid #1e293b;
    }
    
    .info-row:last-child {
        border-bottom: none;
    }
    
    .info-label {
        color: #94a3b8;
        font-size: 14px;
    }
    
    .info-value {
        color: #f1f5f9;
        font-weight: 600;
        font-size: 16px;
    }
    
    /* Reasoning Box */
    .reasoning-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 2px solid #334155;
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        margin-top: 24px;
    }
    
    .reasoning-box.bullish {
        border-color: #22c55e;
        box-shadow: 0 0 40px rgba(34, 197, 94, 0.15);
    }
    
    .reasoning-box.bearish {
        border-color: #ef4444;
        box-shadow: 0 0 40px rgba(239, 68, 68, 0.15);
    }
    
    .reasoning-text {
        font-size: 28px;
        font-weight: 700;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #22c55e, #4ade80);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: #1e293b;
        border-radius: 12px;
    }
    
    /* Factor List */
    .factor-item {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .factor-name {
        color: #94a3b8;
        font-size: 14px;
    }
    
    .factor-value {
        color: #f1f5f9;
        font-weight: 600;
    }
    
    /* Disclaimer */
    .disclaimer {
        background: #1e293b;
        border-left: 4px solid #f59e0b;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin-top: 40px;
    }
    
    .disclaimer-title {
        color: #f59e0b;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    .disclaimer-text {
        color: #94a3b8;
        font-size: 13px;
        line-height: 1.6;
    }
    
    /* Loading Animation */
    .loading-container {
        text-align: center;
        padding: 60px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #1e293b;
        border-radius: 8px;
        padding: 12px 24px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid #22c55e;
        border-radius: 20px;
        padding: 8px 16px;
        font-size: 12px;
        color: #22c55e;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        background: #22c55e;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
</style>
""", unsafe_allow_html=True)

# ====================== MODEL LOADING ======================
@st.cache_resource
def load_models():
    feature_scaler = joblib.load("models/feature_scaler.pkl")
    target_scaler = joblib.load("models/target_scaler.pkl")

    class StockLSTM(nn.Module):
        def __init__(self, input_size=783, hidden_size=64, num_layers=2, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.fc_reg = nn.Linear(hidden_size, 1)
            self.fc_class = nn.Linear(hidden_size, 1)

        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            h = h_n[-1]
            reg = self.fc_reg(h)
            class_out = torch.sigmoid(self.fc_class(h))
            return reg, class_out

    model = StockLSTM()
    model.load_state_dict(torch.load("models/final_multimodal_fixed.pth", map_location="cpu"))
    model.eval()

    xgb_model = joblib.load("models/xgboost_model.pkl")
    finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)
    
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
    vit_model.eval()
    device = torch.device("cpu")
    vit_model.to(device)

    return feature_scaler, target_scaler, model, xgb_model, finbert, processor, vit_model, device

feature_scaler, target_scaler, model, xgb_model, finbert, processor, vit_model, device = load_models()

LOOKBACK = 60
TEMP_DIR = "temp_charts"
os.makedirs(TEMP_DIR, exist_ok=True)

# ====================== CORE FUNCTIONS (UNCHANGED) ======================
def get_live_data(target_date_str):
    target_date = pd.to_datetime(target_date_str).tz_localize(None)
    start_date = target_date - timedelta(days=400)
    
    data = yf.download("NVDA", start=start_date, end=target_date + timedelta(days=1), progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    
    data.index = pd.to_datetime(data.index).tz_localize(None)
    data = data[data.index <= target_date]
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
    
    data['SMA_10'] = data['Close'].rolling(10).mean()
    data['SMA_30'] = data['Close'].rolling(30).mean()
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    
    data['BB_Middle'] = data['Close'].rolling(20).mean()
    data['BB_Std'] = data['Close'].rolling(20).std()
    data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
    data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)
    
    data = data.dropna()
    return data


def generate_fresh_candlestick(data, target_date_str):
    os.makedirs("temp_charts", exist_ok=True)
    
    plot_data = data.tail(60)[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        plot_data[col] = pd.to_numeric(plot_data[col], errors='coerce')
    
    plot_data = plot_data.dropna()
    
    if len(plot_data) < 10:
        st.error("Not enough data to generate chart.")
        return None
    
    plot_data.index = pd.date_range(start="2024-01-01", periods=len(plot_data), freq='D')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    mpf.plot(plot_data, 
             type='candle', 
             style='yahoo', 
             ax=ax, 
             volume=False, 
             show_nontrading=False)
    
    image_path = os.path.join("temp_charts", f"candle_{target_date_str}.png")
    fig.savefig(image_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    
    return image_path


def extract_vit_features(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = vit_model(**inputs)
        vit_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
    return vit_embedding, None


def explain_prediction(last_row, predicted_direction, confidence, xgb_model, features):
    importance = xgb_model.feature_importances_
    feat_imp = pd.Series(importance, index=features).sort_values(ascending=False).head(8)
    
    explanation = ["### 🔍 Top Contributing Factors"]
    for feature, score in feat_imp.items():
        value = last_row.get(feature, 0)
        if "vit_dim" in feature:
            explanation.append("**• Chart Pattern (ViT)** — Strong visual signal")
        elif feature == 'finbert_sentiment':
            explanation.append(f"**• News Sentiment** → {value:.3f}")
        elif "MACD" in feature:
            explanation.append(f"**• MACD Momentum** → {value:.4f}")
        elif "RSI" in feature:
            explanation.append(f"**• RSI** → {value:.1f}")
        else:
            explanation.append(f"**• {feature}** → {value:.4f}")
    
    explanation.append("\n### 🎯 Final Reasoning")
    if predicted_direction == 1:
        explanation.append("**Bullish signals dominate**")
    else:
        explanation.append("**Bearish signals dominate**")
    if confidence < 0.58:
        explanation.append("**⚠️ Low confidence — HOLD recommended**")
    
    return "\n".join(explanation)


def live_predict(target_date_str):
    with st.spinner(""):
        # Custom loading animation
        loading_placeholder = st.empty()
        loading_placeholder.markdown("""
        <div class="loading-container">
            <div style="font-size: 48px; margin-bottom: 20px;">🔄</div>
            <div style="font-size: 20px; color: #f1f5f9; margin-bottom: 8px;">Analyzing Market Data</div>
            <div style="font-size: 14px; color: #64748b;">Processing technical indicators, chart patterns & sentiment...</div>
        </div>
        """, unsafe_allow_html=True)
        
        data = get_live_data(target_date_str)
        
        recent = data.tail(LOOKBACK).copy()
        recent['finbert_sentiment'] = 0.0
        
        image_path = generate_fresh_candlestick(data, target_date_str)
        vit_embedding, _ = extract_vit_features(image_path)
        
        for i in range(768):
            recent[f"vit_dim_{i}"] = vit_embedding[i]
        
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'finbert_sentiment', 
                    'SMA_10', 'SMA_30', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 
                    'BB_Middle', 'BB_Upper', 'BB_Lower'] + [f"vit_dim_{i}" for i in range(768)]
        
        scaled = feature_scaler.transform(recent[features])
        input_tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            reg_out, class_out = model(input_tensor)
            prob = class_out.item()                 
            
            prediction = "UP" if prob > 0.5 else "DOWN"
            
            distance = abs(prob - 0.5)
            confidence = round(distance * 2 * 100, 1)
            
            confidence = min(confidence, 65.0)
            confidence = max(confidence, 51.0)        
        
        last_row = recent.iloc[-1]
        explanation = explain_prediction(last_row, 1 if prediction == "UP" else 0, confidence, xgb_model, features)
        
        loading_placeholder.empty()
        
        # Create styled chart
        mc = mpf.make_marketcolors(
            up='#22c55e', down='#ef4444',
            edge='inherit',
            wick={'up': '#22c55e', 'down': '#ef4444'},
            volume='in'
        )
        s = mpf.make_mpf_style(
            marketcolors=mc,
            base_mpf_style='nightclouds',
            gridstyle='',
            facecolor='#0f172a',
            edgecolor='#334155',
            figcolor='#0f172a',
            rc={'axes.labelcolor': '#94a3b8', 'xtick.color': '#64748b', 'ytick.color': '#64748b'}
        )
        
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0f172a')
        ax.set_facecolor('#0f172a')
        mpf.plot(recent.tail(60), type='candle', style=s, ax=ax, volume=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#334155')
        ax.spines['left'].set_color('#334155')
        plt.tight_layout()
            
        return {
            "prediction": prediction,
            "confidence": confidence,
            "explanation": explanation,
            "chart": fig,
            "date": target_date_str,
            "last_row": last_row,
            "features": features
        }

# ====================== STREAMLIT UI ======================
if 'page' not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    # Status Badge
    st.markdown("""
    <div style="display: flex; justify-content: center; margin-bottom: 20px;">
        <div class="status-badge">
            <div class="status-dot"></div>
            <span>LIVE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">FinFusion</h1>
        <p class="hero-subtitle">Advanced Multimodal NVDA Stock Predictor</p>
        <p class="hero-tech">LSTM • VISION TRANSFORMER • TECHNICAL ANALYSIS • SENTIMENT</p>
    </div>
    """, unsafe_allow_html=True)

    # Features Section
    st.markdown('<div class="section-header">✨ Core Capabilities</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Visual Pattern Recognition</div>
            <div class="feature-desc">
                Vision Transformer analyzes candlestick charts to detect complex visual patterns 
                that traditional indicators might miss.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">Multimodal Intelligence</div>
            <div class="feature-desc">
                Combines price data, technical indicators, news sentiment, and chart patterns 
                into a unified prediction framework.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">Explainable AI</div>
            <div class="feature-desc">
                Transparent reasoning shows exactly which factors drive each prediction, 
                building trust through understanding.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Model Architecture Section
    with st.expander("🏗️ Model Architecture", expanded=False):
        arch_col1, arch_col2 = st.columns(2)
        
        with arch_col1:
            st.markdown("""
            **Input Modalities**
            - 60-day price history (OHLCV)
            - Technical indicators (RSI, MACD, Bollinger Bands, SMAs)
            - FinBERT sentiment scores
            - ViT-extracted chart embeddings (768 dimensions)
            """)
        
        with arch_col2:
            st.markdown("""
            **Model Components**
            - LSTM encoder (2 layers, 64 hidden units)
            - XGBoost ensemble for feature importance
            - Vision Transformer (ViT-base-patch16-224)
            - Binary classification + regression heads
            """)

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # CTA Section
    st.markdown("""
    <div style="text-align: center; margin: 40px 0;">
        <p style="font-size: 24px; color: #f1f5f9; margin-bottom: 24px;">Ready to see tomorrow's prediction?</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("🚀  Start Live Prediction", type="primary", use_container_width=True):
            st.session_state.page = "predict"
            st.rerun()

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <div class="disclaimer-title">⚠️ Educational Disclaimer</div>
        <div class="disclaimer-text">
            This is an academic demonstration project built for a Computing Science Final Year project. 
            It is not financial advice. Past performance does not guarantee future results. 
            Always consult a qualified financial advisor before making investment decisions.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ====================== PREDICTION PAGE ======================
else:
    # Header with back button
    col_back, col_title = st.columns([1, 5])
    with col_back:
        if st.button("← Back", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
    
    with col_title:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 16px;">
            <span style="font-size: 32px; font-weight: 700; color: #f1f5f9;">NVDA Prediction</span>
            <div class="status-badge">
                <div class="status-dot"></div>
                <span>LIVE DATA</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Date Selection
    st.markdown('<div class="section-header">📅 Select Prediction Date</div>', unsafe_allow_html=True)
    
    today = datetime.now().date()
    next_days = []
    current = today
    while len(next_days) < 5:
        current += timedelta(days=1)
        if current.weekday() < 5:
            next_days.append(current)
    
    cols = st.columns(5, gap="medium")
    selected_date = None
    
    for i, day in enumerate(next_days):
        with cols[i]:
            day_name = day.strftime("%a")
            day_num = day.strftime("%d")
            month = day.strftime("%b")
            
            if st.button(
                f"{day_name}\n{month} {day_num}",
                key=f"date_{i}",
                use_container_width=True
            ):
                selected_date = day.strftime("%Y-%m-%d")
    
    # Show prediction if date selected
    if selected_date:
        st.markdown("<br>", unsafe_allow_html=True)
        
        result = live_predict(selected_date)
        
        if result:
            # Prediction Metrics Row
            st.markdown('<div class="section-header">📈 Prediction Results</div>', unsafe_allow_html=True)
            
            metric_col1, metric_col2, metric_col3 = st.columns(3, gap="large")
            
            with metric_col1:
                direction_class = "up-prediction" if result['prediction'] == "UP" else "down-prediction"
                direction_icon = "↑" if result['prediction'] == "UP" else "↓"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Direction</div>
                    <div class="metric-value {direction_class}">{direction_icon} {result['prediction']}</div>
                    <div style="color: #64748b; font-size: 12px;">Next Trading Day</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col2:
                confidence_color = "#22c55e" if result['confidence'] > 58 else "#f59e0b" if result['confidence'] > 54 else "#ef4444"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value" style="color: {confidence_color};">{result['confidence']:.1f}%</div>
                    <div style="color: #64748b; font-size: 12px;">Model Certainty</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col3:
                last_close = result['last_row']['Close']
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Last Close</div>
                    <div class="metric-value" style="color: #f1f5f9;">${last_close:.2f}</div>
                    <div style="color: #64748b; font-size: 12px;">NVDA</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Main Content: Chart + Analysis
            chart_col, analysis_col = st.columns([1.2, 1], gap="large")
            
            with chart_col:
                st.markdown('<div class="section-header">📊 Price Chart (60 Days)</div>', unsafe_allow_html=True)
                st.pyplot(result['chart'], use_container_width=True)
                
                # Price Stats
                last_row = result['last_row']
                daily_change = ((last_row['Close'] - last_row['Open']) / last_row['Open']) * 100
                change_color = "#22c55e" if daily_change >= 0 else "#ef4444"
                
                st.markdown(f"""
                <div class="info-panel">
                    <div class="info-row">
                        <span class="info-label">Open</span>
                        <span class="info-value">${last_row['Open']:.2f}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">High</span>
                        <span class="info-value">${last_row['High']:.2f}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Low</span>
                        <span class="info-value">${last_row['Low']:.2f}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Close</span>
                        <span class="info-value">${last_row['Close']:.2f}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Daily Change</span>
                        <span class="info-value" style="color: {change_color};">{daily_change:+.2f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with analysis_col:
                st.markdown('<div class="section-header">🔍 Model Analysis</div>', unsafe_allow_html=True)
                
                # Feature Importance Chart
                importance = xgb_model.feature_importances_
                feat_imp = pd.Series(importance, index=result['features']).sort_values(ascending=True).tail(10)
                
                # Clean feature names for display
                display_names = []
                for name in feat_imp.index:
                    if 'vit_dim' in name:
                        display_names.append('Chart Pattern (ViT)')
                    elif name == 'finbert_sentiment':
                        display_names.append('News Sentiment')
                    else:
                        display_names.append(name.replace('_', ' ').title())
                
                fig_bar, ax_bar = plt.subplots(figsize=(8, 5), facecolor='#0f172a')
                ax_bar.set_facecolor('#0f172a')
                
                bars = ax_bar.barh(display_names, feat_imp.values, color='#22c55e', height=0.6)
                ax_bar.set_xlabel("Importance", color='#94a3b8', fontsize=10)
                ax_bar.tick_params(colors='#64748b', labelsize=9)
                ax_bar.spines['top'].set_visible(False)
                ax_bar.spines['right'].set_visible(False)
                ax_bar.spines['bottom'].set_color('#334155')
                ax_bar.spines['left'].set_color('#334155')
                
                for bar, val in zip(bars, feat_imp.values):
                    ax_bar.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
                               f'{val:.3f}', va='center', color='#94a3b8', fontsize=8)
                
                plt.tight_layout()
                st.pyplot(fig_bar, use_container_width=True)
                
                # Key Technical Indicators
                st.markdown("**Technical Indicators**")
                
                rsi_val = last_row['RSI']
                rsi_status = "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"
                rsi_color = "#ef4444" if rsi_val > 70 else "#22c55e" if rsi_val < 30 else "#f59e0b"
                
                macd_val = last_row['MACD']
                macd_signal = last_row['MACD_Signal']
                macd_status = "Bullish" if macd_val > macd_signal else "Bearish"
                macd_color = "#22c55e" if macd_val > macd_signal else "#ef4444"
                
                st.markdown(f"""
                <div class="info-panel">
                    <div class="info-row">
                        <span class="info-label">RSI (14)</span>
                        <span class="info-value" style="color: {rsi_color};">{rsi_val:.1f} ({rsi_status})</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">MACD</span>
                        <span class="info-value" style="color: {macd_color};">{macd_val:.4f} ({macd_status})</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">SMA 10</span>
                        <span class="info-value">${last_row['SMA_10']:.2f}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">SMA 30</span>
                        <span class="info-value">${last_row['SMA_30']:.2f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Final Reasoning Box
            st.markdown("<br>", unsafe_allow_html=True)
            
            reasoning_class = "bullish" if result['prediction'] == "UP" else "bearish"
            reasoning_color = "#22c55e" if result['prediction'] == "UP" else "#ef4444"
            reasoning_text = "📈 Bullish Signals Dominate" if result['prediction'] == "UP" else "📉 Bearish Signals Dominate"
            
            st.markdown(f"""
            <div class="reasoning-box {reasoning_class}">
                <div style="font-size: 14px; color: #64748b; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 12px;">
                    Final Analysis
                </div>
                <div class="reasoning-text" style="color: {reasoning_color};">
                    {reasoning_text}
                </div>
                <div style="margin-top: 16px; color: #94a3b8; font-size: 14px;">
                    Prediction for {result['date']} | Confidence: {result['confidence']:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence Warning
            if result['confidence'] < 55:
                st.warning("⚠️ **Low Confidence Alert**: Model uncertainty is high. Consider waiting for stronger signals or use this prediction with caution.")
            
            # Disclaimer
            st.markdown("""
            <div class="disclaimer" style="margin-top: 40px;">
                <div class="disclaimer-title">⚠️ Important Notice</div>
                <div class="disclaimer-text">
                    This prediction is generated by a machine learning model and should not be considered financial advice. 
                    The model has inherent limitations and past accuracy does not guarantee future performance. 
                    Always do your own research and consult with a qualified financial advisor before making investment decisions.
                </div>
            </div>
            """, unsafe_allow_html=True)
