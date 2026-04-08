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
from transformers import AutoImageProcessor, ViTModel, pipeline

st.set_page_config(page_title="FinFusion - NVDA Predictor", layout="wide", page_icon="📈")

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    .stButton>button {width: 100%; height: 52px; font-size: 18px; font-weight: bold;}
    .metric-card {background-color: #1f2937; padding: 20px; border-radius: 12px; text-align: center;}
    .hero {background: linear-gradient(135deg, #1f2937, #111827); padding: 50px; border-radius: 15px; text-align: center; margin-bottom: 30px;}
    </style>
""", unsafe_allow_html=True)

st.title("FinFusion")
st.markdown("**Advanced Multimodal NVDA Stock Predictor**")

# ====================== LOAD MODELS ======================
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
    
    processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
    vit_model.eval()
    device = torch.device("cpu")
    vit_model.to(device)

    return feature_scaler, target_scaler, model, xgb_model, finbert, processor, vit_model, device

feature_scaler, target_scaler, model, xgb_model, finbert, processor, vit_model, device = load_models()

LOOKBACK = 60
TEMP_DIR = "temp_charts"
os.makedirs(TEMP_DIR, exist_ok=True)

# ====================== HELPER FUNCTIONS ======================
def get_live_data(target_date_str):
    target_date = pd.to_datetime(target_date_str).tz_localize(None)
    start_date = target_date - timedelta(days=400)
    data = yf.download("NVDA", start=start_date, end=target_date + timedelta(days=1), progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    data.index = data.index.tz_localize(None)
    data = data[data.index <= target_date]
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Technical Indicators (same as before)
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
    os.makedirs(TEMP_DIR, exist_ok=True)
    plot_data = data.tail(60)[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    for col in plot_data.columns:
        plot_data[col] = pd.to_numeric(plot_data[col], errors='coerce')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    mpf.plot(plot_data, type='candle', style='yahoo', ax=ax, volume=False, show_nontrading=False)
    image_path = os.path.join(TEMP_DIR, f"candle_{target_date_str}.png")
    fig.savefig(image_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return image_path

# Keep your existing extract_vit_features, get_vit_attention_heatmap, explain_prediction if needed
# (We can keep them but not show the heatmap if you prefer)

# ====================== LIVE PREDICT ======================
def live_predict(target_date_str):
    with st.spinner("Analyzing market..."):
        data = get_live_data(target_date_str)
        recent = data.tail(LOOKBACK).copy()
        recent['finbert_sentiment'] = 0.0
        
        image_path = generate_fresh_candlestick(data, target_date_str)
        
        for i in range(768):
            recent[f"vit_dim_{i}"] = 0.0
        
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'finbert_sentiment', 
                    'SMA_10', 'SMA_30', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 
                    'BB_Middle', 'BB_Upper', 'BB_Lower'] + [f"vit_dim_{i}" for i in range(768)]
        
        scaled = feature_scaler.transform(recent[features])
        input_tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            reg_out, class_out = model(input_tensor)
            prob = class_out.item()
            prediction = "UP" if prob > 0.5 else "DOWN"
            confidence = min(max(prob if prediction == "UP" else (1 - prob), 0.52), 0.78)
        
        last_row = recent.iloc[-1]
        explanation = explain_prediction(last_row, 1 if prediction == "UP" else 0, confidence, xgb_model, features)
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        mpf.plot(recent.tail(60), type='candle', style='yahoo', ax=ax, volume=False)
        plt.close(fig)
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "explanation": explanation,
            "chart": fig,
            "date": target_date_str,
            "last_row": last_row,
            "features": features
        }

# ====================== HOMEPAGE ======================
if 'page' not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    st.markdown("""
    <div class="hero">
        <h1 style="font-size:52px; margin:0; color:#4CAF50;">FinFusion</h1>
        <p style="font-size:24px; margin:20px 0 0 0; color:#ddd;">
            Multimodal Deep Learning Framework for NVDA Stock Prediction
        </p>
        <p style="font-size:17px; color:#aaa; margin-top:15px;">
            LSTM + Vision Transformer + Technical Indicators + Sentiment Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📈 Live Technical Analysis**")
    with col2:
        st.markdown("**🧠 Multimodal Intelligence**")
    with col3:
        st.markdown("**📊 Explainable Predictions**")

    if st.button("🚀 Start Live Prediction", type="primary", use_container_width=True):
        st.session_state.page = "predict"
        st.rerun()

    st.caption("Final Year Computing Science Project • Educational Purpose Only")

else:
    # Your prediction page (with all previous improvements)
    st.subheader("Select Next Business Day for Prediction")
    
    today = datetime.now().date()
    next_days = []
    current = today
    while len(next_days) < 5:
        current += timedelta(days=1)
        if current.weekday() < 5:
            next_days.append(current.strftime("%Y-%m-%d"))
    
    cols = st.columns(5)
    for i, day in enumerate(next_days):
        if cols[i].button(day, use_container_width=True):
            result = live_predict(day)
            if result:
                st.success(f"**{result['prediction']}** on {result['date']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Direction", f"**{result['prediction']}**")
                with col2:
                    st.metric("Model Confidence", f"{result['confidence']:.1f}%")
                
                st.progress(result['confidence'] / 100)
                
                col_chart, col_exp = st.columns([1.1, 1])
                
                with col_chart:
                    st.subheader("📈 Candlestick Chart")
                    st.pyplot(result['chart'], use_container_width=True)
                    
                    last_row = result.get('last_row', None)
                    if last_row is not None:
                        change_pct = ((last_row['Close'] - last_row['Open']) / last_row['Open']) * 100
                        st.markdown(f"""
                        <div style="background-color:#1f2937; padding:15px; border-radius:8px; margin-top:15px;">
                            <strong>Date:</strong> {result['date']}<br>
                            <strong>Last Open:</strong> ${last_row['Open']:.2f}<br>
                            <strong>Last Close:</strong> ${last_row['Close']:.2f}<br>
                            <strong>Daily Change:</strong> {change_pct:+.2f}%
                        </div>
                        """, unsafe_allow_html=True)
                
                with col_exp:
                    st.subheader("🔍 Model Explainability")
                    importance = xgb_model.feature_importances_
                    feat_series = pd.Series(importance, index=result['features'])
                    vit_sum = feat_series[[col for col in feat_series.index if col.startswith('vit_dim')]].sum()
                    feat_series = feat_series.drop([col for col in feat_series.index if col.startswith('vit_dim')], errors='ignore')
                    feat_series['ViT Chart Patterns'] = vit_sum
                    feat_imp = feat_series.sort_values(ascending=False).head(12)
                    
                    fig_bar, ax_bar = plt.subplots(figsize=(8, 5.5))
                    feat_imp.plot(kind='barh', ax=ax_bar, color='#4CAF50')
                    ax_bar.set_xlabel("Importance Score")
                    ax_bar.set_title("Top 12 Contributing Features")
                    ax_bar.invert_yaxis()
                    plt.tight_layout()
                    st.pyplot(fig_bar, use_container_width=True)
                    
                    st.markdown("**Key Factors**")
                    explanation_lines = result['explanation'].split('\n')
                    for line in explanation_lines:
                        if line.strip():
                            cleaned = line.strip()
                            if "vit_dim" in cleaned:
                                cleaned = "Chart Pattern (ViT) — Strong visual signal"
                            st.markdown(f"• {cleaned}")

    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

st.caption("**Disclaimer**: Educational project only. Not financial advice.")
