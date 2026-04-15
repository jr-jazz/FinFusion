# FinFusion: Multimodal Deep Learning Framework for NVDA Stock Prediction

## Project Overview

FinFusion is a multimodal deep learning system designed to predict the next-day directional movement (UP/DOWN) of NVIDIA (NVDA) stock. The project integrates multiple data modalities including historical price data, technical indicators, financial news sentiment, and visual chart patterns extracted using a Vision Transformer.

This system was developed as part of a final-year Computing Science project, demonstrating the application of advanced machine learning techniques to financial time-series forecasting.

## Features

- Next-day directional stock price prediction for NVIDIA (NVDA)
- Multimodal architecture combining tabular, textual, and image-based features
- Technical indicator engineering (SMA, RSI, MACD, Bollinger Bands)
- Vision Transformer (ViT) for candlestick chart pattern recognition
- FinBERT for financial sentiment analysis
- Explainability using XGBoost feature importance
- Interactive web interface built with Streamlit
- Live data fetching using yfinance

## Architecture

The system employs a late fusion approach where features from different modalities are extracted independently and then combined:

- Tabular Features: OHLCV prices and technical indicators
- Text Features: Financial news sentiment using FinBERT
- Image Features: 768-dimensional embeddings extracted from candlestick charts using ViT-base-patch16-224
- Model: LSTM with dual heads (regression and binary classification)
- Explainability: XGBoost trained on the same feature set to provide feature importance

## Technologies Used

- Deep Learning: PyTorch, LSTM
- Computer Vision: Vision Transformer (ViT)
- NLP: FinBERT (Hugging Face Transformers)
- Data Processing: pandas, NumPy, yfinance
- Visualization: mplfinance, Matplotlib
- Explainability: XGBoost
- Deployment: Streamlit
- Version Control: Git

## Repository Structure
```
FinFusion/
├── app.py                      # Main Streamlit web application (entry point)
├── requirements.txt            # List of all Python dependencies
├── README.md                   # Project overview and how to run
│
├── data/                       # All notebooks, datasets and generated files 
│   ├── candlestick_images/     # Candlestick Images pre-generated
│   ├── temp_charts/            # Temporary candlestick images for live prediction
│   ├── 01_data_collection.ipynb
│   ├── 02_sentiment_analysis.ipynb
│   ├── 03_lstm_baseline.ipynb
│   ├── 04_vit_chart_branch.ipynb
│   ├── 05_full_multimodel.ipynb
│   ├── 06_training_full_multimodel.ipynb
│   ├── 07_xgboost.ipynb
│   ├── 08_abation_study.ipynb
│   ├── Kaggle_finbert_validation_results.csv                   
│   ├── Kaggle_finbert_validation.csv
│   ├── nvda_complete_multimodel.csv                 #Final Dataset with Features
│   ├── nvda_full_multimodel.csv              
│   ├── nvda_news_sentiment.csv                      #News Sentiment from Kaggle
│   ├── nvda_prices_sentiment_vit.csv
│   ├── nvda_prices_with_sentiment.csv
│   └── nvda_prices.csv                              #Fetched data from yfinance
│
├── models/                            # Trained models and scalers
│   ├── final_multimodal_fixed.pth     # Main LSTM model
│   ├── feature_scaler.pkl
│   ├── target_scaler.pkl
│   └── xgboost_model.pkl              # For explainability
│
├── src/
│
└── temp_charts/                # Generated candlestick images during runtime 

```

## How to Run 
The web application is deployed on streamlit community server.
Copy the weblink to the browser to run the application.
https://finfusion-stock-market-predictor.streamlit.app/

Model Performance
The multimodal LSTM model achieved approximately 54% directional accuracy during training and validation. Feature importance analysis shows meaningful contributions from technical indicators and ViT-derived chart patterns.
Note: This project is for educational and research purposes only. The predictions should not be considered as financial advice.
Disclaimer
This system is developed as part of an academic project. Investment decisions based on the outputs of this model carry significant risk. Historical performance does not guarantee future results.
Author
Jaswinder Singh
