
# To run this code, save it as a .py file (e.g., app.py) and execute 'streamlit run app.py' in your terminal.

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Load the trained model and scaler ---
@st.cache_resource # Cache the model loading for better performance
def load_model_and_scaler():
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model_and_scaler()

# --- 2. Define the features (must match training order) ---
# This list should be exactly the `features` list used during training
features = [
    'year', 'quarter',
    'title_len_words', 'keyword_score',
    'kw_ai_ml', 'kw_5g', 'kw_cloud_edge',
    'kw_security', 'kw_iot', 'kw_network',
    'patent_count_lag1', 'patent_count_lag2', 'patent_count_lag4',
    'patent_count_roll4_mean', 'patent_count_roll8_mean',
    'patent_count_qoq', 'patent_count_yoy',
    'patent_type_reissue', 'patent_type_utility',
    'tech_era_legacy_pre_1990', 'tech_era_mobile_2000s',
    'tech_era_modern_2020s', 'tech_era_smartphone_2010s'
]