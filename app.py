import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'laliga_transfer_model.pkl')
    csv_path = os.path.join(base_dir, 'Data_LaLiga_2024_25.csv')
    
    # 1. If model file already exists, load and return it
    if os.path.exists(model_path):
        return joblib.load(model_path)
    
    # 2. If model file is missing, auto-train and save it locally
    st.info("⚡ First-time setup: Training machine learning model on local dataset...")
    
    if not os.path.exists(csv_path):
        st.error(f"❌ Cannot train model! Missing 'Data_LaLiga_2024_25.csv' in:\n{base_dir}")
        st.stop()
        
    df = pd.read_csv(csv_path)
    
    # Clean European decimal formatting
    float_cols = ['xG', 'xAG', 'Gls/90', 'Ast/90', 'xG/90', 'xAG/90']
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
            
    # Feature Engineering
    df['Age_Sq'] = df['Age'] ** 2
    tier_1 = ['Real Madrid', 'Barcelona', 'Atlético Madrid']
    tier_2 = ['Real Sociedad', 'Athletic Club', 'Villarreal', 'Betis', 'Girona']
    df['Club_Tier'] = df['Team'].apply(lambda x: 1 if x in tier_1 else (2 if x in tier_2 else 3))
    df_encoded = pd.get_dummies(df, columns=['Position'], drop_first=True)
    
    # Target Valuation Formula
    np.random.seed(42)
    base_val = (
        (38 - df['Age']).clip(lower=0) * 1.8e6 +
        df['Minutes'] * 3500 +
        df['Goals'] * 2.5e6 +
        df['Assists'] * 1.8e6 +
        (4 - df['Club_Tier']) * 2.0e7 +
        df['xG/90'] * 1.2e7
    )
    df['Market_Value_EUR'] = np.maximum(base_val * np.random.normal(1.0, 0.15, len(df)), 500000)
    df['Target_Log'] = np.log1p(df['Market_Value_EUR'])
    
    features = ['Age', 'Age_Sq', 'Match', 'Minutes', 'Goals', 'Assists', 'Yellow_Cards', 'Red_Cards',
                'xG', 'xAG', 'Gls/90', 'Ast/90', 'xG/90', 'xAG/90', 'Club_Tier'] + \
               [c for c in df_encoded.columns if c.startswith('Position_')]
               
    X = df_encoded[features]
    y = df['Target_Log']
    
    # Train GBDT Model
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_depth=6, random_state=42)
    model.fit(X, y)
    
    # Save model locally for future sessions
    joblib.dump(model, model_path)
    st.success("✅ Model trained and saved as 'laliga_transfer_model.pkl'!")
    
    return model