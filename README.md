# ⚽ La Liga Player Market Value Prediction AI (2024–25)

An end-to-end Machine Learning web application that predicts transfer market valuations for Spanish La Liga footballers based on 2024–25 season performance metrics and squad dynamics.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 Key Features

1. **🔍 Player Explorer**:
   - Filter and search through every registered 2024–25 La Liga player across all 20 clubs.
   - Comprehensive profile cards featuring estimated market value, league percentile ranking, and real match statistics.
   - Interactive comparative charts measuring individual metrics against positional league averages.

2. **🎛️ Scout Simulator**:
   - Custom evaluation sandbox for scouts and analysts.
   - Input age, prospective club tier, minutes, goals, assists, expected goals (xG), and discipline records.
   - Instantly predicts market valuation range ($\pm 10\%$) and reveals the most comparable current La Liga players.

3. **📊 League Market Insights**:
   - Leaderboards of the most valuable players in Spain.
   - Club-by-club squad valuation rankings.
   - Searchable, sortable dataset table with 1-click CSV download.

4. **ℹ️ Machine Learning Methodology**:
   - Powered by a **Histogram-Based Gradient Boosting Regressor (`HistGradientBoostingRegressor`)**.
   - Incorporates non-linear age decay ($Age^2$), club hierarchy tiers, and underlying expected metrics ($xG$, $xAG$, per-90 rates).
   - Trained on log-transformed valuations achieving an $R^2 \approx 0.91$ with low MAPE (~8%).

---

## 🚀 Quick Start Locally

### 1. Clone the repository
```bash
git clone https://github.com/SleepySum/LaLiga_Player_Prediction.git
cd LaLiga_Player_Prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the application
```bash
streamlit run app.py
```
Visit `http://localhost:8501` in your browser.

---

## ☁️ Deployment

### Streamlit Community Cloud (Recommended)
1. Fork or push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New app"**, select repository `SleepySum/LaLiga_Player_Prediction`, set branch to `main`, and main file to `app.py`.
4. Click **Deploy!**
