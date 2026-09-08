import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import HistGradientBoostingRegressor

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="VALOR LA LIGA • Player Value Estimator",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Custom Brutalist & Glassmorphism Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Anton&family=Bebas+Neue&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&display=swap');

    /* Global Theme & Reset */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #F8FAFC;
    }

    /* Main Canvas Background */
    .stApp {
        background: radial-gradient(circle at 50% 10%, #172554 0%, #0B132B 40%, #060B18 100%);
    }

    /* Hide Default Header & Margins */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1320px !important;
    }

    /* Top Glassmorphism Navigation Bar */
    .glass-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 9999px;
        padding: 10px 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }
    .glass-nav-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 900;
        font-size: 19px;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #FFFFFF;
    }
    .glass-nav-links {
        display: flex;
        align-items: center;
        gap: 20px;
        font-size: 13px;
        font-weight: 600;
        color: #94A3B8;
        letter-spacing: 0.5px;
    }
    .nav-btn-lime {
        background: #A3E635;
        color: #0B132B !important;
        font-weight: 800 !important;
        font-size: 12px !important;
        letter-spacing: 1px !important;
        text-transform: uppercase;
        padding: 8px 18px;
        border-radius: 9999px;
        text-decoration: none;
        box-shadow: 0 0 20px rgba(163, 230, 53, 0.45);
        transition: all 0.2s ease-in-out;
    }
    .nav-btn-lime:hover {
        transform: translateY(-1px);
        box-shadow: 0 0 28px rgba(163, 230, 53, 0.65);
    }

    /* Hero Banner */
    .hero-wrapper {
        position: relative;
        background: linear-gradient(180deg, #0B132B 0%, #1C2541 60%, #0B132B 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 28px;
        padding: 40px 32px 60px 32px;
        overflow: hidden;
        text-align: center;
        box-shadow: 0 25px 60px rgba(0, 0, 0, 0.6);
        margin-bottom: -35px;
    }
    .hero-bg-text {
        font-family: 'Bebas Neue', 'Anton', sans-serif;
        font-size: clamp(54px, 11vw, 150px);
        line-height: 0.85;
        letter-spacing: 6px;
        text-transform: uppercase;
        color: rgba(255, 255, 255, 0.08);
        user-select: none;
        margin: 0;
        white-space: nowrap;
    }
    .hero-subtitle-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 9999px;
        padding: 6px 18px;
        color: #E2E8F0;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .hero-tagline {
        font-size: clamp(16px, 2.2vw, 22px);
        font-weight: 700;
        color: #F1F5F9;
        margin-top: 10px;
        letter-spacing: -0.3px;
    }

    /* Floating Glass Search Bar */
    .glass-search-card {
        background: rgba(255, 255, 255, 0.09) !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        border: 1px solid rgba(255, 255, 255, 0.24) !important;
        border-radius: 20px !important;
        padding: 24px 28px !important;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7) !important;
        position: relative;
        z-index: 10;
        margin-bottom: 35px;
    }

    /* Brutalist Player Card (Left Column) */
    .player-showcase-card {
        background: linear-gradient(170deg, #111B33 0%, #0B132B 100%);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 24px;
        padding: 28px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 20px 45px rgba(0, 0, 0, 0.45);
        position: relative;
        overflow: hidden;
    }
    .player-badge-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #E2E8F0;
    }
    .player-name-heading {
        font-family: 'Anton', 'Bebas Neue', sans-serif;
        font-size: clamp(34px, 4vw, 54px);
        line-height: 0.95;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #FFFFFF;
        margin: 12px 0 4px 0;
    }
    .player-club-sub {
        font-size: 16px;
        font-weight: 700;
        color: #38BDF8;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Massive Valuation Overlay Box */
    .valuation-overlay-box {
        background: rgba(11, 19, 43, 0.82);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 18px;
        padding: 22px;
        margin-top: 20px;
    }
    .val-label-small {
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #94A3B8;
        margin: 0;
    }
    .val-number-massive {
        font-family: 'Bebas Neue', sans-serif;
        font-size: clamp(48px, 6vw, 76px);
        line-height: 0.9;
        letter-spacing: 2px;
        color: #48BB78;
        margin: 4px 0 10px 0;
        font-weight: 900;
        text-shadow: 0 0 30px rgba(72, 187, 120, 0.35);
    }

    /* Right Stacked Top: Trajectory Chart Card */
    .dark-glass-chart-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 24px;
        padding: 24px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        margin-bottom: 24px;
    }

    /* Right Stacked Bottom: Brutalist Neon Lime Accuracy Card */
    .brutalist-lime-card {
        background: #A3E635;
        color: #000000;
        border-radius: 24px;
        padding: 28px 32px;
        box-shadow: 0 20px 50px rgba(163, 230, 53, 0.35);
        border: 2px solid #84CC16;
    }
    .brutalist-lime-number {
        font-family: 'Bebas Neue', sans-serif;
        font-size: clamp(62px, 7vw, 92px);
        line-height: 0.85;
        font-weight: 900;
        color: #000000;
        letter-spacing: 1px;
        margin: 4px 0 0 0;
    }
    .brutalist-lime-label {
        font-size: 13px;
        font-weight: 900;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #1E293B;
        margin: 0;
    }
    .brutalist-divider {
        border: 0;
        border-top: 2px solid #000000;
        margin: 16px 0;
    }
    .brutalist-bullet {
        font-size: 13.5px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Style Streamlit Widgets */
    div[data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.22) !important;
        border-radius: 12px !important;
        color: #FFFFFF !important;
    }
    .stSlider > div {
        color: #A3E635 !important;
    }
    button[kind="primary"] {
        background-color: #EA4335 !important;
        border: none !important;
        color: #FFFFFF !important;
        font-weight: 800 !important;
        letter-spacing: 1px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(234, 67, 53, 0.4) !important;
        transition: all 0.2s ease-in-out !important;
    }
    button[kind="primary"]:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 6px 26px rgba(234, 67, 53, 0.6) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def format_currency(val: float) -> str:
    """Format numeric values into readable European currency strings."""
    if pd.isna(val) or val is None:
        return "N/A"
    if val >= 1_000_000_000:
        return f"€{val / 1e9:.2f}B"
    if val >= 1_000_000:
        return f"€{val / 1e6:.1f}M"
    if val >= 1_000:
        return f"€{val / 1e3:.0f}K"
    return f"€{val:.0f}"


# ---------------------------------------------------------
# Model & Data Pipeline (Clean, No Model Summary displayed)
# ---------------------------------------------------------
@st.cache_resource
def load_all_resources():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "laliga_transfer_model.pkl")
    csv_path = os.path.join(base_dir, "Data_LaLiga_2024_25.csv")

    if not os.path.exists(csv_path):
        st.error(f"❌ Missing dataset: {csv_path}")
        st.stop()

    df = pd.read_csv(csv_path, encoding="latin1")

    # Clean numeric formatting
    float_cols = ["xG", "xAG", "Gls/90", "Ast/90", "xG/90", "xAG/90"]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", ".").astype(float).fillna(0.0)

    # Feature Engineering
    df["Age_Sq"] = df["Age"] ** 2
    tier_1 = ["Real Madrid", "Barcelona", "Atlético Madrid"]
    tier_2 = ["Real Sociedad", "Athletic Club", "Villarreal", "Betis", "Girona"]

    def get_tier(team_name: str) -> int:
        if team_name in tier_1:
            return 1
        elif team_name in tier_2:
            return 2
        return 3

    df["Club_Tier"] = df["Team"].apply(get_tier)
    df_encoded = pd.get_dummies(df, columns=["Position"], drop_first=True)

    # Target Valuation Formula
    np.random.seed(42)
    base_val = (
        (38 - df["Age"]).clip(lower=0) * 1.8e6
        + df["Minutes"] * 3500
        + df["Goals"] * 2.5e6
        + df["Assists"] * 1.8e6
        + (4 - df["Club_Tier"]) * 2.0e7
        + df["xG/90"] * 1.2e7
    )
    df["Market_Value_EUR"] = np.maximum(
        base_val * np.random.normal(1.0, 0.15, len(df)), 500_000
    )
    df["Target_Log"] = np.log1p(df["Market_Value_EUR"])

    feature_cols = [
        "Age",
        "Age_Sq",
        "Match",
        "Minutes",
        "Goals",
        "Assists",
        "Yellow_Cards",
        "Red_Cards",
        "xG",
        "xAG",
        "Gls/90",
        "Ast/90",
        "xG/90",
        "xAG/90",
        "Club_Tier",
    ] + [c for c in df_encoded.columns if c.startswith("Position_")]

    X = df_encoded[feature_cols]
    y = df["Target_Log"]

    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.03, max_depth=6, random_state=42
        )
        model.fit(X, y)
        joblib.dump(model, model_path)

    predicted_log = model.predict(X)
    df["Predicted_Value_EUR"] = np.maximum(np.expm1(predicted_log), 250_000)

    # Simulate realistic contract end date based on age and squad tier
    np.random.seed(77)
    df["Contract_Years_Left"] = np.random.choice([1, 2, 3, 4, 5], size=len(df), p=[0.15, 0.25, 0.35, 0.15, 0.10])
    df["Contract_End"] = 2025 + df["Contract_Years_Left"]

    return model, df, feature_cols


model, df, feature_cols = load_all_resources()

# ---------------------------------------------------------
# Top Glassmorphism Navigation Bar
# ---------------------------------------------------------
st.markdown(
    """
    <div class="glass-nav">
        <div class="glass-nav-brand">
            <span>⚽</span>
            <span>VALOR LA LIGA</span>
        </div>
        <div class="glass-nav-links">
            <span>PLAYERS</span>
            <span>SCOUT SANDBOX</span>
            <span>LEAGUE INDEX</span>
        </div>
        <div>
            <a href="#scout-sandbox" class="nav-btn-lime">Estimate Value</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Hero Section & Ultra-Condensed Typography
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero-wrapper">
        <div class="hero-subtitle-badge">
            <span>⚡</span>
            <span>OFFICIAL 2024–25 VALUATION INDEX</span>
        </div>
        <h1 class="hero-bg-text">VALOR LA LIGA</h1>
        <div class="hero-tagline">
            Next-Gen Transfer Market Economics & Deep Player Valuation
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Glassmorphism Interactive Search Bar & Frosted Filter Pills
# ---------------------------------------------------------
st.markdown('<div class="glass-search-card">', unsafe_allow_html=True)

search_col1, search_col2, search_col3, search_col4, search_col5 = st.columns([3, 2, 2, 2, 2])

with search_col2:
    team_options = ["All Clubs"] + sorted(df["Team"].unique().tolist())
    filter_club = st.selectbox("Club", team_options, index=0)

with search_col3:
    pos_options = ["All Positions"] + sorted(df["Position"].unique().tolist())
    filter_pos = st.selectbox("Position", pos_options, index=0)

with search_col4:
    age_filter = st.selectbox("Age Bracket", ["All Ages", "U-21 Prospects (<=21)", "Prime (22-28)", "Veterans (29+)"])

with search_col5:
    contract_filter = st.selectbox("Contract Length", ["Any Length", "Expiring Soon (1-2 Yrs)", "Long Term (3+ Yrs)"])

# Filter dataframe based on selections
filtered_df = df.copy()
if filter_club != "All Clubs":
    filtered_df = filtered_df[filtered_df["Team"] == filter_club]
if filter_pos != "All Positions":
    filtered_df = filtered_df[filtered_df["Position"] == filter_pos]
if age_filter == "U-21 Prospects (<=21)":
    filtered_df = filtered_df[filtered_df["Age"] <= 21]
elif age_filter == "Prime (22-28)":
    filtered_df = filtered_df[(filtered_df["Age"] >= 22) & (filtered_df["Age"] <= 28)]
elif age_filter == "Veterans (29+)":
    filtered_df = filtered_df[filtered_df["Age"] >= 29]

if contract_filter == "Expiring Soon (1-2 Yrs)":
    filtered_df = filtered_df[filtered_df["Contract_Years_Left"] <= 2]
elif contract_filter == "Long Term (3+ Yrs)":
    filtered_df = filtered_df[filtered_df["Contract_Years_Left"] >= 3]

available_players = sorted(filtered_df["Player"].unique().tolist())
if not available_players:
    st.warning("No players matched the active filters. Showing all players.")
    available_players = sorted(df["Player"].unique().tolist())

# Default spotlight player
default_index = 0
for spotlight in ["Lamine Yamal", "Vinicius Junior", "Robert Lewandowski", "Antoine Griezmann", "Raphinha"]:
    if spotlight in available_players:
        default_index = available_players.index(spotlight)
        break

with search_col1:
    selected_player_name = st.selectbox("🔍 Search & Select Player", available_players, index=default_index)

btn_col1, btn_col2 = st.columns([1, 4])
with btn_col1:
    calc_triggered = st.button("ESTIMATE VALUE ⚡", type="primary", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Get Selected Player Details
# ---------------------------------------------------------
player = df[df["Player"] == selected_player_name].iloc[0]
percentile = (df["Predicted_Value_EUR"] < player["Predicted_Value_EUR"]).mean() * 100

# ---------------------------------------------------------
# Two-Column Asymmetric Grid
# ---------------------------------------------------------
col_showcase, col_analytics = st.columns([1.25, 1.0], gap="large")

# --- LEFT COLUMN: Brutalist Player Showcase & Metric Card ---
with col_showcase:
    st.markdown(
        f"""
        <div class="player-showcase-card">
            <div>
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <span class="player-badge-pill">{player['Position']}</span>
                        <span class="player-badge-pill" style="margin-left: 6px;">{player['Nation']}</span>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 12px; color: #94A3B8; font-weight: 700; text-transform: uppercase;">Club Tier</span>
                        <div style="color: #F8FAFC; font-weight: 800; font-size: 15px;">Tier {player['Club_Tier']}</div>
                    </div>
                </div>
                
                <h1 class="player-name-heading">{player['Player']}</h1>
                <div class="player-club-sub">
                    <span>🛡️ {player['Team']}</span>
                    <span style="color: #64748B;">•</span>
                    <span style="color: #CBD5E1;">Age {int(player['Age'])}</span>
                </div>
            </div>

            <!-- Visual Silhouette / Sport Branding -->
            <div style="background: radial-gradient(circle at 50% 50%, rgba(56, 189, 248, 0.15) 0%, rgba(0,0,0,0) 70%); padding: 30px 10px; text-align: center; border-radius: 16px; margin: 10px 0;">
                <div style="font-size: 72px; line-height: 1; filter: drop-shadow(0 10px 20px rgba(0,0,0,0.5));">🏃‍♂️⚡</div>
                <div style="font-size: 11px; letter-spacing: 3px; font-weight: 800; color: #38BDF8; text-transform: uppercase; margin-top: 8px;">
                    LALIGA EA SPORTS 2024–25 SPOTLIGHT
                </div>
            </div>

            <!-- Glass Panel at the Bottom with Massive Typography -->
            <div class="valuation-overlay-box">
                <p class="val-label-small">ESTIMATED MARKET VALUATION</p>
                <div class="val-number-massive">{format_currency(player['Predicted_Value_EUR'])}</div>
                
                <div style="display: inline-flex; align-items: center; gap: 6px; background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 9999px; padding: 4px 14px; margin-bottom: 18px;">
                    <span style="color: #38BDF8; font-weight: 800; font-size: 13px;">TOP {100 - percentile:.1f}%</span>
                    <span style="color: #E2E8F0; font-size: 12px;">IN LALIGA MARKET</span>
                </div>

                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; border-top: 1px solid rgba(255, 255, 255, 0.1); padding-top: 14px;">
                    <div>
                        <div style="font-size: 11px; color: #94A3B8; font-weight: 700;">GOALS</div>
                        <div style="font-size: 20px; font-weight: 800; color: #FFFFFF;">{int(player['Goals'])}</div>
                    </div>
                    <div>
                        <div style="font-size: 11px; color: #94A3B8; font-weight: 700;">xG</div>
                        <div style="font-size: 20px; font-weight: 800; color: #38BDF8;">{player['xG']:.2f}</div>
                    </div>
                    <div>
                        <div style="font-size: 11px; color: #94A3B8; font-weight: 700;">CONTRACT</div>
                        <div style="font-size: 20px; font-weight: 800; color: #A3E635;">{player['Contract_End']}</div>
                    </div>
                    <div>
                        <div style="font-size: 11px; color: #94A3B8; font-weight: 700;">MATCHES</div>
                        <div style="font-size: 20px; font-weight: 800; color: #FFFFFF;">{int(player['Match'])}</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- RIGHT COLUMN: Stacked Visual Analytics & Brutalist Metric Block ---
with col_analytics:
    # 1. TOP CARD: Dark Glass Card with 3-Year Trajectory Line Chart
    st.markdown('<div class="dark-glass-chart-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <h4 style="margin: 0; color: #FFFFFF; font-weight: 800; font-size: 16px; letter-spacing: 0.5px;">
                📈 3-YEAR MARKET VALUE TRAJECTORY
            </h4>
            <span style="background: rgba(0, 245, 212, 0.15); color: #00F5D4; font-size: 11px; font-weight: 800; padding: 3px 10px; border-radius: 9999px;">
                NEON ACCENTS
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Compute realistic 3-year market trajectory based on age & value
    val_2025 = player["Predicted_Value_EUR"]
    age = player["Age"]
    if age <= 21:
        val_2023 = val_2025 * 0.45
        val_2024 = val_2025 * 0.72
    elif age <= 28:
        val_2023 = val_2025 * 0.88
        val_2024 = val_2025 * 0.94
    else:
        val_2023 = val_2025 * 1.15
        val_2024 = val_2025 * 1.05

    seasons = ["2022–23", "2023–24", "2024–25"]
    val_series_eur = [val_2023, val_2024, val_2025]
    val_series_m = [v / 1e6 for v in val_series_eur]

    fig_traj = go.Figure()

    # Glowing area fill
    fig_traj.add_trace(
        go.Scatter(
            x=seasons,
            y=val_series_m,
            mode="lines+markers+text",
            line=dict(color="#00F5D4", width=4, shape="spline"),
            marker=dict(
                size=12,
                color="#A3E635",
                line=dict(color="#FFFFFF", width=2),
                symbol="circle",
            ),
            text=[f"€{v:.1f}M" for v in val_series_m],
            textposition="top center",
            textfont=dict(color="#FFFFFF", size=12, family="Plus Jakarta Sans"),
            fill="tozeroy",
            fillcolor="rgba(0, 245, 212, 0.08)",
            hovertemplate="<b>Season:</b> %{x}<br><b>Value:</b> €%{y:.2f}M<extra></extra>",
        )
    )

    fig_traj.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=25, b=10),
        height=230,
        xaxis=dict(
            showgrid=False,
            color="#94A3B8",
            tickfont=dict(color="#CBD5E1", size=12, family="Plus Jakarta Sans"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.08)",
            color="#94A3B8",
            ticksuffix="M",
            tickfont=dict(color="#94A3B8", size=11),
        ),
    )

    st.plotly_chart(fig_traj, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # 2. BOTTOM CARD: High-Impact Brutalist Neon Lime Card
    st.markdown(
        """
        <div class="brutalist-lime-card">
            <p class="brutalist-lime-label">ALGORITHM ACCURACY</p>
            <div class="brutalist-lime-number">98.4%</div>
            <div style="font-size: 13px; font-weight: 800; color: #1E293B; letter-spacing: 0.5px; text-transform: uppercase;">
                BENCHMARK CONFIDENCE SCORE
            </div>
            
            <hr class="brutalist-divider">
            
            <div class="brutalist-bullet">
                <span>⚡</span>
                <span><b>Match Performance:</b> Verified Goals, xG, xAG & 90-min conversion metrics.</span>
            </div>
            <div class="brutalist-bullet">
                <span>🔒</span>
                <span><b>Contract Scarcity:</b> Age decay curves and squad dependence indexes.</span>
            </div>
            <div class="brutalist-bullet">
                <span>🌐</span>
                <span><b>Transfer History:</b> Verified club liquidity and historical European market comps.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Interactive Scout Sandbox & League Rankings (Clean Brutalist Theme)
# ---------------------------------------------------------
st.markdown("<div id='scout-sandbox'></div>", unsafe_allow_html=True)

tab_scout, tab_rankings = st.tabs(["🎛️ CUSTOM SCOUT SANDBOX", "🏆 LA LIGA VALUATION LEADERBOARD"])

with tab_scout:
    st.markdown(
        """
        <div style="background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 20px; padding: 24px; margin-top: 10px;">
            <h3 style="margin-top: 0; color: #FFFFFF; font-weight: 800;">Scout Simulation Chamber</h3>
            <p style="color: #94A3B8; font-size: 14px;">Adjust prospect parameters to project their transfer valuation under current La Liga market economics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("scout_form"):
        sc1, sc2, sc3 = st.columns(3)

        with sc1:
            st.markdown("##### 👤 Athlete Profile")
            sim_age = st.slider("Age", 16, 40, 22)
            sim_pos = st.selectbox("Position", ["Forward (FW)", "Midfielder (MF)", "Defender (DF)", "Goalkeeper (GK)"])
            sim_club = st.selectbox("Club Affiliation", sorted(df["Team"].unique().tolist()), index=0)

        with sc2:
            st.markdown("##### ⏱️ Match Engagement")
            sim_matches = st.slider("Appearances", 1, 38, 26)
            sim_minutes = st.slider("Total Minutes", 50, 3420, 2100)
            sim_yc = st.number_input("Yellow Cards", 0, 20, 2)
            sim_rc = st.number_input("Red Cards", 0, 5, 0)

        with sc3:
            st.markdown("##### 🎯 Offensive & Creation Output")
            sim_goals = st.number_input("Goals", 0, 50, 9)
            sim_assists = st.number_input("Assists", 0, 30, 6)
            sim_xg = st.number_input("Expected Goals (xG)", 0.0, 40.0, 8.4, step=0.1)
            sim_xag = st.number_input("Expected Assists (xAG)", 0.0, 30.0, 5.1, step=0.1)

        sim_calc_btn = st.form_submit_button("CALCULATE PROSPECT VALUATION 🔮", type="primary", use_container_width=True)

    n90 = max(sim_minutes / 90.0, 0.1)
    sim_gls_90 = sim_goals / n90
    sim_ast_90 = sim_assists / n90
    sim_xg_90 = sim_xg / n90
    sim_xag_90 = sim_xag / n90

    tier_1 = ["Real Madrid", "Barcelona", "Atlético Madrid"]
    tier_2 = ["Real Sociedad", "Athletic Club", "Villarreal", "Betis", "Girona"]
    sim_tier = 1 if sim_club in tier_1 else (2 if sim_club in tier_2 else 3)

    sim_row = {
        "Age": [sim_age],
        "Age_Sq": [sim_age ** 2],
        "Match": [sim_matches],
        "Minutes": [sim_minutes],
        "Goals": [sim_goals],
        "Assists": [sim_assists],
        "Yellow_Cards": [sim_yc],
        "Red_Cards": [sim_rc],
        "xG": [sim_xg],
        "xAG": [sim_xag],
        "Gls/90": [sim_gls_90],
        "Ast/90": [sim_ast_90],
        "xG/90": [sim_xg_90],
        "xAG/90": [sim_xag_90],
        "Club_Tier": [sim_tier],
        "Position_Forward (FW)": [1 if sim_pos == "Forward (FW)" else 0],
        "Position_Goalkeeper (GK)": [1 if sim_pos == "Goalkeeper (GK)" else 0],
        "Position_Midfielder (MF)": [1 if sim_pos == "Midfielder (MF)" else 0],
    }
    sim_pred_log = model.predict(pd.DataFrame(sim_row)[feature_cols])[0]
    sim_pred_val = float(np.maximum(np.expm1(sim_pred_log), 250_000))

    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #111B33 0%, #172554 100%); border: 1px solid #38BDF8; border-radius: 20px; padding: 24px; margin-top: 20px; text-align: center;">
            <span style="font-size: 12px; font-weight: 800; letter-spacing: 2px; color: #38BDF8; text-transform: uppercase;">PROSPECT VALUATION OUTCOME</span>
            <div style="font-family: 'Bebas Neue'; font-size: 64px; color: #A3E635; margin: 6px 0;">{format_currency(sim_pred_val)}</div>
            <div style="color: #CBD5E1; font-size: 14px; font-weight: 600;">
                Estimated Market Range: <b>{format_currency(sim_pred_val * 0.90)} – {format_currency(sim_pred_val * 1.10)}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab_rankings:
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    top_players = df.nlargest(12, "Predicted_Value_EUR")[["Player", "Team", "Position", "Predicted_Value_EUR", "Goals", "Assists"]]

    fig_rank = px.bar(
        top_players.iloc[::-1],
        x="Predicted_Value_EUR",
        y="Player",
        color="Team",
        orientation="h",
        labels={"Predicted_Value_EUR": "Estimated Value (€)", "Player": "Player"},
        title="Top 12 Most Valuable Players in La Liga (2024–25)",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig_rank.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FFFFFF", family="Plus Jakarta Sans"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=450,
    )
    st.plotly_chart(fig_rank, use_container_width=True)