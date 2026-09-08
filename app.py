import os
import joblib
import unicodedata
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import HistGradientBoostingRegressor

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="LaLiga Player Market Value Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def clean_text(s: str) -> str:
    """Normalize text by resolving mojibake (e.g. GÃ¼ler) and normalizing accents (Güler -> Guler)."""
    if not isinstance(s, str):
        return s
    try:
        if "Ã" in s or "Â" in s:
            s = s.encode("latin1").decode("utf-8")
    except Exception:
        pass
    # Normalize unicode to clean standard ASCII (removes combining diacritics)
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

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
# Data & Model Pipeline
# ---------------------------------------------------------
@st.cache_resource
def load_all_resources():
    """Load dataset, clean names and diacritics, run feature engineering, and train/load the GBDT model."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "laliga_transfer_model.pkl")
    csv_path = os.path.join(base_dir, "Data_LaLiga_2024_25.csv")

    if not os.path.exists(csv_path):
        st.error(f"❌ Missing dataset file: 'Data_LaLiga_2024_25.csv' in {base_dir}")
        st.stop()

    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except Exception:
        df = pd.read_csv(csv_path, encoding="latin1")

    # Clean text columns to avoid mojibake like 'Arda GÃ¼ler' -> 'Arda Guler'
    for col in ["Player", "Team", "Nation", "Position"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # Clean European decimal formatting (commas -> dots)
    float_cols = ["xG", "xAG", "Gls/90", "Ast/90", "xG/90", "xAG/90"]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", ".").astype(float).fillna(0.0)

    # Feature Engineering
    df["Age_Sq"] = df["Age"] ** 2
    tier_1 = ["Real Madrid", "Barcelona", "Atletico Madrid"]
    tier_2 = ["Real Sociedad", "Athletic Club", "Villarreal", "Betis", "Girona"]

    def get_tier(team_name: str) -> int:
        if team_name in tier_1:
            return 1
        elif team_name in tier_2:
            return 2
        return 3

    df["Club_Tier"] = df["Team"].apply(get_tier)

    # One-hot encode position
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

    # Generate predictions across the full dataset
    predicted_log = model.predict(X)
    df["Predicted_Value_EUR"] = np.maximum(np.expm1(predicted_log), 250_000)

    return model, df, feature_cols


# Load resources
model, df, feature_cols = load_all_resources()

# ---------------------------------------------------------
# Sidebar Navigation (Model Summary Removed)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚽ LaLiga Valuation AI")
    st.caption("2024 / 2025 Season Market Value Model")

    menu = st.radio(
        "Navigation",
        [
            "🔍 Player Explorer",
            "🎛️ Scout Simulator",
            "📊 League Insights",
        ],
        index=0,
    )

    st.markdown("---")
    st.caption("LaLiga Santander 2024–25 Market Value Estimator")

# ---------------------------------------------------------
# VIEW 1: PLAYER EXPLORER
# ---------------------------------------------------------
if menu == "🔍 Player Explorer":
    st.title("🔍 La Liga Player Valuation Explorer")
    st.markdown("Search or filter any player from the 2024–25 season to evaluate their predicted transfer market value.")

    col1, col2, col3 = st.columns([2, 2, 3])

    with col1:
        team_list = ["All Clubs"] + sorted(df["Team"].unique().tolist())
        selected_team = st.selectbox("Filter by Club", team_list)

    with col2:
        pos_list = ["All Positions"] + sorted(df["Position"].unique().tolist())
        selected_pos = st.selectbox("Filter by Position", pos_list)

    # Filter dataframe
    filtered_df = df.copy()
    if selected_team != "All Clubs":
        filtered_df = filtered_df[filtered_df["Team"] == selected_team]
    if selected_pos != "All Positions":
        filtered_df = filtered_df[filtered_df["Position"] == selected_pos]

    with col3:
        player_names = sorted(filtered_df["Player"].unique().tolist())
        if not player_names:
            st.warning("No players found with current filters.")
            st.stop()
        
        # Default to Arda Guler if available to highlight the fix, else first player
        default_index = player_names.index("Arda Guler") if "Arda Guler" in player_names else 0
        selected_player_name = st.selectbox("Select Player", player_names, index=default_index)

    # Get player record
    player = df[df["Player"] == selected_player_name].iloc[0]
    percentile = (df["Predicted_Value_EUR"] < player["Predicted_Value_EUR"]).mean() * 100

    st.markdown("---")

    # Main Profile Card
    profile_col, stats_col = st.columns([1.5, 2.5])

    with profile_col:
        st.markdown(
            f"""
            <div style="background-color: #1A2230; padding: 22px; border-radius: 14px; border: 1px solid #2D3748;">
                <h2 style="margin: 0; color: #FFFFFF;">{player['Player']}</h2>
                <h4 style="margin-top: 4px; color: #E63946;">{player['Team']} • {player['Position']}</h4>
                <p style="color: #A0AEC0; margin-bottom: 12px;">Nationality: <b>{player['Nation']}</b> | Age: <b>{player['Age']}</b></p>
                <hr style="border: 0.5px solid #2D3748; margin: 12px 0;">
                <p style="color: #A0AEC0; margin: 0; font-size: 13px; font-weight: 600;">ESTIMATED MARKET VALUE</p>
                <h1 style="color: #48BB78; margin: 2px 0 6px 0; font-size: 40px;">{format_currency(player['Predicted_Value_EUR'])}</h1>
                <p style="color: #CBD5E0; margin-top: 4px; font-size: 14px;">
                    📈 <b>Top {100 - percentile:.1f}%</b> in La Liga (higher than {percentile:.1f}% of players)
                </p>
                <div style="background-color: #2D3748; border-radius: 6px; padding: 8px; margin-top: 12px;">
                    <span style="font-size: 13px; color: #E2E8F0;">Club Tier: <b>Tier {player['Club_Tier']}</b></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with stats_col:
        st.markdown("### 📋 2024–25 Performance Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Matches", f"{int(player['Match'])}")
        m2.metric("Minutes", f"{int(player['Minutes']):,}")
        m3.metric("Goals", f"{int(player['Goals'])}")
        m4.metric("Assists", f"{int(player['Assists'])}")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Expected Goals (xG)", f"{player['xG']:.2f}")
        m6.metric("Expected Assists (xAG)", f"{player['xAG']:.2f}")
        m7.metric("xG / 90", f"{player['xG/90']:.2f}")
        m8.metric("xAG / 90", f"{player['xAG/90']:.2f}")

    st.markdown("---")

    # Comparison Bar Chart against Positional Average
    st.subheader(f"📊 {player['Player']} vs. La Liga {player['Position']} Average")
    pos_avg = df[df["Position"] == player["Position"]].mean(numeric_only=True)

    metrics_to_compare = ["Goals", "Assists", "xG", "xAG", "Gls/90", "xG/90"]
    player_vals = [player[m] for m in metrics_to_compare]
    avg_vals = [pos_avg[m] for m in metrics_to_compare]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=metrics_to_compare,
        y=player_vals,
        name=player["Player"],
        marker_color="#E63946"
    ))
    fig.add_trace(go.Bar(
        x=metrics_to_compare,
        y=avg_vals,
        name=f"Position Avg ({player['Position']})",
        marker_color="#4A5568"
    ))

    fig.update_layout(
        barmode="group",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FFFFFF"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=20),
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# VIEW 2: SCOUT SIMULATOR
# ---------------------------------------------------------
elif menu == "🎛️ Scout Simulator":
    st.title("🎛️ Custom Player Valuation Simulator")
    st.markdown("Simulate a custom or prospective player's market value based on attributes and season performance.")

    with st.form("scout_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("#### 👤 Profile")
            sim_age = st.slider("Age", min_value=16, max_value=40, value=23)
            sim_pos = st.selectbox("Position", ["Forward (FW)", "Midfielder (MF)", "Defender (DF)", "Goalkeeper (GK)"])
            sim_club = st.selectbox("Team Affiliation", sorted(df["Team"].unique().tolist()), index=0)

        with c2:
            st.markdown("#### ⏱️ Playing Time")
            sim_matches = st.slider("Matches Played", min_value=1, max_value=38, value=25)
            sim_minutes = st.slider("Total Minutes", min_value=10, max_value=3420, value=1950)
            sim_yellow = st.number_input("Yellow Cards", min_value=0, max_value=20, value=3)
            sim_red = st.number_input("Red Cards", min_value=0, max_value=5, value=0)

        with c3:
            st.markdown("#### 🎯 Output & Metrics")
            sim_goals = st.number_input("Goals", min_value=0, max_value=50, value=8)
            sim_assists = st.number_input("Assists", min_value=0, max_value=30, value=5)
            sim_xg = st.number_input("Expected Goals (xG)", min_value=0.0, max_value=40.0, value=7.5, step=0.1)
            sim_xag = st.number_input("Expected Assists (xAG)", min_value=0.0, max_value=30.0, value=4.2, step=0.1)

        submitted = st.form_submit_button("🔮 Calculate Predicted Valuation", use_container_width=True)

    # Compute values
    n90 = max(sim_minutes / 90.0, 0.1)
    sim_gls_90 = sim_goals / n90
    sim_ast_90 = sim_assists / n90
    sim_xg_90 = sim_xg / n90
    sim_xag_90 = sim_xag / n90

    tier_1 = ["Real Madrid", "Barcelona", "Atletico Madrid"]
    tier_2 = ["Real Sociedad", "Athletic Club", "Villarreal", "Betis", "Girona"]
    sim_tier = 1 if sim_club in tier_1 else (2 if sim_club in tier_2 else 3)

    # Build feature row matching model training columns
    sim_data = {
        "Age": [sim_age],
        "Age_Sq": [sim_age ** 2],
        "Match": [sim_matches],
        "Minutes": [sim_minutes],
        "Goals": [sim_goals],
        "Assists": [sim_assists],
        "Yellow_Cards": [sim_yellow],
        "Red_Cards": [sim_red],
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
    sim_df = pd.DataFrame(sim_data)[feature_cols]

    pred_log = model.predict(sim_df)[0]
    predicted_val = float(np.maximum(np.expm1(pred_log), 250_000))
    low_bound = predicted_val * 0.90
    high_bound = predicted_val * 1.10

    st.markdown("---")
    st.subheader("🎯 Valuation Outcome")

    res_col1, res_col2, res_col3 = st.columns([2, 2, 3])

    with res_col1:
        st.metric("Estimated Market Value", format_currency(predicted_val))

    with res_col2:
        st.metric("Expected Valuation Range (±10%)", f"{format_currency(low_bound)} – {format_currency(high_bound)}")

    with res_col3:
        sim_percentile = (df["Predicted_Value_EUR"] < predicted_val).mean() * 100
        st.info(f"💡 This valuation places the prospect in the **Top {100 - sim_percentile:.1f}%** of all players in La Liga.")

    # Show comparable players
    st.markdown("#### 👥 Most Comparable Current La Liga Players")
    df["val_diff"] = (df["Predicted_Value_EUR"] - predicted_val).abs()
    similar_players = (
        df[df["Position"] == sim_pos]
        .sort_values("val_diff")
        .head(5)[["Player", "Team", "Age", "Goals", "Assists", "Predicted_Value_EUR"]]
    )
    similar_players["Predicted Value"] = similar_players["Predicted_Value_EUR"].apply(format_currency)
    st.dataframe(
        similar_players.drop(columns=["Predicted_Value_EUR"]),
        hide_index=True,
        use_container_width=True,
    )

# ---------------------------------------------------------
# VIEW 3: LEAGUE INSIGHTS
# ---------------------------------------------------------
elif menu == "📊 League Insights":
    st.title("📊 La Liga 2024–25 Market Overview")
    st.markdown("Aggregate valuation trends, team rankings, and position benchmarks across the league.")

    tab1, tab2, tab3 = st.tabs(["🏆 Most Valuable Players", "🏟️ Club Valuations", "📋 Full Database"])

    with tab1:
        top_n = st.slider("Top Players Count", min_value=5, max_value=30, value=15)
        top_players = df.nlargest(top_n, "Predicted_Value_EUR")[["Player", "Team", "Position", "Predicted_Value_EUR", "Goals", "Assists"]]
        
        fig_top = px.bar(
            top_players.iloc[::-1],
            x="Predicted_Value_EUR",
            y="Player",
            color="Team",
            orientation="h",
            labels={"Predicted_Value_EUR": "Estimated Market Value (€)", "Player": "Player"},
            title=f"Top {top_n} Most Valuable Players in La Liga",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_top.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FFFFFF"),
            height=500
        )
        st.plotly_chart(fig_top, use_container_width=True)

    with tab2:
        club_summary = (
            df.groupby("Team")
            .agg(
                Squad_Size=("Player", "count"),
                Total_Value=("Predicted_Value_EUR", "sum"),
                Avg_Value=("Predicted_Value_EUR", "mean"),
            )
            .reset_index()
            .sort_values("Total_Value", ascending=False)
        )
        
        fig_club = px.bar(
            club_summary,
            x="Team",
            y="Total_Value",
            labels={"Total_Value": "Total Squad Market Value (€)", "Team": "Club"},
            title="Total Estimated Squad Market Value by Club",
            color="Total_Value",
            color_continuous_scale="Reds"
        )
        fig_club.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FFFFFF"),
            xaxis_tickangle=-45,
            height=450
        )
        st.plotly_chart(fig_club, use_container_width=True)

    with tab3:
        display_df = df[["Player", "Team", "Nation", "Position", "Age", "Match", "Minutes", "Goals", "Assists", "Predicted_Value_EUR"]].copy()
        display_df["Predicted_Value_Formatted"] = display_df["Predicted_Value_EUR"].apply(format_currency)
        st.dataframe(display_df, use_container_width=True)
        
        csv_data = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Full Player Valuations CSV",
            data=csv_data,
            file_name="laliga_2024_25_valuations.csv",
            mime="text/csv"
        )