import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, spearmanr
import pandas as pd

st.set_page_config(layout="wide")

# ------------------------------------------
# Sidebar Inputs – OOIP, RF, Rate, Decline
# ------------------------------------------
st.sidebar.title("📊 Monte Carlo Rate & EUR Forecast")

n_sims = st.sidebar.slider("Number of Simulations", 100, 50000, 10000, step=100)

run_forecast = st.sidebar.button("▶️ Run Forecast Simulation")

# OOIP and Recovery Factor inputs
st.sidebar.markdown("---")
st.sidebar.subheader("🪨 OOIP & Recovery Factor")
ooip_min = st.sidebar.number_input("OOIP Min (MMSTB)", value=100, min_value=0, max_value=10000, step=10)
ooip_mode = st.sidebar.number_input("OOIP Mode (MMSTB)", value=250, min_value=0, max_value=10000, step=10)
ooip_max = st.sidebar.number_input("OOIP Max (MMSTB)", value=500, min_value=0, max_value=10000, step=10)
rf_min = st.sidebar.number_input("Recovery Factor Min (%)", value=25)
rf_mode = st.sidebar.number_input("Recovery Factor Mode (%)", value=30)
rf_max = st.sidebar.number_input("Recovery Factor Max (%)", value=35)

# Initial Rate input
st.sidebar.markdown("---")
st.sidebar.subheader("🚀 Initial Rate (stb/day)")
rate_min = st.sidebar.number_input("Rate Min", value=3000, min_value=0, max_value=30000, step=100)
rate_mode = st.sidebar.number_input("Rate Mode", value=7500, min_value=0, max_value=30000, step=100)
rate_max = st.sidebar.number_input("Rate Max", value=15000, min_value=0, max_value=30000, step=100)

# Decline inputs
st.sidebar.markdown("---")
st.sidebar.subheader("📉 Decline Curve Inputs")
decline_model = st.sidebar.selectbox("Decline Model", ["Hyperbolic", "Exponential", "Plateau then Hyperbolic", "Plateau then Exponential"])
forecast_months = st.sidebar.slider("Forecast Duration (months)", 12, 360, 120, step=12)
b_factor = st.sidebar.slider("b-factor (Hyperbolic Exponent)", 0.0, 1.5, 0.5, 0.05)
Di = st.sidebar.slider("Initial Decline Rate (Di)", 0.01, 0.8, 0.3, 0.01)
plateau_duration = st.sidebar.slider("Plateau Duration (months)", 0, 60, 24, step=6)

# ----------------------------
# Monte Carlo OOIP & RF & Rate
# ----------------------------#


if run_forecast:
    # Run your forecast here

    # your simulation code goes here

    st.title("🛢️ Monte Carlo Simulation – Deepwater GoM Forecasting")

    ooip_samples = np.random.triangular(ooip_min, ooip_mode, ooip_max, size=n_sims)
    rf_samples = np.random.triangular(rf_min, rf_mode, rf_max, size=n_sims) / 100
    eur_max_samples = ooip_samples * rf_samples

    q_results = np.random.triangular(rate_min, rate_mode, rate_max, size=n_sims)
    months = np.arange(0, forecast_months + 1)

    q_forecasts, Np_forecasts = [], []

    for i in range(n_sims):
        qi = q_results[i]
        eur_max = eur_max_samples[i]
        q_t = np.zeros_like(months)

        if decline_model == "Hyperbolic":
            q_t = qi / (1 + b_factor * Di * months) ** (1 / b_factor)
        elif decline_model == "Exponential":
            q_t = qi * np.exp(-Di * months)
        elif decline_model == "Plateau then Hyperbolic":
            for j, t in enumerate(months):
                q_t[j] = qi if t <= plateau_duration else qi / (1 + b_factor * Di * (t - plateau_duration)) ** (1 / b_factor)
        elif decline_model == "Plateau then Exponential":
            for j, t in enumerate(months):
                q_t[j] = qi if t <= plateau_duration else qi * np.exp(-Di * (t - plateau_duration))

        Np_t = np.cumsum(q_t * 30.44) / 1_000_000  # MMBO
        Np_t = np.minimum(Np_t, eur_max)
        q_forecasts.append(q_t)
        Np_forecasts.append(Np_t)

    q_forecasts = np.array(q_forecasts)
    Np_forecasts = np.array(Np_forecasts)
    eur = Np_forecasts[:, -1]

    # Summary Stats
    p90, p50, p10 = np.percentile(q_results, [10, 50, 90])
    q_mean = np.mean(q_results)

    p90_eur, p50_eur, p10_eur = np.percentile(eur, [10, 50, 90])
    eur_mean = np.mean(eur)


    # ----------------------------
    # EUR PDF and CDF Plots
    # ----------------------------

    fig_pdf, ax_pdf = plt.subplots(figsize=(6, 4), dpi=100)
    kde = gaussian_kde(eur)
    x = np.linspace(min(eur), max(eur), 300)
    pdf = kde(x)
    ax_pdf.hist(eur, bins=40, density=True, alpha=0.5, color='skyblue', edgecolor='white')
    ax_pdf.plot(x, pdf, color='skyblue', linewidth=2)
    for val, label, color in zip([p90_eur, p50_eur, p10_eur], ['P90', 'P50', 'P10'], ['red', 'green', 'orange']):
        y_val = kde.evaluate([val])[0]
        ax_pdf.vlines(val, 0, y_val, color=color, linestyle='--')
        ax_pdf.scatter(val, y_val, color=color, s=40, zorder=5)
        ax_pdf.text(val + 0.4, y_val - 0.0, f"{label}\n({val:.2f})", color=color, fontsize=8)
    ax_pdf.set_title("EUR Distribution (PDF)")
    ax_pdf.set_xlabel("EUR (MMBO)")
    ax_pdf.set_ylabel("Probability Density")
    fig_pdf.tight_layout()

    fig_cdf, ax_cdf = plt.subplots(figsize=(6, 4), dpi=100)
    sorted_eur = np.sort(eur)
    cdf = np.linspace(0, 1, len(sorted_eur))
    ax_cdf.plot(sorted_eur, cdf, color='darkblue', linewidth=2)
    for val, prob, label, color in zip([p90_eur, p50_eur, p10_eur], [0.10, 0.50, 0.90], ['P90', 'P50', 'P10'], ['red', 'green', 'orange']):
        ax_cdf.vlines(val, 0, prob, color=color, linestyle='--')
        ax_cdf.scatter(val, prob, color=color, s=40, zorder=5)
        ax_cdf.text(val + 0.5, prob - 0.06, f"{label}\n({val:.2f})", color=color, fontsize=8)
    ax_cdf.set_title("EUR Cumulative Distribution")
    ax_cdf.set_xlabel("EUR (MMBO)")
    ax_cdf.set_ylabel("Cumulative Probability")
    ax_cdf.set_ylim(0, 1.05)
    fig_cdf.tight_layout()

    # ----------------------------
    # Forecast Rate Plot
    # ----------------------------

    q_p10 = np.percentile(q_forecasts, 90, axis=0)
    q_p50 = np.percentile(q_forecasts, 50, axis=0)
    q_p90 = np.percentile(q_forecasts, 10, axis=0)
    fig_forecast, axf = plt.subplots(figsize=(6, 4), dpi=100)
    axf.plot(months, q_p10, linestyle="--", color="red", label="P10")
    axf.plot(months, q_p50, linestyle="-", color="green", label="P50")
    axf.plot(months, q_p90, linestyle="--", color="orange", label="P90")
    axf.set_xlabel("Time (months)")
    axf.set_ylabel("Production Rate (stb/day)")
    axf.set_title(f"Forecast – {decline_model}")
    axf.legend()
    fig_forecast.tight_layout()

    # ----------------------------
    # Summary – Initial Rate
    # ----------------------------
    st.subheader("🚀 Initial Rate Distribution", divider=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("P90", f"{p90:,.0f} stb/day")
    col2.metric("P50", f"{p50:,.0f} stb/day")
    col3.metric("P10", f"{p10:,.0f} stb/day")
    col4.metric("Mean", f"{q_mean:,.0f} stb/day")




    # ----------------------------
    # Summary – EUR
    # ----------------------------
    st.subheader("🛢️ EUR Distribution", divider=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("P90", f"{p90_eur:.2f} MMBO")
    col2.metric("P50", f"{p50_eur:.2f} MMBO")
    col3.metric("P10", f"{p10_eur:.2f} MMBO")
    col4.metric("Mean", f"{eur_mean:.2f} MMBO")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.pyplot(fig_pdf)
        st.caption("📈 EUR Distribution (PDF)")
    with col2:
        st.pyplot(fig_cdf)
        st.caption("📉 EUR Cumulative Distribution")
    with col3:
        st.pyplot(fig_forecast)
        st.caption("📈 Forecasted Production Rates")

