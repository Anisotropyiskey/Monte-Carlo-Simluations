# app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import triang, beta, spearmanr, gaussian_kde

# ----- Plotting Functions -----

def plot_pdf_with_annotations(ooip_mmstb, p90, p50, p10):
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)

    x_min, x_max = min(ooip_mmstb), max(ooip_mmstb)
    kde = gaussian_kde(ooip_mmstb, bw_method=0.25)
    pdf_p90, pdf_p50, pdf_p10 = kde(p90), kde(p50), kde(p10)

    sns.histplot(ooip_mmstb, bins=100, kde=True, stat="density", ax=ax, color="skyblue", edgecolor="white")

    ax.vlines(p90, 0, pdf_p90, color='red', linestyle='--')
    ax.vlines(p50, 0, pdf_p50, color='green', linestyle='--')
    ax.vlines(p10, 0, pdf_p10, color='orange', linestyle='--')

    ax.scatter(p90, pdf_p90, color='red', s=40, zorder=5)
    ax.scatter(p50, pdf_p50, color='green', s=40, zorder=5)
    ax.scatter(p10, pdf_p10, color='orange', s=40, zorder=5)

    ax.text(p90 + 2, pdf_p90 + 0.001, f'P90\n({p90:.1f})', fontsize=8, fontweight='bold')
    ax.text(p50 + 2, pdf_p50 + 0.001, f'P50\n({p50:.1f})', fontsize=8, fontweight='bold')
    ax.text(p10 + 2, pdf_p10 + 0.001, f'P10\n({p10:.1f})', fontsize=8, fontweight='bold')

    ax.set_title("Probability Distribution of OOIP", fontsize=11, fontweight='bold')
    ax.set_xlabel("OOIP (MMSTB)", fontsize=10)
    ax.set_ylabel("Probability Density", fontsize=10)
    ax.set_xlim(x_min, x_max)
    fig.tight_layout()
    return fig

def plot_cdf_with_annotations(ooip_mmstb, p90, p50, p10):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    sorted_ooip = np.sort(ooip_mmstb)
    cdf = np.linspace(0, 1, len(sorted_ooip))
    x_min, x_max = min(ooip_mmstb), max(ooip_mmstb)

    ax.plot(sorted_ooip, cdf, color='darkblue', linewidth=2)
    ax.vlines(p90, 0, 0.10, color='red', linestyle='--')
    ax.vlines(p50, 0, 0.50, color='green', linestyle='--')
    ax.vlines(p10, 0, 0.90, color='orange', linestyle='--')

    ax.scatter(p90, 0.10, color='red', s=40, zorder=5)
    ax.scatter(p50, 0.50, color='green', s=40, zorder=5)
    ax.scatter(p10, 0.90, color='orange', s=40, zorder=5)

    ax.text(p90 + 2, 0.10 + 0.03, f'P90\n({p90:.1f}, 10%)', fontsize=8, fontweight='bold')
    ax.text(p50 + 2, 0.50 + 0.03, f'P50\n({p50:.1f}, 50%)', fontsize=8, fontweight='bold')
    ax.text(p10 + 2, 0.90 + 0.03, f'P10\n({p10:.1f}, 90%)', fontsize=8, fontweight='bold')

    ax.set_ylim(0, 1.05)
    ax.set_xlim(x_min, x_max)
    ax.set_title("Cumulative Distribution (CDF) of OOIP", fontsize=11, fontweight='bold')
    ax.set_xlabel("OOIP (MMSTB)", fontsize=10)
    ax.set_ylabel("Cumulative Probability", fontsize=10)
    fig.tight_layout()
    return fig

# ----- Streamlit UI -----

st.set_page_config(layout="wide")
st.title("OOIP Monte Carlo Simulator (Anchored)")

st.sidebar.header("Simulation Controls")
n_sim = st.sidebar.slider("Number of Simulations", 100, 150000, 1000, step=100)
base_weight = st.sidebar.slider("Base Distribution Weight", 0.0000001, 0.999999, 0.7)

# ----- Base Distribution Parameters -----
st.sidebar.markdown("### 📊 Base Distribution Parameters")

with st.sidebar.expander("📐 Area (acres)"):
    a_min = st.slider("Min", 100, 3000, 700, step=10, key="a_min")
    a_mode = st.slider("Mode", 100, 3000, 1000, step=10, key="a_mode")
    a_max = st.slider("Max", 100, 3000, 1500, step=10, key="a_max")

with st.sidebar.expander("📏 Net Pay (ft)"):
    h_min = st.slider("Min", 10, 200, 93, key="h_min")
    h_mode = st.slider("Mode", 10, 200, 110, key="h_mode")
    h_max = st.slider("Max", 10, 200, 120, key="h_max")

with st.sidebar.expander("🧱 Porosity (fraction)"):
    phi_min = st.slider("Min", 0.05, 0.35, 0.15, step=0.01, key="phi_min")
    phi_mode = st.slider("Mode", 0.05, 0.35, 0.20, step=0.01, key="phi_mode")
    phi_max = st.slider("Max", 0.05, 0.35, 0.25, step=0.01, key="phi_max")

with st.sidebar.expander("💧 Water Saturation (Sw)"):
    sw_min = st.slider("Min", 0.0, 1.0, 0.10, step=0.01, key="sw_min")
    sw_mode = st.slider("Mode", 0.0, 1.0, 0.20, step=0.01, key="sw_mode")
    sw_max = st.slider("Max", 0.0, 1.0, 0.50, step=0.01, key="sw_max")

with st.sidebar.expander("🛢️ Bo (RB/STB)"):
    bo_min = st.slider("Min", 1.0, 2.0, 1.20, step=0.01, key="bo_min")
    bo_mode = st.slider("Mode", 1.0, 2.0, 1.25, step=0.01, key="bo_mode")
    bo_max = st.slider("Max", 1.0, 2.0, 1.30, step=0.01, key="bo_max")

jitter_type = st.sidebar.selectbox(
    "Anchor Jitter Type",
    ["Uniform", "Triangular", "Normal", "Fixed"]
)

st.sidebar.markdown("### Anchor Jitter Controls")
with st.sidebar.expander("🎚️ Jitter Settings"):
    if jitter_type == "Uniform":
        st.markdown("**Uniform Spread Controls**")
        area_spread = st.slider("Area Spread", 0, 500, 50, step=10)
        h_spread = st.slider("Net Pay Spread", 0, 50, 5)
        phi_spread = st.slider("Porosity Spread", 0.0, 0.05, 0.005, step=0.001)
        sw_spread = st.slider("Sw Spread", 0.0, 0.1, 0.02, step=0.005)
        bo_spread = st.slider("Bo Spread", 0.0, 0.1, 0.02, step=0.005)

    elif jitter_type == "Normal":
        st.markdown("**Normal Std Dev Controls**")
        area_std = st.slider("Area Std Dev", 0, 300, 25, step=5)
        h_std = st.slider("Net Pay Std Dev", 0, 20, 3)
        phi_std = st.slider("Porosity Std Dev", 0.0, 0.05, 0.002, step=0.001)
        sw_std = st.slider("Sw Std Dev", 0.0, 0.1, 0.01, step=0.005)
        bo_std = st.slider("Bo Std Dev", 0.0, 0.1, 0.01, step=0.005)

    elif jitter_type == "Fixed":
        st.info("Using fixed values with no spread.")

    elif jitter_type == "Triangular":
        #st.markdown("**Triangular Distribution Controls**")

        with st.sidebar.expander("📐 Area (acres)"):
            area_tri_low = st.slider("Area Low", 100, 2000, 600)
            area_tri_mode = st.slider("Area Mode", 100, 3000, 700)
            area_tri_high = st.slider("Area High", 100, 3000, 900)

        with st.sidebar.expander("📏 Net Pay (ft)"):
            h_tri_low = st.slider("Net Pay Low", 50, 200, 93)
            h_tri_mode = st.slider("Net Pay Mode", 50, 200, 100)
            h_tri_high = st.slider("Net Pay High", 50, 200, 120)

        with st.sidebar.expander("🧱 Porosity (fraction)"):
            phi_tri_low = st.slider("Porosity Low", 0.05, 0.30, 0.18, step=0.005)
            phi_tri_mode = st.slider("Porosity Mode", 0.05, 0.30, 0.20, step=0.005)
            phi_tri_high = st.slider("Porosity High", 0.05, 0.30, 0.25, step=0.005)

        with st.sidebar.expander("💧 Water Saturation (Sw)"):
            sw_tri_low = st.slider("Sw Low", 0.0, 1.0, 0.15, step=0.01)
            sw_tri_mode = st.slider("Sw Mode", 0.0, 1.0, 0.30, step=0.01)
            sw_tri_high = st.slider("Sw High", 0.0, 1.0, 0.45, step=0.01)

        with st.sidebar.expander("🛢️ Bo (RB/STB)"):
            bo_tri_low = st.slider("Bo Low", 1.0, 2.0, 1.20, step=0.01)
            bo_tri_mode = st.slider("Bo Mode", 1.0, 2.0, 1.25, step=0.01)
            bo_tri_high = st.slider("Bo High", 1.0, 2.0, 1.30, step=0.01)


# (Optional: add controls for Triangular if you want to override default low/mode/high too)


# ----- Simulation Logic -----

def run_simulation(
    n_sim, base_weight, jitter_type,
    a_min, a_mode, a_max,
    h_min, h_mode, h_max,
    phi_min, phi_mode, phi_max,
    sw_min, sw_mode, sw_max,
    bo_min, bo_mode, bo_max
):
    np.random.seed(42)
    n_base = int(n_sim * base_weight)
    n_anchor = n_sim - n_base
    n_each = n_anchor // 3


    area_base = triang.rvs((a_mode - a_min)/(a_max - a_min), loc=a_min, scale=a_max - a_min, size=n_base)
    alpha_h = 4
    beta_h_val = alpha_h * (h_max - h_mode) / (h_mode - h_min)
    h_base = beta.rvs(alpha_h, beta_h_val, size=n_base) * (h_max - h_min) + h_min
    alpha_phi = 4
    beta_phi_val = alpha_phi * (phi_max - phi_mode) / (phi_mode - phi_min)
    phi_base = beta.rvs(alpha_phi, beta_phi_val, size=n_base) * (phi_max - phi_min) + phi_min
    alpha_sw = 4
    beta_sw_val = alpha_sw * (sw_max - sw_mode) / (sw_mode - sw_min)
    sw_base = beta.rvs(alpha_sw, beta_sw_val, size=n_base) * (sw_max - sw_min) + sw_min
    bo_base = triang.rvs((bo_mode - bo_min)/(bo_max - bo_min), loc=bo_min, scale=bo_max - bo_min, size=n_base)




    # --- Anchor Sampling Based on Jitter Type ---

    if jitter_type == "Uniform":
        def jitter(center, spread, size):
            return np.random.uniform(center - spread, center + spread, size)

        area_anchor = np.concatenate([
            jitter(600, 50, n_each), jitter(700, 50, n_each), jitter(900, 50, n_each)])
        h_anchor = np.concatenate([
            jitter(93, 5, n_each), jitter(100, 5, n_each), jitter(120, 5, n_each)])
        phi_anchor = np.concatenate([
            jitter(0.18, 0.005, n_each), jitter(0.20, 0.005, n_each), jitter(0.25, 0.005, n_each)])
        sw_anchor = np.concatenate([
            jitter(0.45, 0.02, n_each), jitter(0.30, 0.02, n_each), jitter(0.15, 0.02, n_each)])
        bo_anchor = np.concatenate([
            jitter(1.30, 0.02, n_each), jitter(1.25, 0.02, n_each), jitter(1.20, 0.02, n_each)])

    elif jitter_type == "Triangular":
        def jitter(low, mode, high, size):
            return np.random.triangular(left=low, mode=mode, right=high, size=size)

        area_anchor = np.concatenate([
            jitter(area_tri_low, area_tri_mode, area_tri_high, n_each),
            jitter(area_tri_low, area_tri_mode, area_tri_high, n_each),
            jitter(area_tri_low, area_tri_mode, area_tri_high, n_each)
        ])

        h_anchor = np.concatenate([
            jitter(h_tri_low, h_tri_mode, h_tri_high, n_each),
            jitter(h_tri_low, h_tri_mode, h_tri_high, n_each),
            jitter(h_tri_low, h_tri_mode, h_tri_high, n_each)
        ])

        phi_anchor = np.concatenate([
            jitter(phi_tri_low, phi_tri_mode, phi_tri_high, n_each),
            jitter(phi_tri_low, phi_tri_mode, phi_tri_high, n_each),
            jitter(phi_tri_low, phi_tri_mode, phi_tri_high, n_each)
        ])

        sw_anchor = np.concatenate([
            jitter(sw_tri_low, sw_tri_mode, sw_tri_high, n_each),
            jitter(sw_tri_low, sw_tri_mode, sw_tri_high, n_each),
            jitter(sw_tri_low, sw_tri_mode, sw_tri_high, n_each)
        ])

        bo_anchor = np.concatenate([
            jitter(bo_tri_low, bo_tri_mode, bo_tri_high, n_each),
            jitter(bo_tri_low, bo_tri_mode, bo_tri_high, n_each),
            jitter(bo_tri_low, bo_tri_mode, bo_tri_high, n_each)
        ])


    elif jitter_type == "Normal":
        def jitter(mean, std, size):
            return np.random.normal(mean, std, size)

        area_anchor = np.concatenate([
            jitter(600, 25, n_each), jitter(700, 25, n_each), jitter(900, 25, n_each)])
        h_anchor = np.concatenate([
            jitter(93, 3, n_each), jitter(100, 3, n_each), jitter(120, 3, n_each)])
        phi_anchor = np.concatenate([
            jitter(0.18, 0.002, n_each), jitter(0.20, 0.002, n_each), jitter(0.25, 0.002, n_each)])
        sw_anchor = np.concatenate([
            jitter(0.45, 0.01, n_each), jitter(0.30, 0.01, n_each), jitter(0.15, 0.01, n_each)])
        bo_anchor = np.concatenate([
            jitter(1.30, 0.01, n_each), jitter(1.25, 0.01, n_each), jitter(1.20, 0.01, n_each)])

    elif jitter_type == "Fixed":
        def jitter(value, *_):
            return np.full(n_each, value)

        area_anchor = np.concatenate([
            jitter(600), jitter(700), jitter(900)])
        h_anchor = np.concatenate([
            jitter(93), jitter(100), jitter(120)])
        phi_anchor = np.concatenate([
            jitter(0.18), jitter(0.20), jitter(0.25)])
        sw_anchor = np.concatenate([
            jitter(0.45), jitter(0.30), jitter(0.15)])
        bo_anchor = np.concatenate([
            jitter(1.30), jitter(1.25), jitter(1.20)])

    else:
        raise ValueError(f"Unsupported jitter type: {jitter_type}")

    area = np.concatenate([area_base, area_anchor])
    h = np.concatenate([h_base, h_anchor])
    phi = np.concatenate([phi_base, phi_anchor])
    sw = np.concatenate([sw_base, sw_anchor])
    bo = np.concatenate([bo_base, bo_anchor])
    nrv_acreft = area * h

    ooip = 7758 * area * h * phi * (1 - sw) / bo
    ooip_mmstb = ooip / 1e6

    p10 = np.percentile(ooip_mmstb, 90)
    p50 = np.percentile(ooip_mmstb, 50)
    p90 = np.percentile(ooip_mmstb, 10)
    mean = np.mean(ooip_mmstb) 

    nrv_p90 = np.percentile(nrv_acreft, 10)
    nrv_p50 = np.percentile(nrv_acreft, 50)
    nrv_p10 = np.percentile(nrv_acreft, 90)
    nrv_mean = np.mean(nrv_acreft)

    inputs = {'Area': area, 'Net Pay': h, 'Porosity': phi, 'Sw': sw, 'Bo': bo}
    correlations = {k: spearmanr(v, ooip)[0] for k, v in inputs.items()}
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    labels, values = zip(*sorted_corr)

    fig_pdf = plot_pdf_with_annotations(ooip_mmstb, p90, p50, p10)
    fig_cdf = plot_cdf_with_annotations(ooip_mmstb, p90, p50, p10)

    # --- Tornado Plot with color-coded bars ---
    fig_tornado, ax = plt.subplots(figsize=(6, 4), dpi=100)

    # Color map: red for negative, blue for positive
    bar_colors = ['red' if val < 0 else 'steelblue' for val in values]

    ax.barh(labels, values, color=bar_colors)

    ax.set_title("Tornado: Sensitivity", fontsize=11, fontweight='bold')
    ax.set_xlabel("Spearman Correlation", fontsize=10)
    ax.set_xlim(-1, 1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    # Grid
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    fig_tornado.tight_layout()

    return {
        'pdf_fig': fig_pdf,
        'cdf_fig': fig_cdf,
        'tornado_fig': fig_tornado,
        'p10': p10,
        'p50': p50,
        'p90': p90,
        'mean': mean,
        'correlations': correlations,
        'nrv_p10': nrv_p10,
        'nrv_p50': nrv_p50,
        'nrv_p90': nrv_p90,
        'nrv_mean': nrv_mean,
        'nrv_array': nrv_acreft  # Optional: for export or charting

    }

# ----- Run Button -----

if st.button("Run Simulation"):
    with st.spinner("Running simulation..."):
        results = run_simulation(
            n_sim, base_weight, jitter_type,
            a_min, a_mode, a_max,
            h_min, h_mode, h_max,
            phi_min, phi_mode, phi_max,
            sw_min, sw_mode, sw_max,
            bo_min, bo_mode, bo_max
        )

    st.subheader("🛢️ OOIP Distribution")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("P90", f"{results['p90']:.2f} MMSTB")
    col2.metric("P50", f"{results['p50']:.2f} MMSTB")
    col3.metric("P10", f"{results['p10']:.2f} MMSTB")
    col4.metric("Mean", f"{results['mean']:.2f} MMSTB")

    st.subheader("🪨 Net Rock Volume (NRV)")

    col_n1, col_n2, col_n3, col_n4 = st.columns(4)
    col_n1.metric("NRV P90", f"{results['nrv_p90']:.0f} acre-ft")
    col_n2.metric("NRV P50", f"{results['nrv_p50']:.0f} acre-ft")
    col_n3.metric("NRV P10", f"{results['nrv_p10']:.0f} acre-ft")
    col_n4.metric("NRV Mean", f"{results['nrv_mean']:.0f} acre-ft")

if 'results' in locals():
    st.subheader("Distributions & Sensitivity")
    col_pdf, col_cdf, col_tornado = st.columns(3)

    with col_pdf:
        st.markdown("**PDF**")
        st.pyplot(results['pdf_fig'])

    with col_cdf:
        st.markdown("**CDF**")
        st.pyplot(results['cdf_fig'])

    with col_tornado:
        st.markdown("**Tornado**")
        st.pyplot(results['tornado_fig'])
