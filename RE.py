import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd
from scipy.stats import spearmanr


st.set_page_config(layout="wide")

# ----------------------------
# 1. Sidebar Parameter Input (Consolidated)
# ----------------------------

st.sidebar.title("🛠️ FLow Rate Input Parameters")


run_sim = st.sidebar.button("▶️ Run Simulation")


n_sims = st.sidebar.slider(
    "Number of Simulations",
    min_value=100,
    max_value=50000,
    value=10000,
    step=100
)




# Define your parameter labels and default values
parameter_defaults = {
    "Permeability k (md)": 500,
    "Relative Permeability kro": 0.9,
    "Thickness h (ft)": 100,
    "Viscosity μ (cp)": 6,
    "Formation Volume Factor Bo": 1.2,
    "Pressure Drawdown ΔP (psi)": 1000,
    "Skin": 2.0,
    "Shape Factor CA": 31.6,
    "Drainage Area (acres)": 300,
    "Well Radius rw (ft)": 0.51,
}

def get_samples(label, default=1.0):
    with st.sidebar.expander(label):
        dist_type = st.selectbox(
            "Input type",
            ["Fixed value", "Triangular distribution", "Uniform distribution"],
            key=f"{label}_type"
        )

        if dist_type == "Fixed value":
            val = st.number_input("Value", value=default, key=f"{label}_fixed")
            return np.full(n_sims, val)

        elif dist_type == "Triangular distribution":
            min_val = st.number_input("Min", value=default * 0.5, key=f"{label}_min")
            mode_val = st.number_input("Mode", value=default, key=f"{label}_mode")
            max_val = st.number_input("Max", value=default * 1.5, key=f"{label}_max")
            return np.random.triangular(min_val, mode_val, max_val, size=n_sims)

        elif dist_type == "Uniform distribution":
            min_val = st.number_input("Min", value=default * 0.5, key=f"{label}_umin")
            max_val = st.number_input("Max", value=default * 1.5, key=f"{label}_umax")
            return np.random.uniform(min_val, max_val, size=n_sims)

# ----------------------------
# Categorized Input Summary
# ----------------------------



# Define categories
reservoir_params = [
    "Permeability k (md)",
    "Relative Permeability kro",
    "Thickness h (ft)",
    "Shape Factor CA",
    "Drainage Area (acres)",
    "Skin"
]

fluid_params = [
    "Viscosity μ (cp)",
    "Formation Volume Factor Bo"
]

well_params = [
    "Pressure Drawdown ΔP (psi)",
    "Well Radius rw (ft)"
]
st.header("🧾 Input Summary", divider=True)

# Define parameter groups
reservoir_params = [
    "Permeability k (md)",
    "Relative Permeability kro",
    "Thickness h (ft)",
    "Shape Factor CA",
    "Drainage Area (acres)",
    "Skin"
]

fluid_params = [
    "Viscosity μ (cp)",
    "Formation Volume Factor Bo"
]

well_params = [
    "Pressure Drawdown ΔP (psi)",
    "Well Radius rw (ft)"
]

# Helper to format one parameter line
def format_param_summary(label, default):
    dist_type = st.session_state.get(f"{label}_type", "Fixed value")

    if dist_type == "Fixed value":
        val = st.session_state.get(f"{label}_fixed", default)
        return f"**{label}**\nFixed = `{val}`"

    elif dist_type == "Triangular distribution":
        min_val = st.session_state.get(f"{label}_min", default * 0.5)
        mode_val = st.session_state.get(f"{label}_mode", default)
        max_val = st.session_state.get(f"{label}_max", default * 1.5)
        return f"**{label}**\nTriangular = `{min_val}` / `{mode_val}` / `{max_val}`"

    elif dist_type == "Uniform distribution":
        min_val = st.session_state.get(f"{label}_umin", default * 0.5)
        max_val = st.session_state.get(f"{label}_umax", default * 1.5)
        return f"**{label}**\nUniform = `{min_val}` – `{max_val}`"

    return f"**{label}**\n[Unknown Input Type]"

# Create three columns
col_res, col_fld, col_well = st.columns(3)

with col_res:
    st.subheader("🪨 Reservoir")
    for label in reservoir_params:
        st.markdown(format_param_summary(label, parameter_defaults[label]))

with col_fld:
    st.subheader("💧 Fluid")
    for label in fluid_params:
        st.markdown(format_param_summary(label, parameter_defaults[label]))

with col_well:
    st.subheader("🛢️ Well")
    for label in well_params:
        st.markdown(format_param_summary(label, parameter_defaults[label]))


#st.markdown("\n".join(summary_lines))



# Loop through parameters and collect samples
param_samples = {
    label: get_samples(label, default)
    for label, default in parameter_defaults.items()
}




def safe_kde_plot(q_results, p10, p50, p90, title="PDF of Flow Rate"):
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    import numpy as np

    if np.std(q_results) < 1e-6 or np.min(q_results) == np.max(q_results):
        return None

    kde = gaussian_kde(q_results)
    x = np.linspace(min(q_results), max(q_results), 300)
    pdf = kde(x)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.hist(q_results, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='white')
    ax.plot(x, pdf, color='skyblue', linewidth=2)

    for val, label, color in zip([p90, p50, p10], ['P90', 'P50', 'P10'], ['red', 'green', 'orange']):
        y_val = kde.evaluate([val])[0]
        ax.vlines(val, 0, y_val, color=color, linestyle='--')
        ax.scatter(val, y_val, color=color, s=40, zorder=5)
        ax.text(val - 0.02 * max(q_results), y_val + 0.01 * max(pdf),
                f"{label}\n({val:.1f})", color=color, fontsize=8, fontweight='bold')

    ax.set_xlabel("Flow Rate (stb/day)")
    ax.set_ylabel("Probability Density")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def safe_cdf_plot(q_results, p10, p50, p90, title="CDF of Flow Rate"):
    import matplotlib.pyplot as plt
    import numpy as np

    if np.std(q_results) < 1e-6 or np.min(q_results) == np.max(q_results):
        return None

    sorted_q = np.sort(q_results)
    cdf = np.linspace(0, 1, len(sorted_q))

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.plot(sorted_q, cdf, color='darkblue', linewidth=2)

    for val, prob, label, color in zip([p90, p50, p10], [0.10, 0.50, 0.90], ['P90', 'P50', 'P10'], ['red', 'green', 'orange']):
        ax.vlines(val, 0, prob, color=color, linestyle='--')
        ax.scatter(val, prob, color=color, s=40, zorder=5)
        ax.text(val + 0.02 * max(q_results), prob - 0.06,
                f"{label}\n({val:.1f})",
                color=color, fontsize=8, fontweight='bold')

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Flow Rate (stb/day)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ----------------------------
# 2. Run Monte Carlo Simulation
# ----------------------------

if run_sim:
    # Monte Carlo Simulation
    q_results = np.zeros(n_sims)

    for i in range(n_sims):
        k     = param_samples["Permeability k (md)"][i]
        kro   = param_samples["Relative Permeability kro"][i]
        h     = param_samples["Thickness h (ft)"][i]
        mu    = param_samples["Viscosity μ (cp)"][i]
        Bo    = param_samples["Formation Volume Factor Bo"][i]
        dp    = param_samples["Pressure Drawdown ΔP (psi)"][i]
        s     = param_samples["Skin"][i]
        CA    = param_samples["Shape Factor CA"][i]
        A     = param_samples["Drainage Area (acres)"][i] * 43560.0
        rw    = param_samples["Well Radius rw (ft)"][i]

        k_eff = k * kro

        try:
            geom_term = 0.5 * np.log(2.2458 / CA) + 0.5 * np.log(A / rw**2) + s
            q = (dp * 0.00708 * k_eff * h) / (mu * Bo * geom_term)
        except:
            q = 0.0

        q_results[i] = q

    # P-values
    p90 = np.percentile(q_results, 10)
    p50 = np.percentile(q_results, 50)
    p10 = np.percentile(q_results, 90)
    mean = np.mean(q_results)

    st.markdown(
        "<hr style='border: 2px solid lightgray; margin-top: 20px; margin-bottom: 20px;'>",
        unsafe_allow_html=True
)


    st.subheader("Flow Rate Distribution", divider=True)
    col1, col2, col3, col4 = st.columns(4)
    col3.metric("P10", f"{p10:,.1f} stb/day")
    col2.metric("P50", f"{p50:,.1f} stb/day")
    col1.metric("P90", f"{p90:,.1f} stb/day")
    col4.metric("Mean", f"{mean:,.1f} stb/day")


    # Plots
    fig_pdf = safe_kde_plot(q_results, p10, p50, p90)
    fig_cdf = safe_cdf_plot(q_results, p10, p50, p90)

    # Sensitivity Analysis
    fig_sens = None
    sensitivity = []
    for name, values in {
        "Permeability (k)": param_samples["Permeability k (md)"],
        "Rel Perm (kro)": param_samples["Relative Permeability kro"],
        "Thickness (h)": param_samples["Thickness h (ft)"],
        "Viscosity (μ)": param_samples["Viscosity μ (cp)"],
        "FVF (Bo)": param_samples["Formation Volume Factor Bo"],
        "Drawdown (ΔP)": param_samples["Pressure Drawdown ΔP (psi)"],
        "Skin": param_samples["Skin"],
        "Shape Factor (CA)": param_samples["Shape Factor CA"],
        "Area (acres)": param_samples["Drainage Area (acres)"],
        "Well Radius (rw)": param_samples["Well Radius rw (ft)"],
    }.items():
        if np.all(values == values[0]):
            continue
        rho, _ = spearmanr(values, q_results)
        if not np.isnan(rho):
            sensitivity.append((name, rho))

    if sensitivity:
        sens_df = pd.DataFrame(sensitivity, columns=["Parameter", "Spearman ρ"])
        sens_df["|ρ|"] = sens_df["Spearman ρ"].abs()
        sens_df = sens_df.sort_values(by="|ρ|", ascending=True)
        fig_sens, ax_sens = plt.subplots(figsize=(6, 4), dpi=100)
        ax_sens.barh(sens_df["Parameter"], sens_df["Spearman ρ"], color="mediumseagreen")
        ax_sens.axvline(0, color="gray", linestyle="--")
        ax_sens.set_xlabel("Spearman ρ (Correlation with Flow Rate)")
        ax_sens.set_title("Parameter Sensitivity to Flow Rate")
        fig_sens.tight_layout()

    st.markdown(
        "<hr style='border: 2px solid lightgray; margin-top: 20px; margin-bottom: 20px;'>",
        unsafe_allow_html=True
)


    # Display plots
    col1, col2, col3 = st.columns(3)
    with col1:
        if fig_pdf:
            st.caption("📈 Probability Density Fucntion of Flow Rate")
            st.pyplot(fig_pdf)
            
        else:
            st.warning("📈 PDF not available.")
    with col2:
        if fig_cdf:
            st.caption("📉 Cumulative Density Function of Flow Rate")
            st.pyplot(fig_cdf)
            
        else:
            st.warning("📉 CDF not available.")
    with col3:
        if fig_sens:
            st.caption("🌪️ Sensitivity (Tornado Chart)")
            st.pyplot(fig_sens)

        else:
            st.warning("🌪️ No varying inputs for sensitivity plot.")
else:
    st.info("👈 Adjust parameters in the sidebar and click **Run Simulation**.")

