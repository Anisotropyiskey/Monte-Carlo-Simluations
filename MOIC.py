import streamlit as st

st.set_page_config(page_title="Barrels-to-MOIC Calculator", layout="centered")
st.title("🛢️ Barrels-to-MOIC Calculator")

st.markdown("Estimate the **Multiple on Invested Capital (MOIC)** for an upstream oil project using recoverable reserves, oil price, and capital.")

# --- Inputs ---
st.header("Input Assumptions")

# User inputs in MMbbl
barrels_mmbbl = st.number_input(
    "🛢️ Recoverable Volume (MMbbl)", 
    min_value=0.0, 
    value=20.0, 
    step=1.0, 
    format="%.2f"
)

# Convert to bbl for internal calculation
barrels = barrels_mmbbl * 1_000_000

royalty_interest_loss = st.number_input("📉 Revenue Loss from Royalty & Interest (%)", min_value=0.0, max_value=100.0, value=35.0, step=5.0)

price = st.number_input("💵 Flat Oil Price ($/bbl)", min_value=0.0, value=70.0, step=1.0, format="%.2f")

capital = st.number_input("🏗️ Capital Invested ($MM)", min_value=0.0, value=300.0, step=10.0, format="%.2f")

# --- Calculate MOIC ---
st.header("📈 MOIC Result")

if st.button("Calculate MOIC"):
    retained_fraction = 1 - royalty_interest_loss / 100
    net_revenue = barrels * retained_fraction * price
    moic = net_revenue / (capital * 1_000_000)  # Capital in dollars, not MM

    st.success(f"💹 MOIC = {moic:.2f}x")

    # Optional: interpretation
    if moic < 1.0:
        st.warning("⚠️ Project returns less than capital—likely uneconomic.")
    elif moic < 2.0:
        st.info("ℹ️ Moderate return—review risk, costs, and timing.")
    else:
        st.success("✅ Strong return potential—worth deeper evaluation.")

    # Display breakdown
    st.subheader("📊 Calculation Breakdown")
    st.markdown(f"""
    - **Recoverable Barrels:** {barrels:,.0f} bbl  
    - **Retained Revenue Fraction:** {retained_fraction:.2%}  
    - **Flat Oil Price:** ${price:.2f}/bbl  
    - **Capital Invested:** ${capital:.2f} MM  
    - **Net Revenue:** ${net_revenue:,.0f}  
    - **MOIC:** {moic:.2f}x
    """)
