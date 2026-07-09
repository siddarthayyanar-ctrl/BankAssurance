"""app.ui
Streamlit UI for the Predictive Bancassurance Engine.

This module provides a lightweight dashboard that queries a running
FastAPI ML service to display customer-level CLV forecasts and
cross-sell recommendations for relationship managers.

Usage:
- Start the FastAPI backend locally (default: `http://127.0.0.1:8000`).
- Run this Streamlit app: `streamlit run app/ui.py`.

Expected API response (JSON):
{
    "demographics": {"age": int, "annual_income": float, "current_balance": float},
    "metrics": {"predicted_12m_clv": float},
    "cross_sell_recommendations": [{"product": str, "probability": float, "trigger": str}, ...]
}

Data files:
- Reads `data/processed_features.csv` to populate the customer selector.
"""

import streamlit as st
import requests
import pandas as pd

# Set webpage configuration
st.set_page_config(
    page_title="Predictive Bancassurance Engine",
    page_icon="🏦",
    layout="wide"
)

# App Title & Header
st.title("🏦 Predictive Bancassurance Engine")
st.markdown("### ML-Driven Customer Lifetime Value (CLV) & Cross-Sell Intelligence")
st.write("This dashboard queries our FastAPI machine learning service to provide real-time recommendations for bank relationship managers.")

st.markdown("---")

# Backend API Configuration (adjust if your FastAPI server runs elsewhere)
API_URL = "http://127.0.0.1:8000/api/v1/predict"

# Load the customer list for the selector sidebar
@st.cache_data
def get_customer_ids():
    """Load and return available customer IDs.

    Reads `data/processed_features.csv` and returns a list of values
    from the `customer_id` column. On error, logs a Streamlit error
    message and returns an empty list so the UI can handle missing data.
    """
    try:
        df = pd.read_csv("data/processed_features.csv")
        return df['customer_id'].tolist()
    except Exception as e:
        st.error(f"Error loading customer data repository: {e}")
        return []

customer_list = get_customer_ids()

# --- SIDEBAR INTERFACE ---
st.sidebar.header("Navigation Panel")
if customer_list:
    selected_customer = st.sidebar.selectbox(
        "Select Customer ID to Analyze",
        customer_list
    )
else:
    st.sidebar.error("No customers found. Verify data folder paths.")
    selected_customer = None

# --- MAIN DASHBOARD INTERFACE ---
if selected_customer:
    # Trigger API Request to FastAPI backend
    with st.spinner(f"Querying ML models for {selected_customer}..."):
        try:
            response = requests.post(API_URL, json={"customer_id": selected_customer})
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract components from JSON response
                demo = data["demographics"]
                metrics = data["metrics"]
                recommendations = data["cross_sell_recommendations"]
                
                # Layout Component: Split into 3 columns for core KPI profile metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="Customer Age", value=f"{demo['age']} years")
                with col2:
                    st.metric(label="Annual Income Profile", value=f"${demo['annual_income']:,.2f}")
                with col3:
                    st.metric(label="Current Account Balance", value=f"${demo['current_balance']:,.2f}")
                
                st.markdown("---")
                
                # Layout Component: Split screen into Core Metric and Action Items
                left_panel, right_panel = st.columns([1, 2])
                
                with left_panel:
                    st.subheader("💡 Forecasted Value")
                    st.markdown("#### **Predicted 12M CLV**")
                    # Visual callout card for financial trajectory metric
                    st.info(f"### **${metrics['predicted_12m_clv']:,.2f}**")
                    st.caption("Estimated continuous account profitability index combined with prospective policy cross-sell conversions.")
                
                with right_panel:
                    st.subheader("🎯 Cross-Sell Target Prioritization")
                    
                    if not recommendations:
                        st.warning("No high-propensity insurance recommendations identified for this customer profile at this time.")
                    else:
                        for rec in recommendations:
                            # Expandable cards for each insurance product recommendation
                            with st.expander(f"📦 {rec['product']} — Propensity: {rec['probability']*100:.1f}%", expanded=True):
                                # Progress bar indicating probability scale
                                st.progress(rec['probability'])
                                st.markdown(f"**Recommended Trigger Action:** {rec['trigger']}")
            
            else:
                st.error(f"API Error. Status Code: {response.status_code}. Message: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the FastAPI server. Please make sure your backend is running on `http://127.0.0.1:8000` via terminal execution.")
else:
    st.info("Select a customer from the sidebar panel to generate real-time predictive telemetry profiles.")