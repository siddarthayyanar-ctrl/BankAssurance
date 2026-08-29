import streamlit as st
import requests
import pandas as pd
import sqlite3

import os
import json
from pathlib import Path
from typing import Optional

from google import genai

CONTEXT_FILE_PATH = "app/context/knowledge.json"


def load_context(file_path: str) -> str:
    path = Path(file_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Context file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Context path is not a file: {path}")

    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except Exception as exc:
        raise ValueError(f"Failed to parse JSON context file {path}: {exc}")

    # Accept several JSON shapes: a top-level string, dict with 'context' or 'entries', or a list
    if isinstance(data, str):
        return data.strip()
    if isinstance(data, dict):
        if 'context' in data and isinstance(data['context'], str):
            return data['context'].strip()
        if 'entries' in data and isinstance(data['entries'], list):
            return "\n".join(str(e) for e in data['entries']).strip()
        # fallback to a pretty-printed JSON blob
        return json.dumps(data, indent=2)
    if isinstance(data, list):
        return "\n".join(str(e) for e in data).strip()

    return str(data)

def build_prompt(question: str, context: Optional[str] = None) -> str:
    if context:
        return (
            "Use the following context to answer the question.\n\n"
            "Context:\n"
            f"{context}\n\n"
            "Question:\n"
            f"{question}"
        )
    return question

def generate_ai_response(prompt_text: str, context: Optional[str] = None, model_name: Optional[str] = None) -> str:
    #api_key = os.environ.get("GEMINI_API_KEY")
    #api_key = os.environ.get("GEMINI_API_KEY")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "Gemini chat is unavailable because GEMINI_API_KEY is not set."

    try:
        client = genai.Client(api_key=api_key)
    except Exception as exc:
        return f"Gemini client could not be initialized: {exc}"

    resolved_model = model_name or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
    prompt = build_prompt(prompt_text, context)

    # Do not want to load gen AI for testing purpose, for development, we can comment this out and return a mock response 
    # return prompt  # Mock response for testing without actual API call
    try:
        response = client.models.generate_content(
            model=resolved_model,
            contents=prompt
        )
        return getattr(response, "text", str(response)).strip()
    except Exception as exc:
        return f"Gemini chat error: {exc}"

# Set webpage configuration
st.set_page_config(
    page_title="Predictive Bancassurance Engine - V2.0",
    page_icon="🏦",
    layout="wide"
)

# App Title & Header
st.title("🏦 Enterprise Predictive Bancassurance Engine")
st.markdown("### Production Platform v2.0: SQL Relational Layer, SHAP Explainability & MLOps Monitoring")

# Backend API Configuration
BASE_API_URL = "http://127.0.0.1:8000/api/v2"

# Local DB Helper to load customer dropdown
def get_customer_ids():
    try:
        conn = sqlite3.connect("data/bancassurance.db")
        df = pd.read_sql_query("SELECT customer_id FROM customer_profiles", conn)
        conn.close()
        return df['customer_id'].tolist()
    except Exception as e:
        st.error(f"Database error: {e}")
        return []

customer_list = get_customer_ids()

# Load GenAI knowledge context for the chat tab
ai_context = None
try:
    ai_context = load_context(CONTEXT_FILE_PATH)
except Exception:
    ai_context = None

# --- DESIGN APPLICATION TABS ---
tab_portfolio, tab_sandbox, tab_chat, tab_monitor = st.tabs([
    "🎯 Customer Portfolio Analysis", 
    "🧪 Live Sandbox Predictor", 
    "🤖 Chat BOT",
    "📊 System Operations Monitor"
])

# ==========================================
# TAB 1: CUSTOMER PORTFOLIO ANALYSIS
# ==========================================
with tab_portfolio:
    st.subheader("Database Record Search")
    
    if customer_list:
        selected_customer = st.selectbox(
            "Select Customer ID for real-time relational lookup:",
            customer_list,
            key="portfolio_select"
        )
        
        if selected_customer:
            with st.spinner(f"Querying models and calculating feature attribution weights..."):
                try:
                    response = requests.post(f"{BASE_API_URL}/predict", json={"customer_id": selected_customer})
                    if response.status_code == 200:
                        data = response.json()
                        demo = data["demographics"]
                        metrics = data["metrics"]
                        recommendations = data["cross_sell_recommendations"]
                        
                        # Display Core Bank Demographics from DB
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(label="Current Demographics", value=f"{demo['age']} Yrs Old")
                        with col2:
                            st.metric(label="Registered Annual Income", value=f"${demo['annual_income']:,.2f}")
                        with col3:
                            st.metric(label="SQL Account Balance", value=f"${demo['current_balance']:,.2f}")
                        
                        st.markdown("---")
                        
                        # Splitting screen into Valuation and ML cross-sell
                        left_panel, right_panel = st.columns([1, 2])
                        with left_panel:
                            st.markdown("#### **Forecasted Financial Target**")
                            st.info(f"### **12M CLV: ${metrics['predicted_12m_clv']:,.2f}**")
                            st.caption("Calculated continuous portfolio valuation index driven by customer trajectory model.")
                        
                        with right_panel:
                            st.markdown("#### **AI Cross-Sell Prioritization & SHAP Insights**")
                            if not recommendations:
                                st.warning("No prospective insurance policies cross-sell triggers match this profile.")
                            else:
                                for rec in recommendations:
                                    with st.expander(f"📦 {rec['product']} — Propensity: {rec['probability']*100:.1f}%", expanded=True):
                                        st.progress(rec['probability'])
                                        # Displaying the explainable AI message packed with SHAP attributions
                                        st.success(f"**SHAP Attribution Explainer:** {rec['trigger']}")
                    else:
                        st.error("Error communicating with API layer.")
                except requests.exceptions.ConnectionError:
                    st.error("FastAPI backend offline on port 8000.")
    else:
        st.error("SQL database initialization needed.")

# ==========================================
# TAB 2: LIVE SANDBOX PREDICTOR
# ==========================================
with tab_sandbox:
    st.subheader("Manual Inference Form")
    st.markdown("Simulate cross-sell matching for prospects outside the primary database records by entering raw inputs manually.")
    
    with st.form("sandbox_form"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            age = st.number_input("Age", min_value=18, max_value=90, value=35)
            annual_income = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=65000)
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
            tenure_months = st.number_input("Bank Tenure (Months)", min_value=1, max_value=200, value=24)
        with col_b:
            account_balance = st.number_input("Account Balance ($)", min_value=0, max_value=1000000, value=5000)
            total_salary = st.number_input("Last 6M Salary Credited ($)", min_value=0, max_value=250000, value=32500)
            total_debit = st.number_input("Last 6M Total Expenses ($)", min_value=0, max_value=250000, value=28000)
            avg_tx_size = st.number_input("Average Transaction Size ($)", min_value=1, max_value=10000, value=75)
            tx_count = st.number_input("Total Transaction Count", min_value=1, max_value=1000, value=80)
        with col_c:
            spend_auto = st.number_input("6M Auto Service Spending ($)", min_value=0, max_value=50000, value=200)
            spend_fuel = st.number_input("6M Fuel Spending ($)", min_value=0, max_value=50000, value=400)
            spend_medical = st.number_input("6M Medical Spending ($)", min_value=0, max_value=50000, value=150)
            spend_pharmacy = st.number_input("6M Pharmacy Spending ($)", min_value=0, max_value=50000, value=50)
            spend_estate = st.number_input("6M Real Estate Spending ($)", min_value=0, max_value=100000, value=0)
            spend_groceries = st.number_input("6M Groceries Spending ($)", min_value=0, max_value=50000, value=1200)
            spend_dining = st.number_input("6M Dining Spending ($)", min_value=0, max_value=50000, value=600)
            spend_utilities = st.number_input("6M Utilities Spending ($)", min_value=0, max_value=50000, value=300)

        submit_btn = st.form_submit_button("Execute Sandbox Prediction Run")
        
    if submit_btn:
        payload = {
            "age": age, "annual_income": annual_income, "credit_score": credit_score, "tenure_months": tenure_months,
            "current_account_balance": account_balance, "total_salary_credited": total_salary, "spend_Auto_Service": spend_auto,
            "spend_Dining": spend_dining, "spend_Fuel": spend_fuel, "spend_Groceries": spend_groceries, "spend_Medical": spend_medical,
            "spend_Pharmacy": spend_pharmacy, "spend_Real_Estate": spend_estate, "spend_Utilities": spend_utilities,
            "total_debit_amount": total_debit, "avg_transaction_size": avg_tx_size, "transaction_count": tx_count
        }
        
        with st.spinner("Calculating custom matrix pipeline values..."):
            try:
                res = requests.post(f"{BASE_API_URL}/predict-raw", json=payload)
                if res.status_code == 200:
                    sandbox_data = res.json()
                    st.success("Inference Successfully Compiled!")
                    
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.markdown("#### **Forecasted Sandbox Value**")
                        st.info(f"### Predicted CLV: ${sandbox_data['predicted_clv']:,.2f}")
                    with res_col2:
                        st.markdown("#### **Propensity Vector Map**")
                        probs = sandbox_data["probabilities"]
                        st.write(f"🚗 Auto Propensity: **{probs['Auto']*100:.0f}%**")
                        st.write(f"🏥 Health Propensity: **{probs['Health']*100:.0f}%**")
                        st.write(f"📈 Life Propensity: **{probs['Life']*100:.0f}%**")
                else:
                    st.error("Failed to parse variables via standard API configuration.")
            except Exception as e:
                st.error(f"Sandbox communications breakdown: {e}")

# ==========================================
# TAB X: CHAT BOT (GenAI)
# ==========================================
with tab_chat:
    st.subheader("Chat BOT — GenAI Assistant")
    st.markdown("Ask domain or product questions and get responses powered by a GenAI model (Gemini). Chat history is kept for this session only.")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    def send_to_gemini(question: str) -> str:
        """Use the local Gemini helper to generate a response from the shared GenAI function."""
        return generate_ai_response(question, context=ai_context)

    if ai_context is None:
        st.warning(f"GenAI knowledge context could not be loaded from '{CONTEXT_FILE_PATH}'. The assistant will still reply without context.")

    # Display chat history
    chat_container = st.container()
    with st.form("chat_form", clear_on_submit=True):
        user_question = st.text_input("Your question", key="chat_input")
        send_btn = st.form_submit_button("Send")

    if send_btn and user_question:
        # append user message
        st.session_state.chat_history.append({"role": "user", "text": user_question})
        with st.spinner("Generating GenAI response..."):
            reply = send_to_gemini(user_question)
        st.session_state.chat_history.append({"role": "assistant", "text": reply})

    # Render messages in reverse chronological order (most recent at bottom)
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(f"**You:** {msg['text']}")
            else:
                st.markdown(f"**Bot:** {msg['text']}")
        st.markdown("---")

# ==========================================
# TAB 3: SYSTEM OPERATIONS MONITOR (MLOps)
# ==========================================
with tab_monitor:
    st.subheader("Database Operations & Data Drift Analytics")
    st.markdown("Tracks production metrics, population health aggregates, and target data distributions directly from the SQLite infrastructure.")
    
    try:
        meta_res = requests.get(f"{BASE_API_URL}/meta")
        if meta_res.status_code == 200:
            meta_data = meta_res.json()
            
            # Operational status cards
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric(label="Total Database Row Transactions", value=meta_data["total_records"])
            with m_col2:
                st.metric(label="System System Average Age", value=f"{meta_data['avg_age']:.1f} Yrs")
            with m_col3:
                st.metric(label="Mean Customer Account Assets", value=f"${meta_data['avg_balance']:,.2f}")
            
            st.markdown("---")
            
            st.markdown("#### System Baseline Policy Conversion Distributions")
            rates = meta_data["conversion_rates"]
            
            # Pack conversion rates into charts to give data visualization footprint
            chart_data = pd.DataFrame({
                "Insurance Product": ["Auto Insurance", "Health Insurance", "Life Insurance"],
                "Systemic Conversion Rate (%)": [rates["auto"] * 100, rates["health"] * 100, rates["life"] * 100]
            })
            
            st.bar_chart(data=chart_data, x="Insurance Product", y="Systemic Conversion Rate (%)", use_container_width=True)
            st.caption("This visualization helps identify changes in data distribution over time. Changes in conversion rates point to potential shifts in customer behavior (data drift).")
        else:
            st.error("Unable to gather platform health matrices.")
    except Exception as e:
        st.error(f"Monitoring pipeline connection problem: {e}")