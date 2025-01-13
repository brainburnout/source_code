import streamlit as st, yfinance as yf
from datetime import datetime, timedelta
import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import requests
import os
from scipy.interpolate import interp1d

# Hide the Streamlit header and footer
st.markdown(
    """
    <style>
    /* Hide the Streamlit header */
    header {visibility: hidden;}
    /* Hide the Streamlit footer */
    
    </style>
    """, 
    unsafe_allow_html=True
)

        
st.markdown("# What stock do you want to predict?")
st.sidebar.header("Input stock name and date 👉")
st.write(
    """On this page, you can choose up to 5 stocks and set the timeline for their price forecast. 
    Simply enter the start and end date for the period you'd like to predict, and select companies whose 
    stock price you want to forecast. The predictions will be made on a weekly basis, providing you with 
    clear insights on the stock's potential movement every week throughout the selected period."""
)

# ---------- INPUT FROM USER: Start & End Prediction Dates, Stock Names ----------
today_date = datetime.today()

# Check if there are existing session states for the inputs and use them as defaults
default_start_input = st.session_state.get('start_input', today_date)
default_end_input = st.session_state.get('end_input', today_date + timedelta(days=2*30))
default_stock_input = st.session_state.get('stock_input', [])

# Custom label for company selection
st.markdown("#### 🏢 Company")
# User input for stock names
stock_input = st.multiselect(
    '',  # Empty string for the label
    [ 
     'BBCA - PT. Bank Central Asia Tbk', 
     'BBNI - PT. Bank Negara Indonesia (Persero) Tbk', 
     'BBRI - PT. Bank Rakyat Indonesia (Persero) Tbk', 
     'BMRI - PT. Bank Mandiri (Persero) Tbk',
     'MEGA - PT. Bank Mega Tbk Tbk'],  # List of options
    default=[stock for stock in default_stock_input],  # Set default to previous input
    label_visibility="collapsed"  # Hides the label
)

# Two columns for side-by-side input
col1, col2 = st.columns(2)

# Today's date as default
today_date = datetime.today()

with col1:
    # Custom label and hidden native label
    st.markdown("#### 📅 Start Date")
    start_input = st.date_input(
        "Start Date",  # Invisible label
        value=today_date,
        max_value=today_date,
        label_visibility="collapsed",  # Hides the native label
    )

with col2:
    # Custom label and hidden native label
    st.markdown("#### 📅 End Date")
    end_input = st.date_input(
        "End Date",  # Invisible label
        value=today_date + timedelta(days=7),
        min_value=start_input,  # Ensures end date cannot be earlier than start date
        label_visibility="collapsed",  # Hides the native label
    )

# Add note about the maximum 3-month prediction aligned to the left
st.markdown(
    """
    <p style="font-size: 12px; color: red; text-align: left;">
        *Prediction duration is suggested to not exceed 9 months due to model limits.
    </p>
    """, 
    unsafe_allow_html=True
)

# Single button for both confirmation and navigation
if st.button("📈 PREDICT NOW", use_container_width=True):
    if stock_input:  # Validate stock selection
        # Save selections to session state
        st.session_state['stock_input'] = stock_input
        st.session_state['start_input'] = start_input
        st.session_state['end_input'] = end_input
        
        # Success message
        st.success("Selection confirmed! Redirecting to prediction results...")
        
        # Navigate to prediction results page
        page_file = "./pages/3_📈_Prediction_Result.py"
        st.switch_page(page_file)
    else:
        # Error message if no stock selected
        st.error("You have not selected the stock.")

disclaimer_text = "This material is provided for informational purposes only and should not be construed as financial advice. The decision to invest remains solely your responsibility."
st.markdown(
    f"""
    <br><h5 style="text-align:center;color:#133E87;font-weight:bold;"><b><i>{str(disclaimer_text).upper()}</b></i></h5><br>
    """, unsafe_allow_html=True
)

col1, col2= st.columns(2)
with col1:
    switch_dashboard_page = st.button("📊 VIEW DASHBOARD & ANALYSIS")
    if switch_dashboard_page:
        page_file = "./pages/1_📊_Dashboard.py"
        st.switch_page(page_file)
with col2:
    switch_home_page = st.button("🏠 BACK TO HOME")
    if switch_home_page:
        page_file = "./Home.py"
        st.switch_page(page_file)