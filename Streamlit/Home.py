import streamlit as st
import base64

st.set_page_config(
    page_title="StockSeer - Home",
    page_icon="👋",
)
st.write("# Welcome to StockSeer! 👋")
st.sidebar.header("Select above menu 👆")
st.markdown(
    """
    StockSeer is an advanced stock price prediction app that leverages cutting-edge 
    AI algorithms to provide accurate market forecasts. We offer real-time data 
    analysis, empowering investors to make informed decisions and maximize your returns.

    <br>
    <div style="text-align: center; font-weight: normal; font-size: medium; color: #133e87;">
    🔮 <i>Your Crystal Ball for Market Success</i> 🔮
    </div>
    <br>

    ### To use StockSeer, simply follow these steps:
    1. Select companies whose stock price you want to predict in the *Prediction Request* page. Available 
    companies to predict: `BBCA`, `BBNI`, `BBRI`, `BMRI`, `MEGA`.
    2. Choose your desired prediction timeline.
    3. Go to the *Prediction Result* page to see the prediction results.
    4. If needed, you can download the historical price and predicted price data in CSV or Excel format.
    <br>
    """,
    unsafe_allow_html=True,
)

disclaimer_text = "This material is provided for informational purposes only and should not be construed as financial advice. The decision to invest remains solely your responsibility."
st.markdown(
    f"""
    <h5 style="text-align:center;color:#133E87;font-weight:bold;"><b><i>{str(disclaimer_text).upper()}</b></i></h5><br>
    """, unsafe_allow_html=True
)

col1, col2= st.columns(2)
with col1:
    switch_dashboard_page = st.button("📊 VIEW DASHBOARD & ANALYSIS")
    if switch_dashboard_page:
        page_file = "./pages/1_📊_Dashboard.py"
        st.switch_page(page_file)
with col2:
    switch_prediction_page = st.button("🗃️ START SEEING NOW")
    if switch_prediction_page:
        page_file = "./pages/2_🗃️_Prediction_Request.py"
        st.switch_page(page_file)