import json
import openai
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pages.utils.HistoricalDataFunctions import *

st.set_page_config(page_title="StockSeer - Historical Data", page_icon="📊")

with open('./assets/credentials.json', 'r') as file:
   credential = json.load(file)

# Sidebar untuk memilih saham
st.sidebar.title("StockSeer")
stock_options = { 
    "BBCA - Bank Central Asia Tbk": "BBCA.JK",
    "BBRI - Bank Rakyat Indonesia Tbk": "BBRI.JK",
    "BMRI - Bank Mandiri Tbk": "BMRI.JK",
    "BBNI - Bank Negara Indonesia (Persero) Tbk":"BBNI.JK",
    "MEGA - Bank Mega Tbk": "MEGA.JK"
}
selected_stock = st.sidebar.selectbox("Select Stock", options=list(stock_options.keys())) # Stock Selection
period_options = {
    "1 Years": "1y",
    "1 Month": "1mo",
    "6 Months": "6mo",
    "2 Years": "2y",
    "5 years": "5y",
    "10 years":"10y",
    "All": "max"
}
selected_period = st.sidebar.selectbox("Select Period", options=list(period_options.keys())) # Period Selection
financials_option = st.sidebar.selectbox("Select Financials", options=["Annual", "Quarter"])


# (Header) Page Title
st.markdown(
    f"""
        <h4 style="text-align:center;color:#9AA6B2;padding-bottom:0px;">
            <i>Showing Historical Data for</i>
        </h4>
        <h2 style="text-align:center;font-weight:bold;color:#133E87;">
            <b>{selected_stock}</b>
        </h2><br>
    """, unsafe_allow_html = True
) 

# Load dan tampilkan data saham
try:
    with st.spinner("Fetching data..."):
        stock_data = get_stock_data(stock_options[selected_stock], period_options[selected_period])
        ### Analysis & Recommendation ###
        response = get_analysis_and_recommendation(stock_options[selected_stock], stock_data, selected_period, df_fundamental=None, df_financial=None)
        summary = get_summary_from_gpt(response)
        recommendation, summarized_analysis = clean_summarized_analysis(summary)
        # Display the summary in a card
        st.markdown(
                f"""
                <div style="
                    border: 1px solid #ddd; 
                    border-radius: 10px; 
                    padding: 15px; 
                    box-shadow: 2px 2px 12px rgba(0,0,0,0.1); 
                    background-color: #f9f9f9;
                    margin: 20px 0;
                    text-align: justify;
                ">
                    <b style="text-align:left;">{recommendation}</b><br>{summarized_analysis}
                </div>
                """, unsafe_allow_html=True
            )
        
        ### batas ###
        
        if stock_data.empty:  # Cek jika data kosong
            st.warning("No data available for the selected period. Try selecting a different period or stock.")
        else:
            # Menampilkan metrik harga terbaru
            latest_price = stock_data['Close'].iloc[-1]
            lowest_price = stock_data['Low'].min()
            highest_price = stock_data['High'].max()
            highest_volume_day = stock_data['Volume'].idxmax()
            lowest_volume_day = stock_data['Volume'].idxmin()

            # Showing Latest, Lowest, Highest Price
            col1, col2, col3 = st.columns(3) # Membagi metrik harga dan data fundamental dalam grid 3x3
            with col1:
                st.markdown(
                    f"""
                        <h3 style="color:#133E87;">
                            <b>Rp {latest_price:,.2f}</b>
                        </h3>
                    """, unsafe_allow_html = True
                )
                st.write("Latest Price")
            with col2:
                st.markdown(
                    f"""
                        <h3 style="color:#9AA6B2;">
                            <b>Rp {lowest_price:,.2f}</b>
                        </h3>
                    """, unsafe_allow_html = True
                )
                st.write("Lowest Price")
            with col3:
                st.markdown(
                    f"""
                        <h3 style="color:#5F84A2;">
                            <b>Rp {highest_price:,.2f}</b>
                        </h3>
                    """, unsafe_allow_html = True
                )
                st.write("Highest Price")
            st.markdown("""<br>""", unsafe_allow_html = True)

            fig = go.Figure() # Historical Price Data (Candlestick)
            fig.add_trace(
                go.Candlestick(
                    x=stock_data.index,
                    open=stock_data['Open'],
                    high=stock_data['High'],
                    low=stock_data['Low'],
                    close=stock_data['Close'],
                    increasing_line_color='#133E87', 
                    decreasing_line_color='#9AA6B2',
                    name='Candlestick'
                )
            )
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Price (Rp)",
                template="plotly_white",
                hovermode="x unified",
                font=dict(family="Arial", size=12),
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Menampilkan data fundamental dalam kotak
            ticker = stock_options[selected_stock]
            fundamental_data = get_fundamental_data(ticker)
            st.markdown(
                """
                    <h4 style="text-align:center;color:#9AA6B2;"><i>Fundamentals</i></h4>
                """, unsafe_allow_html = True
            )
            df_fundamental, metrics = calculate_metrics(ticker, financials_option)
            display_fundamental_data(df_fundamental, metrics)

            # Menampilkan data berdasarkan pilihan di sidebar
            st.markdown(
                """
                    <h4 style="text-align:center;color:#9AA6B2;margin-bottom:15px;"><i>Financials</i></h4>
                """, unsafe_allow_html = True
            )
            if financials_option == "Annual":
                if not fundamental_data["Annual Financials"].empty:
                    df_financial = fundamental_data["Annual Financials"].reset_index()
                    financial_growth = calculate_financial_metrics(df_financial)
                    display_financials_data(df_financial, financial_growth)
                else:
                    st.write("No relevant data available.")
            elif financials_option == "Quarter":
                if not fundamental_data["Quarterly Financials"].empty:
                    df_financial = fundamental_data["Quarterly Financials"].reset_index()
                    financial_growth = calculate_financial_metrics(df_financial)
                    display_financials_data(df_financial, financial_growth)
                else:
                    st.write("No relevant data available.")
            
            # Analysis & Recommendation

            # Analysis & Recommendation
            st.markdown(
                """
                <h4 style="text-align:center;color:#9AA6B2;margin-bottom:15px;"><i>Analysis & Recommendation</i></h4>
                """, unsafe_allow_html=True
            )

            # response = get_analysis_and_recommendation(ticker, stock_data, df_fundamental, df_financial)
            cleaned_prompted_recommendation, cleaned_prompted_analysis = clean_generated_analysis(response)

            # Display the response
            st.markdown(
                f"""
                <div style="
                    border: 1px solid #ddd; 
                    border-radius: 10px; 
                    padding: 20px; 
                    box-shadow: 2px 2px 12px rgba(0,0,0,0.1); 
                    background-color: #fff;
                    margin: 20px 0;
                    text-align:justify;
                ">
                    <b>{cleaned_prompted_recommendation}</b><br><br>{cleaned_prompted_analysis}
                </div>
                """, unsafe_allow_html=True
            )

            st.markdown(
                """
                <h6 style="text-align:center;color:#9AA6B2;margin-bottom:15px;"><i>Analysis Generated by OpenAI's ChatGPT</i></h6>
                """, unsafe_allow_html=True
            )

            disclaimer_text = "This material is provided for informational purposes only and should not be construed as financial advice. The decision to invest remains solely your responsibility."
            st.markdown(
                f"""
                <h5 style="text-align:center;color:#133E87;font-weight:bold;"><b><i>{str(disclaimer_text).upper()}</b></i></h5><br>
                """, unsafe_allow_html=True
            )

            col1, col2= st.columns(2)
            with col1:
                switch_home_page = st.button("🏠 BACK TO HOME")
                if switch_home_page:
                    page_file = "./Home.py"
                    st.switch_page(page_file)
            with col2:
                switch_prediction_page = st.button("🗃️ START SEEING NOW")
                if switch_prediction_page:
                    page_file = "./pages/2_🗃️_Prediction_Request.py"
                    st.switch_page(page_file)

except Exception as e:
    st.error(f"Error fetching data: {e}")