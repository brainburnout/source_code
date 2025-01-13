import re
import json
import base64
import pandas as pd
import yfinance as yf
import streamlit as st
from io import BytesIO
from openai import OpenAI
import plotly.graph_objects as go

annual_columns = {
   'EPS' : 'EPS (Annual)',
   'PER' : 'PE Ratio (Annual)',
   'ROE' : 'Return on Equity (Annual)',
   'ROA' : 'Return on Assets (Annual)'
}
quarterly_columns = {
   'EPS' : 'EPS (Quarter)',
   'PER' : 'PE Ratio (Quarter)',
   'ROE' : 'Return on Equity (Quarter)',
   'ROA' : 'Return on Assets (Quarter)'
}

def get_stock_data(ticker, period):
   stock = yf.Ticker(ticker)
   hist = stock.history(period=period, interval="1d")  # Interval harian
   return hist

def get_fundamental_data(ticker):
   stock = yf.Ticker(ticker)
   info = stock.info  # Mengambil data info yang berisi metrik penting
   financials_annual = stock.financials.T  # Mengambil laporan keuangan tahunan
   financials_quarterly = stock.quarterly_financials.T  # Mengambil laporan keuangan triwulanan

   # Filter hanya kolom yang relevan untuk Annual dan Quarterly Financials
   relevant_columns = ["Net Interest Income", "Interest Income", "Net Income"]
   financials_annual = financials_annual[relevant_columns] if any(col in financials_annual.columns for col in relevant_columns) else pd.DataFrame()
   financials_quarterly = financials_quarterly[relevant_columns] if any(col in financials_quarterly.columns for col in relevant_columns) else pd.DataFrame()

   # Ambil metrik penting dari info
   net_margin = info.get("profitMargins", "N/A") * 100 if info.get("profitMargins") else "N/A"
   roa = info.get("returnOnAssets", "N/A") * 100 if info.get("returnOnAssets") else "N/A"
   roe = info.get("returnOnEquity", "N/A") * 100 if info.get("returnOnEquity") else "N/A"

   return {
      "Net Margin": f"{net_margin}%" if net_margin != "N/A" else net_margin,
      "ROA": f"{roa}%" if roa != "N/A" else roa,
      "ROE": f"{roe}%" if roe != "N/A" else roe,
      "Annual Financials": financials_annual,
      "Quarterly Financials": financials_quarterly
   }

def clean_column_names(dates):
   cleaned_dates = []
   for date in dates:
      cleaned_dates.append(date.replace('12M ', ''))
   return cleaned_dates

def get_visualization_data(ticker, metric, period = "Annual"):
   ticker_cleaned = ticker.replace(".JK", "")
   file_path = f"./assets/data/{period}/{ticker_cleaned}.xlsx"
   df = pd.read_excel(file_path, index_col = "In Million")
   metric_column = annual_columns[metric] if period == "Annual" else quarterly_columns[metric]
   if metric == "ROE" or metric == "ROA":
      final_values = []
      for value in df.loc[metric_column].values:
         final_values.append(float(value.replace('%', '')))
      return dict(zip(clean_column_names(df.columns.values), final_values))
   else:
      return dict(zip(clean_column_names(df.columns.values), df.loc[metric_column].values))

def calculate_growth(metric_data):
   iteration = iter(metric_data.values())
   metric_latest = float(next(iteration))
   metric_1y = float(next(iteration))
   next(iteration)
   metric_3y = float(next(iteration))
   metric_growth_1y = (metric_latest - metric_1y)/metric_1y
   metric_growth_3y = (metric_latest - metric_3y)/metric_3y
   return metric_latest, metric_growth_1y, metric_growth_3y

def calculate_metrics(ticker, period = "Annual"):
   eps_data = get_visualization_data(ticker, "EPS", period)
   per_data = get_visualization_data(ticker, "PER", period)
   roa_data = get_visualization_data(ticker, "ROA", period)
   roe_data = get_visualization_data(ticker, "ROE", period)

   eps_latest, eps_change_1y, eps_change_3y = calculate_growth(eps_data)
   per_latest, per_change_1y, per_change_3y = calculate_growth(per_data)
   roa_latest, roa_change_1y, roa_change_3y = calculate_growth(roa_data)
   roe_latest, roe_change_1y, roe_change_3y = calculate_growth(roe_data)

   df_fundamental = pd.DataFrame({
      "Period" : list(eps_data.keys()),
      "EPS" : list(eps_data.values()),
      "PER" : list(per_data.values()),
      "ROA" : list(roa_data.values()),
      "ROE" : list(roe_data.values())
   })

   return df_fundamental, {
      "eps" : {
         "latest": eps_latest, "change_1y": eps_change_1y, "change_3y": eps_change_3y,
      },
      "per" : {
         "latest": per_latest, "change_1y": per_change_1y, "change_3y": per_change_3y,
      },
      "roa": {
         "latest": roa_latest, "change_1y": roa_change_1y, "change_3y": roa_change_3y,
      },
      "roe": {
         "latest": roe_latest, "change_1y": roe_change_1y, "change_3y": roe_change_3y,
      },
   }

def display_fundamental_data(df, metrics):
   st.markdown(
      """
         <style>
            .metric-container {
               font-family: Arial, sans-serif;
               border: 1px solid lightgray;
               padding: 15px;
               border-radius: 10px;
               text-align: center;
               background-color: #f9f9f9;
            }
            .metric-title {
               margin-bottom: 0px;
               padding-bottom: 0px;
               text-align: left;
               padding-left: 15px;
               font-size: 30px;
               font-weight: bold;
               font-family: "Times New Roman";
               color: #133E87;
            }
            .metric-subtitle {
               margin-top: 0px;
               padding-top: 0px;
               font-family: "Times New Roman";
               text-align: left;
               padding-left: 15px;
               font-size: 16px;
               color: #5F84A2;
            }
            .metric-subtitle-2 {
               margin-top: 0px;
               padding-top: 0px;
               margin-bottom: 0px;
               padding-bottom: 0px;
               font-family: "Times New Roman";
               text-align: left;
               padding-left: 15px;
               font-size: 16px;
               color: #9AA6B2;
            }
            .metric-average {
               display: flex;
               justify-content: left;
               margin-top: 10px;
               padding-left: 10px;
               padding-bottom: 10px;
            }
            .metric-average div {
               margin: 0 5px;
               padding: 5px 10px;
               border-radius: 5px;
               font-size: 12px;
               font-weight: bold;
            }
            .metric-average .card {
               border: 1px solid #133E87;
               background-color: #D9EAFD;
               color: #133E87;
            }
         </style>
      """,unsafe_allow_html=True,
   )
   col1, col2 = st.columns(2)
   with col1:
      growth_1y_value_color = "green" if metrics['per']['change_1y'] >= 0 else "red"
      growth_3y_value_color = "green" if metrics['per']['change_3y'] >= 0 else "red"
      
      fig = go.Figure(data=[
         go.Bar(
            x=df['Period'], 
            y=df['EPS'], 
            marker_color="#5F84A2", 
            text=df['EPS'], 
            textposition='outside'
         )
      ]) # Growth Chart
      fig.update_layout(
         autosize = True,
         yaxis_title = "Values (In Million)",
         xaxis = dict(autorange = True),
         yaxis = dict(autorange = True),
         height = 500
      )
      buf = BytesIO()
      fig.write_image(buf, format="png")
      buf.seek(0)
      encoded_chart = base64.b64encode(buf.read()).decode("utf-8")
      buf.close()

      st.markdown(
         f"""
            <div class="metric-container">
               <div class="metric-title">{metrics['eps']['latest']:.1f}M</div>
               <div class="metric-subtitle"><b>EPS</b></div>
               <div>
                  <img src="data:image/png;base64,{encoded_chart}" alt="Bar Chart" style="width: 100%; max-width: 500px; margin-bottom: 10px;">
               </div>
               <div class="metric-subtitle-2"><b>Growth:</b></div>
               <div class="metric-average" style="margin-top:2px;">
                  <div class="card">1Y <span style="color:{growth_1y_value_color};">{metrics['eps']['change_1y']:+.1f}%</span></div>
                  <div class="card">3Y <span style="color:{growth_3y_value_color};">{metrics['eps']['change_3y']:+.1f}%</span></div>
               </div>
            </div>
         """, unsafe_allow_html=True,
      )
   with col2:
      growth_1y_value_color = "green" if metrics['per']['change_1y'] >= 0 else "red"
      growth_3y_value_color = "green" if metrics['per']['change_3y'] >= 0 else "red"

      fig = go.Figure(data=[
         go.Bar(
            x=df['Period'], 
            y=df['PER'], 
            marker_color="#5F84A2", 
            text=df['PER'], 
            textposition='outside'
         )
      ]) # Growth Chart
      fig.update_layout(
         autosize = True,
         yaxis_title = "Values (In Million)",
         xaxis = dict(autorange = True),
         yaxis = dict(autorange = True),
         height = 500
      )
      buf = BytesIO()
      fig.write_image(buf, format="png")
      buf.seek(0)
      encoded_chart = base64.b64encode(buf.read()).decode("utf-8")
      buf.close()

      st.markdown(
         f"""
            <div class="metric-container">
               <div class="metric-title">{metrics['per']['latest']:.2f}M</div>
               <div class="metric-subtitle"><b>PER</b></div>
               <div>
                  <img src="data:image/png;base64,{encoded_chart}" alt="Bar Chart" style="width: 100%; max-width: 500px; margin-bottom: 10px;">
               </div>
               <div class="metric-subtitle-2"><b>Growth:</b></div>
               <div class="metric-average" style="margin-top:2px;">
                  <div class="card">1Y <span style="color:{growth_1y_value_color};">{metrics['per']['change_1y']:+.2f}%</span></div>
                  <div class="card">3Y <span style="color:{growth_3y_value_color};">{metrics['per']['change_3y']:+.2f}%</span></div>
               </div>
            </div>
         """, unsafe_allow_html=True,
      )
   st.markdown("<div></div>", unsafe_allow_html=True)
   col3, col4 = st.columns(2)
   with col3:
      growth_1y_value_color = "green" if metrics['roe']['change_1y'] >= 0 else "red"
      growth_3y_value_color = "green" if metrics['roe']['change_3y'] >= 0 else "red"

      fig = go.Figure(data=[
         go.Bar(
            x=df['Period'], 
            y=df['ROE'], 
            marker_color="#5F84A2", 
            text=df['ROE'], 
            textposition='outside'
         )
      ]) # Growth Chart
      fig.update_layout(
         autosize = True,
         yaxis_title = "Values (In Million)",
         xaxis = dict(autorange = True),
         yaxis = dict(autorange = True),
         height = 500
      )
      buf = BytesIO()
      fig.write_image(buf, format="png")
      buf.seek(0)
      encoded_chart = base64.b64encode(buf.read()).decode("utf-8")
      buf.close()

      st.markdown(
         f"""
            <div class="metric-container">
               <div class="metric-title">{metrics['roe']['latest']:.1f}%</div>
               <div class="metric-subtitle"><b>ROE</b></div>
               <div>
                  <img src="data:image/png;base64,{encoded_chart}" alt="Bar Chart" style="width: 100%; max-width: 500px; margin-bottom: 10px;">
               </div>
               <div class="metric-subtitle-2"><b>Growth:</b></div>
               <div class="metric-average" style="margin-top:2px;">
                  <div class="card">1Y <span style="color:{growth_1y_value_color};">{metrics['roe']['change_1y']:+.1f}%</span></div>
                  <div class="card">3Y <span style="color:{growth_3y_value_color};">{metrics['roe']['change_3y']:+.1f}%</span></div>
               </div>
            </div>
         """, unsafe_allow_html=True,
      )
   with col4:
      growth_1y_value_color = "green" if metrics['roa']['change_1y'] >= 0 else "red"
      growth_3y_value_color = "green" if metrics['roa']['change_3y'] >= 0 else "red"

      fig = go.Figure(data=[
         go.Bar(
            x=df['Period'], 
            y=df['ROA'], 
            marker_color="#5F84A2", 
            text=df['ROA'], 
            textposition='outside'
         )
      ]) # Growth Chart
      fig.update_layout(
         autosize = True,
         yaxis_title = "Values (In Million)",
         xaxis = dict(autorange = True),
         yaxis = dict(autorange = True),
         height = 500
      )
      buf = BytesIO()
      fig.write_image(buf, format="png")
      buf.seek(0)
      encoded_chart = base64.b64encode(buf.read()).decode("utf-8")
      buf.close()

      st.markdown(
         f"""
            <div class="metric-container">
               <div class="metric-title">{metrics['roa']['latest']:.2f}%</div>
               <div class="metric-subtitle"><b>ROA</b></div>
               <div>
                  <img src="data:image/png;base64,{encoded_chart}" alt="Bar Chart" style="width: 100%; max-width: 500px; margin-bottom: 10px;">
               </div>
               <div class="metric-subtitle-2"><b>Growth:</b></div>
               <div class="metric-average" style="margin-top:2px;">
                  <div class="card">1Y <span style="color:{growth_1y_value_color};">{metrics['roa']['change_1y']:+.2f}%</span></div>
                  <div class="card">3Y <span style="color:{growth_3y_value_color};">{metrics['roa']['change_3y']:+.2f}%</span></div>
               </div>
            </div><br>
         """, unsafe_allow_html=True,
      )
   return

def calculate_financial_growth(metric_data):
   index = 0
   metric_latest = metric_data.iloc[index]
   metric_1y = metric_data.iloc[index + 1]
   metric_3y = metric_data.iloc[index + 3]
   metric_growth_1y = (metric_latest - metric_1y)/metric_1y
   metric_growth_3y = (metric_latest - metric_3y)/metric_3y
   return metric_latest, metric_growth_1y, metric_growth_3y

def calculate_financial_metrics(financial_data):
   nim_latest, nim_change_1y, nim_change_3y = calculate_financial_growth(financial_data['Net Interest Income'])
   interest_latest, interest_change_1y, interest_change_3y = calculate_financial_growth(financial_data['Interest Income'])
   net_latest, net_change_1y, net_change_3y = calculate_financial_growth(financial_data['Net Income'])
   return {
      "net_interest" : {
         "latest": nim_latest, "change_1y": nim_change_1y, "change_3y": nim_change_3y,
      },
      "interest" : {
         "latest": interest_latest, "change_1y": interest_change_1y, "change_3y": interest_change_3y,
      },
      "net_income": {
         "latest": net_latest, "change_1y": net_change_1y, "change_3y": net_change_3y,
      }
   }

def display_financials_data(df, metrics):
   st.markdown(
      """
         <style>
            .metric-container {
               font-family: Arial, sans-serif;
               border: 1px solid lightgray;
               padding: 15px;
               border-radius: 10px;
               text-align: center;
               background-color: #f9f9f9;
            }
            .metric-title {
               margin-bottom: 0px;
               padding-bottom: 0px;
               text-align: left;
               padding-left: 15px;
               font-size: 30px;
               font-weight: bold;
               font-family: "Times New Roman";
               color: #133E87;
            }
            .metric-subtitle {
               margin-top: 0px;
               padding-top: 0px;
               font-family: "Times New Roman";
               text-align: left;
               padding-left: 15px;
               font-size: 14px;
               color: #5F84A2;
            }
            .metric-subtitle-2 {
               margin-top: 0px;
               padding-top: 0px;
               margin-bottom: 0px;
               padding-bottom: 0px;
               font-family: "Times New Roman";
               text-align: left;
               padding-left: 15px;
               font-size: 13px;
               color: #9AA6B2;
            }
            .metric-average {
               display: flex;
               justify-content: left;
               margin-top: 10px;
               padding-left: 10px;
               padding-bottom: 10px;
            }
            .metric-average div {
               margin: 0 5px;
               padding: 5px 10px;
               border-radius: 5px;
               font-size: 12px;
               font-weight: bold;
            }
            .metric-average .card {
               border: 1px solid #133E87;
               background-color: #D9EAFD;
               color: #133E87;
            }
         </style>
      """,unsafe_allow_html=True,
   )
   col1, col2, col3 = st.columns(3)
   with col1:
      growth_1y_value_color = "green" if metrics['net_interest']['change_1y'] >= 0 else "red"
      growth_3y_value_color = "green" if metrics['net_interest']['change_3y'] >= 0 else "red"
      
      fig = go.Figure(data=[
         go.Bar(
            x=df['index'], 
            y=df['Net Interest Income'], 
            marker_color="#5F84A2", 
            text=df['Net Interest Income'], 
            textposition='outside'
         )
      ]) # Growth Chart
      fig.update_layout(
         plot_bgcolor = "#f9f9f9",
         paper_bgcolor = "#f9f9f9",
         autosize = True,
         yaxis_title = "Values (T)",
         xaxis = dict(autorange = True),
         yaxis = dict(autorange = True),
         height = 500
      )
      buf = BytesIO()
      fig.write_image(buf, format="png")
      buf.seek(0)
      encoded_chart = base64.b64encode(buf.read()).decode("utf-8")
      buf.close()

      st.markdown(
         f"""
            <div class="metric-container">
               <div class="metric-title">{(metrics['net_interest']['latest']/1e12):.2f}T</div>
               <div class="metric-subtitle"><b>NET INTEREST INCOME</b></div>
               <div>
                  <img src="data:image/png;base64,{encoded_chart}" alt="Bar Chart" style="width: 100%; max-width: 500px; margin-bottom: 10px;">
               </div>
               <div class="metric-subtitle-2"><b>Growth:</b></div>
               <div class="metric-average" style="margin-top:2px;">
                  <div class="card">1Y <span style="color:{growth_1y_value_color};">{metrics['net_interest']['change_1y']:+.1f}%</span></div>
                  <div class="card">3Y <span style="color:{growth_3y_value_color};">{metrics['net_interest']['change_3y']:+.1f}%</span></div>
               </div>
            </div>
         """, unsafe_allow_html=True,
      )
   with col2:
      growth_1y_value_color = "green" if metrics['interest']['change_1y'] >= 0 else "red"
      growth_3y_value_color = "green" if metrics['interest']['change_3y'] >= 0 else "red"

      fig = go.Figure(data=[
         go.Bar(
            x=df['index'], 
            y=df['Interest Income'], 
            marker_color="#5F84A2", 
            text=df['Interest Income'], 
            textposition='outside'
         )
      ]) # Growth Chart
      fig.update_layout(
         autosize = True,
         yaxis_title = "Values (T)",
         xaxis = dict(autorange = True),
         yaxis = dict(autorange = True),
         height = 500
      )
      buf = BytesIO()
      fig.write_image(buf, format="png")
      buf.seek(0)
      encoded_chart = base64.b64encode(buf.read()).decode("utf-8")
      buf.close()

      st.markdown(
         f"""
            <div class="metric-container">
               <div class="metric-title">{(metrics['interest']['latest']/1e12):.2f}T</div>
               <div class="metric-subtitle"><b>INTEREST INCOME</b></div>
               <div>
                  <img src="data:image/png;base64,{encoded_chart}" alt="Bar Chart" style="width: 100%; max-width: 500px; margin-bottom: 10px;">
               </div>
               <div class="metric-subtitle-2"><b>Growth:</b></div>
               <div class="metric-average" style="margin-top:2px;">
                  <div class="card">1Y <span style="color:{growth_1y_value_color};">{metrics['interest']['change_1y']:+.2f}%</span></div>
                  <div class="card">3Y <span style="color:{growth_3y_value_color};">{metrics['interest']['change_3y']:+.2f}%</span></div>
               </div>
            </div>
         """, unsafe_allow_html=True,
      )
   with col3:
      growth_1y_value_color = "green" if metrics['net_income']['change_1y'] >= 0 else "red"
      growth_3y_value_color = "green" if metrics['net_income']['change_3y'] >= 0 else "red"

      fig = go.Figure(data=[
         go.Bar(
            x=df['index'], 
            y=df['Net Income'], 
            marker_color="#5F84A2", 
            text=df['Net Income'], 
            textposition='outside'
         )
      ]) # Growth Chart
      fig.update_layout(
         autosize = True,
         yaxis_title = "Values (T)",
         xaxis = dict(autorange = True),
         yaxis = dict(autorange = True),
         height = 500
      )
      buf = BytesIO()
      fig.write_image(buf, format="png")
      buf.seek(0)
      encoded_chart = base64.b64encode(buf.read()).decode("utf-8")
      buf.close()

      st.markdown(
         f"""
            <div class="metric-container">
               <div class="metric-title">{(metrics['net_income']['latest']/1e12):.2f}T</div>
               <div class="metric-subtitle"><b>NET INCOME</b></div>
               <div>
                  <img src="data:image/png;base64,{encoded_chart}" alt="Bar Chart" style="width: 100%; max-width: 500px; margin-bottom: 10px;">
               </div>
               <div class="metric-subtitle-2"><b>Growth:</b></div>
               <div class="metric-average" style="margin-top:2px;">
                  <div class="card">1Y <span style="color:{growth_1y_value_color};">{metrics['net_income']['change_1y']:+.1f}%</span></div>
                  <div class="card">3Y <span style="color:{growth_3y_value_color};">{metrics['net_income']['change_3y']:+.1f}%</span></div>
               </div>
            </div>
         """, unsafe_allow_html=True,
      )
      st.markdown("""<br>""", unsafe_allow_html = True)
   return

def get_analysis_and_recommendation(ticker, stock_data, selected_period, df_fundamental=None, df_financial=None):
   with open('./assets/credentials.json', 'r') as file:
      credential = json.load(file)
   prompt = f"Make a concise and precise analysis and recommendation this given data, in a form of 2 paragraph. Make a concise and precise analysis and recommendation this given data, in a form of 2 paragraph. {stock_data} is a dataframe that consist of the historical daily price of {ticker} company over the period of {selected_period}. {df_financial} is the company's financial data, and {df_fundamental} is the company's fundamental data. From that given data, the analysis must consist of the trend, and seasonality that the historical daily price data has, and analysis the growth of the fundamentals too. Correlate the recommendation to the analysis. The result must strictly be in 2 paragraph, and in the beginning of analysis please give the recommendations to buy or sell or hold, recommendation text should strictly be: Recommendation to Buy or Recommendation to Hold or Recommendation to Sell, the text must be on bold, then followed by an enter. Overall, the analysis must strictly be in 2 paragraphs."
   client = OpenAI(api_key = credential['API_KEY'])
   response = client.chat.completions.create(
      model = "gpt-4o",
      messages = [
         {"role": "user", "content": prompt},
      ],
      max_tokens = 500,
      temperature = 0.5,
   )
   result_content = response.choices[0].message.content
   result_parts = result_content.split('**')
   for i in range(1, len(result_parts), 2):
       result_parts[i] = f"<b>{result_parts[i]}</b>"
   result_content = ''.join(result_parts)
   return result_content

def get_summary_from_gpt(response):
   with open('./assets/credentials.json', 'r') as file:
      credential = json.load(file)
    
   # Prompt to generate a two-sentence summary
   prompt = f"""
   Summarize the following analysis in exactly two sentences, including a clear recommendation (Buy, Hold, or Sell) at the beginning of the summary: {response} 
   """
    
   client = OpenAI(api_key=credential['API_KEY'])
   response_summary = client.chat.completions.create(
      model="gpt-4o",
      messages=[
         {"role": "user", "content": prompt},
      ],
      max_tokens=100,
      temperature=0.5,
   )
   summary_content = response_summary.choices[0].message.content
   return summary_content

def result_analysis(pred_result_str, stock_input, df_fundamental=None, df_financial=None):
   with open('./assets/credentials.json', 'r') as file:
      credential = json.load(file)
   prompt_2 = f"Make a concise and precise analysis and recommendation based on the prediction result {pred_result_str} in a single paragraph. {stock_input} company, while {df_financial} includes the company's financial data, and {df_fundamental} presents the company's fundamental data. The analysis should address the trends and seasonality in the historical daily price data, as well as evaluate the growth of the company's fundamentals. Correlate the recommendation to the analysis."
   client_2 = OpenAI(api_key = credential['API_KEY'])
   response_2 = client_2.chat.completions.create(
      model = "gpt-4o",
      messages = [
         {"role": "user", "content": prompt_2},
      ],
      max_tokens = 500,
      temperature = 0.5,
   )
   result_content = response_2.choices[0].message.content
     # Replace '**' with alternating <b> and </b> tags
   result_parts = result_content.split('**')
   for i in range(1, len(result_parts), 2):
       result_parts[i] = f"<b>{result_parts[i]}</b>"
   result_content = ''.join(result_parts)
   return result_content

def clean_summarized_analysis(text):
   full_text = str(text).upper()
   recommendation = re.findall(r'(RECOMMENDATION TO (HOLD)?(SELL)?(BUY)?:\s)', full_text)
   cleaned_recommendation = recommendation[0][0].replace(':', '')
   cleaned_recommendation = cleaned_recommendation.strip()
   analysis = re.sub(r'(Recommendation to (Hold)?(Buy)?(Sell)?:\s)', ' ', str(text))
   cleaned_analysis = analysis.strip()
   return cleaned_recommendation, cleaned_analysis

def clean_generated_analysis(text):
   full_text = str(text).upper()
   recommendation = re.findall(r'(RECOMMENDATION TO (HOLD)?(SELL)?(BUY)?)', full_text)
   cleaned_recommendation = recommendation[0][0].strip()
   analysis = re.sub(r'(Recommendation to (Hold)?(Buy)?(Sell)?)', ' ', str(text))
   cleaned_analysis = analysis.strip()
   return cleaned_recommendation, cleaned_analysis