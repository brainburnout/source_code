import streamlit as st
import os
import io
import torch
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from pandas import Timestamp, Timedelta
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TCNModel
from pages.utils.HistoricalDataFunctions import result_analysis


st.set_page_config(page_title="StockSeer - Prediction Result", page_icon="📈")

# Define the MAPE values for each stock
mape_values = {
    'BBCA': 1.79,
    'BBNI': 5.58,
    'BMRI': 3.75,
    'BBRI': 5.77,
    'MEGA': 5.66
}

stock_input = st.session_state.get('stock_input', [])
stock_input = [stock.split(' - ')[0] for stock in stock_input]

start_input = st.session_state.get('start_input', None)
end_input = st.session_state.get('end_input', None)


st.markdown("# Prediction Result")
st.sidebar.header("See weekly price prediction 👉")
st.write(
    """On this page, you can explore the weekly predicted future stock price for the selected Bank company. 
    Using advanced Temporal Convolutional Network (TCN) algorithms, the stock's potential movement over the upcoming weeks 
    is forecasted, providing valuable insights to support your investment decisions."""
)

# Calculate the average MAPE
average_mape = sum(mape_values.values()) / len(mape_values)

# Display MAPE for each selected stock in a smaller card format
with st.container():
    st.markdown("""
    <div style="display: flex; flex-wrap: wrap; justify-content: space-between;">
    """, unsafe_allow_html=True)

    for stock in stock_input:
        if stock in mape_values:
            st.markdown(f"""
            <div style="flex: 1 1 calc(45% - 20px); margin: 10px; border: 1px solid #ccc; padding: 8px; border-radius: 10px; box-sizing: border-box;">
                <h1 style="font-size: 24px; text-align: center;">{mape_values[stock]}%</h1>
                <p style="text-align: center;">Model accuracy for {stock}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="flex: 1 1 calc(45% - 20px); margin: 10px; border: 1px solid #ccc; padding: 8px; border-radius: 10px; box-sizing: border-box;">
                <h1 style="font-size: 24px; text-align: center;">N/A</h1>
                <p style="text-align: center;">Akurasi model untuk {stock} tidak tersedia.</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="flex: 1 1 calc(45% - 20px); margin: 10px; border: 1px solid #ccc; padding: 8px; border-radius: 10px; box-sizing: border-box;">
        <h1 style="font-size: 24px; text-align: center;">{average_mape:.2f}%</h1>
        <p style="text-align: center;">Overall average MAPE</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    </div>
    """, unsafe_allow_html=True)

    

if len(stock_input)==0:
  st.error("No stocks selected. Go back to the Prediction Request page and make a selection.")
else:
  days_key = {'Monday' : 0, 'Tuesday' : 1, 'Wednesday' : 2, 'Thursday' : 3, 'Friday' : 4, 'Saturday' : 5, 'Sunday' : 6}
  
  # Get the day of the week
  date_obj = datetime.strptime(str(start_input), '%Y-%m-%d')
  day_of_week = date_obj.strftime('%A')
  certain_day = 'W-' + day_of_week[:3]

# ---------- PREPARE OTHER VARIABLES BASED ON INPUT ----------

  def previous_day_of_the_week(date): # Fungsi untuk menggeser tanggal
      days_to_that_day = (date.weekday() - days_key[day_of_week]) % 7
      return date - pd.to_timedelta(days_to_that_day, unit='D')

  # Function to subtract date into 11 years prior
  def oldest_fund_date(start):
      new_date = start - relativedelta(years=11)
      new_year = new_date.year
      if new_year < 2009:
          return 2009
      return new_year

  # Define start and end date for prediction data
  start_pred = pd.to_datetime(start_input)
  end_pred = previous_day_of_the_week(end_input)

  # Define the historical start and end date
  start_calc = start_input - relativedelta(years=4)
  end_calc = start_input - timedelta(weeks=1)
  start = start_calc.strftime('%Y-%m-%d')
  end = end_calc.strftime('%Y-%m-%d')

  # Splitting Point
  split = 1 + (end_pred - start_input).days // 7
  temp = pd.DataFrame([start, end], columns=['Date'])
  temp['Date'] = pd.to_datetime(temp['Date'])
  temp['Date'] = temp['Date'].apply(lambda x: previous_day_of_the_week(x))
  start_date = temp['Date'][0]
  end_date = temp['Date'][1] + Timedelta(days=1)

  # ---------- DEFINE NECESSARY FUNCTIONS FOR DATA PREPARATION ----------

  def load_yf_data(file_name):
      # Load data from yahoo finance
      df_price = yf.download(file_name, start=start, end=pd.to_datetime(end) + Timedelta(days=1), interval="1d") # Download daily data
      df_price.index = pd.to_datetime(df_price.index).tz_localize(None)
      df_price.columns = df_price.columns.droplevel(1)
    
      # Make sure all of that certain day (ex: Monday) exist dates within the range
      all_certain_day = pd.date_range(start=start, end=end, freq=certain_day)
      new_index = df_price.index.union(all_certain_day)
      df_price = df_price.reindex(new_index)
      df_price = df_price.ffill()

      # Set into weekly data
      target_weekday = days_key[day_of_week]
      df_price = df_price[df_price.index.weekday == target_weekday] # Align data to start on the desired weekday
      weekly_data = df_price.resample(certain_day).mean() # Resample data to weekly based on the target weekday
      return weekly_data

  def interpolate_fund_data(df_fund_weekly):
      """Resample the data to a weekly frequency using interpolation."""
      df_fund_weekly = df_fund_weekly.interpolate(method='polynomial', order=2)
      df_fund_weekly = df_fund_weekly.dropna(how='all')
      df_fund_weekly = df_fund_weekly[(df_fund_weekly.index >= start) & (df_fund_weekly.index <= start_pred)]
      return df_fund_weekly

  def convert_to_weekly(df_fund):
      """Convert financial data to a weekly frequency."""
      df_fund_weekly = df_fund.resample('W').asfreq()
      df_fund_weekly = df_fund_weekly.reset_index()
      df_fund_weekly['Date'] = df_fund_weekly['Date'].apply(lambda x: previous_day_of_the_week(x))
      df_fund_weekly = df_fund_weekly.set_index('Date')
      df_fund_weekly = df_fund_weekly.combine_first(df_fund)
      return interpolate_fund_data(df_fund_weekly)

  def load_data(file_link):
      """Load financial data from a CSV file."""
      df_fund = pd.read_csv(file_link, header=None).T
      df_fund.columns = df_fund.iloc[0]  # Set first row as header
      df_fund = df_fund.drop(0).reset_index(drop=True)

      columns_rename = ['Date', 'EPS', 'PER', 'PTS', 'ROA%', 'ROE%', 'PBV', 'DER', 'FAT']
      for i in range(len(df_fund.columns)):
          df_fund = df_fund.rename(columns={df_fund.columns[i]: columns_rename[i]})

      df_fund['Date'] = pd.to_datetime(df_fund['Date'], format='%Y') + pd.offsets.YearEnd(0)
      df_fund['Date'] = df_fund['Date'].apply(lambda x: previous_day_of_the_week(x))
      df_fund = df_fund[(df_fund['Date'].dt.year >= oldest_fund_date(start_calc)) & (df_fund['Date'] <= start_pred)]

      df_fund['ROA%'] = df_fund['ROA%'].str.replace('%', '')
      df_fund['ROE%'] = df_fund['ROE%'].str.replace('%', '')
      df_fund[['EPS', 'PER', 'PTS', 'ROA%', 'ROE%', 'PBV', 'DER', 'FAT']] = df_fund[['EPS', 'PER', 'PTS', 'ROA%', 'ROE%', 'PBV', 'DER', 'FAT']].astype(float)
      df_fund.set_index('Date', inplace=True)
      return convert_to_weekly(df_fund)

  # Combine with Stock Price Data
  def merge_data(df_price, df_fund):
      df_combined = df_fund.join(df_price, on='Date', how='right')
      df_combined = df_combined[['EPS', 'PER', 'PTS', 'ROA%', 'ROE%', 'PBV', 'DER', 'FAT', 'Close']]
      df_combined = df_combined.reset_index().rename(columns={'index':'Date'})
      return df_combined

  # Function to download file from GitHub
  def download_file_from_github(url, file_path):
      os.makedirs(os.path.dirname(file_path), exist_ok=True) # Ensure the directory exists
      response = requests.get(url) # Download the CSV file from GitHub
      with open(file_path, 'wb') as file: # Write the content to the specified file path
          file.write(response.content)


  # ---------- LOAD HISTORICAL DATASETS ----------

  df_price_bbca = load_yf_data('BBCA.JK')
  df_price_bbni = load_yf_data('BBNI.JK')
  df_price_bmri = load_yf_data('BMRI.JK')
  df_price_bbri = load_yf_data('BBRI.JK')
  df_price_mega = load_yf_data('MEGA.JK')

  # Load and Interpolate fundamental data
  df_fund_bbca = load_data('https://raw.githubusercontent.com/brainburnout/skripsi_data/refs/heads/main/bbca.csv')
  df_fund_bbni = load_data('https://raw.githubusercontent.com/brainburnout/skripsi_data/refs/heads/main/bbni.csv')
  df_fund_bmri = load_data('https://raw.githubusercontent.com/brainburnout/skripsi_data/refs/heads/main/bmri.csv')
  df_fund_bbri = load_data('https://raw.githubusercontent.com/brainburnout/skripsi_data/refs/heads/main/bbri.csv')
  df_fund_mega = load_data('https://raw.githubusercontent.com/brainburnout/skripsi_data/refs/heads/main/mega.csv')

  def date_check(df_price):
    date_diff = (start_pred - df_price.index[-1]).days
    if date_diff > 7:
        last_date = df_price.index[-1]
        next_date = last_date + pd.Timedelta(weeks=1)
        # Create a new row with the specified values
        new_row = pd.DataFrame({
            'Adj Close': df_price['Adj Close'].iloc[-1],
            'Close': df_price['Close'].iloc[-1],
            'High': df_price['High'].iloc[-1],
            'Low': df_price['Low'].iloc[-1],
            'Open': df_price['Open'].iloc[-1],
            'Volume': df_price['Volume'].iloc[-1]
        }, index=[next_date])
        df_price = pd.concat([df_price, new_row]) # Append the new row to the existing DataFrame
        df_price = df_price.rename(columns={'index':'Date'})
    return df_price

  # Merge fundamental data and price data
  df_train_val_bbca = merge_data(date_check(df_price_bbca), df_fund_bbca)
  df_train_val_bbni = merge_data(date_check(df_price_bbni), df_fund_bbni)
  df_train_val_bmri = merge_data(date_check(df_price_bmri), df_fund_bmri)
  df_train_val_bbri = merge_data(date_check(df_price_bbri), df_fund_bbri)
  df_train_val_mega = merge_data(date_check(df_price_mega), df_fund_mega)


  # ---------- DO EXTRAPOLATION ----------

  def extrapolate_dataframe(df_train_val, df_price):
      df = df_train_val[~(df_train_val['EPS'].isna())].copy()
      extrapolate_start_date = df['Date'].iloc[-1] + timedelta(weeks=1)
      df.set_index('Date', inplace=True)
      periods_to_extrapolate = int((Timestamp(end_pred) - extrapolate_start_date).days / 7) + 1
      # Initialize a new DataFrame for future values
      future_dates = pd.date_range(extrapolate_start_date, periods=periods_to_extrapolate, freq=certain_day)
      future_df = pd.DataFrame(index=future_dates)
      columns_to_extrapolate = ['EPS', 'PER', 'PTS', 'ROA%', 'ROE%', 'PBV', 'DER', 'FAT']
      # Apply interpolation and extrapolation for each column using interp1d
      for col in columns_to_extrapolate:
          # Get the x (time) and y (values) for fitting
          x = np.arange(len(df))
          y = df[col].values
          # Create an interpolation function with the option to extrapolate
          interp_func = interp1d(x, y, kind='quadratic', fill_value="extrapolate")
          # Generate x values for future dates
          future_x = np.arange(len(df), len(df) + periods_to_extrapolate)
          # Extrapolate future values using the interpolation function
          future_df[col] = interp_func(future_x)
      df_extended = future_df.reset_index().rename(columns={'index': 'Date'})
      # Create the final DataFrame by combining the original and the future data
      df_train_val_subset = df_train_val.iloc[periods_to_extrapolate:].reset_index(drop=True)
      df_fut = pd.concat([df_train_val_subset, df_extended]).reset_index(drop=True)
      df_fut = df_fut.set_index(df_fut.columns[0])
      df_fut['Close'] = df_fut['Close'].fillna(df_price['Close'])
      df_fut = df_fut.reset_index()
      return df_fut

  df_fut_bbca = extrapolate_dataframe(df_train_val_bbca, df_price_bbca)
  df_fut_bbni = extrapolate_dataframe(df_train_val_bbni, df_price_bbni)
  df_fut_bmri = extrapolate_dataframe(df_train_val_bmri, df_price_bmri)
  df_fut_bbri = extrapolate_dataframe(df_train_val_bbri, df_price_bbri)
  df_fut_mega = extrapolate_dataframe(df_train_val_mega, df_price_mega)

  # ---------- CONVERT VARIABLES TO DARTS DATATYPE ----------

  # for reproducibility
  torch.manual_seed(1)
  np.random.seed(1)

  stock_names = ['BBCA', 'BBNI', 'BMRI', 'BBRI','MEGA']
  fund_columns_c0 = ['PTS', 'EPS', 'DER', 'ROE%']
  fund_columns_c1 = ['PER', 'FAT', 'PTS', 'PBV']

  def data_preprocessing(df, columns):
      # Convert all variables to TimeSeries format
      price = TimeSeries.from_values(df['Close'])
      time_series_data = {col: TimeSeries.from_values(df[col]) for col in columns}
      # Stack the fundamental data to obtain series of 2 dimensions
      covariates = time_series_data[columns[0]] # Start with first column (PTS or PER) as the base
      for col in columns[1:]:  # Skip the first column since it's already included
          covariates = covariates.stack(time_series_data[col])
      # Split the covariates into training and validation sets
      train_covariates, val_covariates = covariates[:-split], covariates[-split:]
      # Scale the covariates between 0 and 1
      scaler_covariates = Scaler()
      train_covariates = scaler_covariates.fit_transform(train_covariates)
      val_covariates = scaler_covariates.transform(val_covariates)
      # Concatenate the scaled results for model input
      covariates = concatenate([train_covariates, val_covariates])
      # Scale the price data
      scaler_price = Scaler()
      price_scaled = scaler_price.fit_transform(price)
      train_df, val_df = price_scaled[:-split], price_scaled[-split:]
      return train_df, val_df, covariates, scaler_price, price_scaled

  train_fut_bbca, val_fut_bbca, covariates_fut_bbca, scaler_fut_bbca, price_fut_bbca = data_preprocessing(df_fut_bbca, fund_columns_c0)
  train_fut_bbni, val_fut_bbni, covariates_fut_bbni, scaler_fut_bbni, price_fut_bbni = data_preprocessing(df_fut_bbni, fund_columns_c0)
  train_fut_bmri, val_fut_bmri, covariates_fut_bmri, scaler_fut_bmri, price_fut_bmri = data_preprocessing(df_fut_bmri, fund_columns_c0)
  train_fut_bbri, val_fut_bbri, covariates_fut_bbri, scaler_fut_bbri, price_fut_bbri = data_preprocessing(df_fut_bbri, fund_columns_c1)
  train_fut_mega, val_fut_mega, covariates_fut_mega, scaler_fut_mega, price_fut_mega = data_preprocessing(df_fut_mega, fund_columns_c1)

  train_fut_lists = [train_fut_bbca, train_fut_bbni, train_fut_bmri, train_fut_bbri, train_fut_mega]
  val_fut_lists = [val_fut_bbca, val_fut_bbni, val_fut_bmri, val_fut_bbri, val_fut_mega]
  covariates_fut_lists = [covariates_fut_bbca, covariates_fut_bbni, covariates_fut_bmri, covariates_fut_bbri, covariates_fut_mega]
  scaler_fut_lists = [scaler_fut_bbca, scaler_fut_bbni, scaler_fut_bmri, scaler_fut_bbri, scaler_fut_mega]
  price_fut_lists = [price_fut_bbca, price_fut_bbni, price_fut_bmri, price_fut_bbri, price_fut_mega]
  df_fut_lists = [df_fut_bbca, df_fut_bbni, df_fut_bmri, df_fut_bbri, df_fut_mega]


  # ---------- MODEL PREDICTION ----------

  # URLs of the model and model checkpoint files on GitHub
  model_url_c0 = 'https://github.com/brainburnout/skripsi_data/raw/refs/heads/main/TCN-Model-c0.pth'
  ckpt_url_c0 = 'https://github.com/brainburnout/skripsi_data/raw/refs/heads/main/TCN-Model-c0.pth.ckpt'
  model_url_c1 = 'https://github.com/brainburnout/skripsi_data/raw/refs/heads/main/TCN-Model-c1.pth'
  ckpt_url_c1 = 'https://github.com/brainburnout/skripsi_data/raw/refs/heads/main/TCN-Model-c1.pth.ckpt'

  # Local file names to save the downloaded models
  model_filename_c0 = './model/TCN-Model-c0.pth'
  ckpt_filename_c0 = './model/TCN-Model-c0.pth.ckpt'
  model_filename_c1 = './model/TCN-Model-c1.pth'
  ckpt_filename_c1 = './model/TCN-Model-c1.pth.ckpt'

  # Download the model and model checkpoint files for BBCA, BBNI, BMRI
  download_file_from_github(model_url_c0, model_filename_c0)
  download_file_from_github(ckpt_url_c0, ckpt_filename_c0)
  tcn_model_c0 = TCNModel.load(model_filename_c0) # Load the .pth file

  # Download the model and model checkpoint files for BBRI and MEGA
  download_file_from_github(model_url_c1, model_filename_c1)
  download_file_from_github(ckpt_url_c1, ckpt_filename_c1)
  tcn_model_c1 = TCNModel.load(model_filename_c1) # Load the .pth file

  future_tcn_predictions = {}

  # Make future predictions
  for train, scaler, stock, covariate, df in zip(train_fut_lists, scaler_fut_lists, stock_names, covariates_fut_lists, df_fut_lists):
      if stock == 'BBCA' or stock == 'BBNI' or stock == 'BMRI':
          future_predictions = tcn_model_c0.predict(n=split, series=train, past_covariates=covariate)
      elif stock == 'BBRI' or stock == 'MEGA':
          future_predictions = tcn_model_c1.predict(n=split, series=train, past_covariates=covariate)
      future_predictions_inverse = scaler.inverse_transform(future_predictions)
      future_dates = pd.date_range(start=start_pred, periods=split, freq=certain_day) # Prepare dates for future predictions
      future_df = pd.DataFrame({
          'Date': future_dates,
          'TCN': future_predictions_inverse.values().flatten()
      })
      future_tcn_predictions[f'future_tcn_{stock.lower()}'] = future_df


  # ---------- PLOT PREDICTION ----------
  def plot_stock_predictions(stock_name):
      # Mapping for stock-specific data
      stock_data = {
          "BBCA": (df_fut_lists[0], future_tcn_predictions['future_tcn_bbca']),
          "BBNI": (df_fut_lists[1], future_tcn_predictions['future_tcn_bbni']),
          "BMRI": (df_fut_lists[2], future_tcn_predictions['future_tcn_bmri']),
          "BBRI": (df_fut_lists[3], future_tcn_predictions['future_tcn_bbri']),
          "MEGA": (df_fut_lists[4], future_tcn_predictions['future_tcn_mega'])
      }

      # Map stock name to company full name
      stock_company_map = {
          'BBCA': 'BBCA - PT. Bank Central Asia Tbk',
          'BBNI': 'BBNI - PT. Bank Negara Indonesia (Persero) Tbk',
          'BMRI': 'BMRI - PT. Bank Mandiri (Persero) Tbk',
          'BBRI': 'BBRI - PT. Bank Rakyat Indonesia (Persero) Tbk',
          'MEGA': 'MEGA - PT. Bank Mega Tbk'
        }

      # Select the data based on the stock name
      if stock_name not in stock_data:
          st.error("Invalid stock name. Choose from 'BBCA', 'BBNI', 'BMRI', 'BBRI', or 'MEGA'.")
          return
      df_temp, future_predictions = stock_data[stock_name]

      # Copy and set up the DataFrame
      df = df_temp.copy()
      df.set_index('Date', inplace=True)
      future_predictions.set_index('Date', inplace=True)

      # Change today's prediction price into latest actual price
      ticker_map = {
          'BBCA': 'BBCA.JK',
          'BBNI': 'BBNI.JK',
          'BMRI': 'BMRI.JK',
          'BBRI': 'BBRI.JK',
          'MEGA': 'MEGA.JK'
      }
      data = yf.Ticker(ticker_map[stock_name]).history(period="1d", interval="1d") # Retrieve today's data
      future_predictions.loc[future_predictions.index[0], 'TCN'] = data['Close'].iloc[-1]

      # Fill NaN values in df['Close'] with predictions
      df.loc[df.index.isin(future_predictions.index), 'Close'] = future_predictions['TCN']
      df = df[~df.index.duplicated(keep='first')]
      df.reset_index(inplace=True)

      # Filter to show only the last year of data (from 1 year before until now)
      start_date_last_year = start_pred - timedelta(weeks=52)
      df_last_year = df[df['Date'] >= start_date_last_year]

      # Plot the last year of data using Streamlit
      fig = go.Figure()

      # Add historical data trace
      fig.add_trace(go.Scatter(
          x=df_last_year['Date'],
          y=df_last_year['Close'],
          mode='lines+markers',  # Adding markers
          name='Historical Data',
          line=dict(color='#9AA6B2')
      ))

      # Add prediction results trace
      df_prediction = df_last_year[df_last_year['Date'] >= start_pred]
      fig.add_trace(go.Scatter(
          x=df_prediction['Date'],
          y=df_prediction['Close'],
          mode='lines+markers',
          name='Prediction Results',
          line=dict(color='#133E87', dash='dot'),  # Custom line style

      ))

      # Update layout
      fig.update_layout(
          title=stock_company_map.get(stock_name),
          title_font=dict(size=22),
          xaxis_title='Date',
          yaxis_title='Weekly Price',
          legend=dict(x=1.05, y=1, bgcolor='rgba(255, 255, 255, 0)', bordercolor='LightGray'),
          margin=dict(l=40, r=40, t=40, b=40),
          template='plotly_white'
      )

      # Display the plot in Streamlit
      st.plotly_chart(fig, use_container_width=True)

      return df_prediction


# Streamlit interface for plotting
import io


# Streamlit interface for plotting
for stock in stock_input:
    pred_result_all = plot_stock_predictions(stock)
    pred_result = pred_result_all[['Date', 'Close']].rename(columns={'Close':'Predicted Close'})
    
    # Display the prediction results as a static table
    pred_result_no_index = pred_result.reset_index(drop=True)
    pred_result_str = pred_result_no_index.to_string(index=False) 
    st.markdown(
        f"""
            <h4 style="text-align:left;color:#9AA6B2;"><i>Prediction Results for {stock}</i></h4>
        """, unsafe_allow_html = True
    )
    st.dataframe(pred_result_no_index)  # Static table display
    # Create download buttons for CSV and Excel file
    col1, col2 = st.columns(2)
    
    with col1:
        # Download button for CSV (placed in the first column)
        csv = pred_result.to_csv(index=False)
        download_csv = st.download_button(
            label="📄 Download to CSV",
            data=csv,
            file_name=f"{stock}_prediction.csv",
            mime="text/csv"
        )
        if download_csv:
            st.success(f"Download of {stock}_prediction.csv was successful!")
    
    with col2:
        # Download button for Excel (placed in the second column)
        excel_buffer = io.BytesIO()
        pred_result.to_excel(excel_buffer, engine='openpyxl', index=False)
        excel_buffer.seek(0)  # Rewind the buffer to the beginning
        download_excel = st.download_button(
            label="📄 Download to Excel",
            data=excel_buffer,
            file_name=f"{stock}_prediction.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        if download_excel:
            st.success(f"Download of {stock}_prediction.xlsx was successful!")

# Analysis & Recommendation
st.markdown(
    """
    <h4 style="text-align:center;color:#9AA6B2;margin-bottom:15px;"><i>Analysis & Recommendation</i></h4>
    """, unsafe_allow_html=True
)

response_2 = result_analysis(pred_result_str, stock_input, df_fundamental=None, df_financial=None)

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
    ">
        {response_2}
    </div>
    """, unsafe_allow_html=True
)

st.markdown(
    """
    <h6 style="text-align:center;color:#9AA6B2;margin-bottom:15px;"><i>Analysis Generated by OpenAI's ChatGPT</i></h6>
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

disclaimer_text = "This material is provided for informational purposes only and should not be construed as financial advice. The decision to invest remains solely your responsibility."
st.markdown(
    f"""
    <br><h5 style="text-align:center;color:#133E87;font-weight:bold;"><b><i>{str(disclaimer_text).upper()}</b></i></h5><br>
    """, unsafe_allow_html=True
)