import streamlit as st
from datetime import date 
import yfinance as yf
import pandas as pd 
import matplotlib.pyplot as plt
import pandas_datareader as data
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go 
from plotly.subplots import make_subplots
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import itertools
import numpy as np
import plotly.tools as tls
import neptune

run = neptune.init_run(
    project="yuki-pikazo/Jotaro",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwOTg0NWVmOC03YTdiLTRhMmMtYmQ5Zi04OGQxMzJjNGJkYWMifQ==",
)  # your cred
start = "2010-01-01"
end = date.today().strftime("%Y-%m-%d")
st.title("Stock Market Prediction")

stocks = ("AAPL","NVDA", 'MSFT', "GME","AMD","MSRT")
selected_stocks = st.selectbox("Select The Stocks for prediction", stocks)

n_years = st.slider("Years of Prediction:", 0,1 , 4)
period = n_years * 365 

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start, end)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load Data....")
data = load_data(selected_stocks)
data_load_state.text("Loading data ..... done")

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y =data['Close'], name = 'stock_close'))
    fig.layout.update(title_text="Time Series data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()


df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
# Define the parameter grid

@st.cache_data
def tune_model(df_train, param_grid):
    # Placeholder for the best RMSE and best params
    best_rmse = None
    best_params = None

    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

    # Fit and evaluate model with each parameter set
    for params in all_params:
        m = Prophet(**params).fit(df_train)  # Fit model with given params
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)
        # Calculate RMSE
        rmse = np.sqrt(np.mean((forecast['yhat'] - df_train['y'])**2))

        # If this RMSE is better than the best seen so far, save this RMSE and these params
        if best_rmse is None or rmse < best_rmse:
            best_rmse = rmse
            best_params = params

    return best_params

param_grid = {  
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
}

# Use the cached function to get the best parameters
best_params = tune_model(df_train, param_grid)
m = Prophet(**best_params)
m.add_seasonality(name='weekly', period=7, fourier_order=6)  # Add custom weekly seasonality
# Fit the model with the best parameters
m = Prophet(**best_params).fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Detect anomalies by getting the difference between the actual and predicted values
df_train['fact'] = df_train['y'].copy()
df_train.loc[(df_train['ds'] > '2023-01-01'), 'fact'] = None
forecast = pd.merge(forecast, df_train, on='ds', how='outer')

# Calculate the residuals (difference between actual and predicted values)
forecast['residuals'] = forecast['fact'] - forecast['yhat']

# Consider data points where the residuals are too high as anomalies
forecast['anomaly'] = forecast.apply(lambda x: 'Yes' if (x['residuals'] > 0.5) or (x['residuals'] < -0.5) else 'No', axis=1)

# Print the anomalies
st.write(forecast[forecast['anomaly'] == 'Yes'])

st.subheader('Forecast data')
st.write(forecast.tail())
 
st.write('Forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('Forecast components')
fig2 = m.plot_components(forecast)
st.pyplot(fig2)

run.stop()