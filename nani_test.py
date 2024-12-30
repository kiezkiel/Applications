import streamlit as st
from datetime import date
import yfinance as yf
import requests
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Define API Key (replace with your NewsAPI key)
API_KEY = "YOUR_NEWSAPI_KEY"

# Define date range for data
start = "1980-09-01"
end = date.today().strftime("%Y-%m-%d")

st.title("Enhanced Stock Market Prediction with Sentiment Analysis")

# Select stock
stocks = ("TSLA", "NVDA", "MSFT", "GME", "AMD", "META", "GOOG", "AAPL", "AMZN", "NFLX", "JPM")
selected_stock = st.selectbox("Select The Stock for Prediction", stocks)

# Select prediction period
n_days = st.slider("Days of Prediction:", 1, 90)
st.write(f"Prediction period: {n_days} days")

# Sidebar for debug logs
with st.sidebar:
    st.subheader("Debug Log")
    st.write("Logs will appear here.")


@st.cache_data
def fetch_news_sentiment(stock):
    """
    Fetch recent news articles for the selected stock and calculate sentiment polarity.
    """
    url = f"https://newsapi.org/v2/everything?q={stock}&apiKey={API_KEY}"
    with st.sidebar:
        st.write(f"Fetching news data from: {url}")  # Log API call

    response = requests.get(url).json()

    # Handle potential API errors
    if response.get("status") != "ok":
        st.error("Error fetching news data. Check API key or connection.")
        with st.sidebar:
            st.write(f"API Error: {response.get('message', 'Unknown error')}")
        return []

    articles = response.get("articles", [])
    with st.sidebar:
        st.write(f"Number of articles fetched: {len(articles)}")

    sentiments = []
    for article in articles:
        title = article.get("title", "")
        description = article.get("description", "")
        text = title + " " + (description if description else "")
        sentiment_score = TextBlob(text).sentiment.polarity
        sentiments.append(sentiment_score)

    with st.sidebar:
        st.write(f"Sample Sentiments: {sentiments[:5]}")

    return sentiments


@st.cache_data
def load_data_with_sentiment(ticker):
    """
    Load stock market data and add sentiment analysis data as a new feature.
    """
    data = yf.download(ticker, start, end)

    # Add sentiment analysis data
    sentiments = fetch_news_sentiment(ticker)
    sentiment_avg = np.mean(sentiments) if sentiments else 0
    data["Sentiment"] = sentiment_avg  # Add as a constant column (placeholder for now)

    data = data.dropna().reset_index()
    return data


# Load and display data
data_load_state = st.text("Loading data...")
data = load_data_with_sentiment(selected_stock)
data_load_state.text("Loading data ... done")

st.subheader("Raw Data with Sentiment")
st.write(data.tail())

# Verify column names
st.sidebar.write("Column names in the DataFrame:", data.columns)

# Check if the 'Close' column exists
if "Close" in data.columns:
    st.subheader("Time Series Data")
    st.line_chart(data[["Close"]])
else:
    st.error("Column 'Close' not found in the DataFrame. Available columns are: " + ", ".join(data.columns))

# Feature selection
features = ["Close", "Sentiment"] if "Close" in data.columns else ["Sentiment"]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features])

# Split data into training and testing sets
train_len = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_len]
test_data = scaled_data[train_len:]

# Prepare training data
sequence_length = 60
X_train, y_train = [], []
for i in range(sequence_length, len(train_data)):
    X_train.append(train_data[i - sequence_length:i])
    y_train.append(train_data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Prepare testing data
X_test, y_test = [], []
for i in range(sequence_length, len(test_data)):
    X_test.append(test_data[i - sequence_length:i])
    y_test.append(test_data[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)

# Build LSTM model
model = Sequential([
    LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(units=32, return_sequences=False),
    Dropout(0.2),
    Dense(units=16, activation="relu"),
    Dense(units=1)
])
model.compile(optimizer="adam", loss="mean_squared_error")

# Train the model
with st.spinner("Training LSTM model..."):
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Predict on test data
test_predictions = model.predict(X_test)
test_predictions = scaler.inverse_transform(
    np.concatenate([test_predictions, np.zeros((len(test_predictions), len(features) - 1))], axis=1)
)[:, 0]

y_test_actual = scaler.inverse_transform(
    np.concatenate([y_test.reshape(-1, 1), np.zeros((len(y_test), len(features) - 1))], axis=1)
)[:, 0]

# Plot predictions vs actual
st.subheader("Prediction vs Actual")
st.line_chart({"Actual": y_test_actual, "Predicted": test_predictions})

# Future predictions
last_60_days = scaled_data[-sequence_length:]
future_inputs = last_60_days.reshape(1, -1, len(features))
future_predictions = []
for _ in range(n_days):
    prediction = model.predict(future_inputs)
    future_predictions.append(prediction[0, 0])
    new_input = np.concatenate([prediction, np.zeros((1, len(features) - 1))], axis=1).reshape(1, 1, len(features))
    future_inputs = np.append(future_inputs[:, 1:, :], new_input, axis=1)

# Scale future predictions back
future_predictions_scaled = scaler.inverse_transform(
    np.concatenate([np.array(future_predictions).reshape(-1, 1), np.zeros((len(future_predictions), len(features) - 1))], axis=1)
)[:, 0]

# Display future predictions
st.subheader("Future Predictions")
st.write(future_predictions_scaled)
st.line_chart(future_predictions_scaled)
