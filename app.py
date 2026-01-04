import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date, timedelta


st.set_page_config(
    page_title="StockVision",  # The name on the browser tab
    page_icon="📈",              # The little icon on the tab
    
)

# --- HIDE STREAMLIT STYLE ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


model = load_model('Stock Predictions Model.keras')

st.header('Stock Price Prediction App')
st.sidebar.write("Navigate the financial markets with AI-powered insights.")

stock= st.text_input('Enter Stock Symbol', 'NVDA')
start = '2015-01-01' 
end = date.today().strftime("%Y-%m-%d")

data = yf.download(stock, start, end)

if data.empty:
    st.error("Invalid Ticker Symbol. Please enter a valid stock (e.g., AAPL, TSLA).")
    st.stop()  # This stops the app here so it doesn't crash later

st.subheader('Stock Data ')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])

#A new column named 'test' 
data_test = pd.DataFrame(data.Close[int(len(data) * 0.8): len(data)])

from sklearn.preprocessing import MinMaxScaler # type: ignore
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scaler = scaler.fit_transform(data_test)

st.subheader('Technical Analysis (MA50 vs MA200)')

#Calculates the standard "Golden Cross" indicators
ma_50_days = data.Close.rolling(50).mean()
ma_200_days = data.Close.rolling(200).mean()

fig1 = plt.figure(figsize=(10,6))
plt.plot(data.Close, 'g', label='Price')
plt.plot(ma_50_days, 'r', label='MA50 (Short Term)')
plt.plot(ma_200_days, 'b', label='MA200 (Long Term)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig1)

x = []
y = []

for i in range(100, data_test_scaler.shape[0]):
    x.append(data_test_scaler[i-100:i])
    y.append(data_test_scaler[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y=y * scale


st.subheader(' Original Price vs Predicted Price')
fig4= plt.figure(figsize=(8,6))
plt.plot(y, 'r', label='Original Price')
plt.plot(predict, 'b', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)



st.subheader('Future Price Prediction (Next 30 Days)')

# 1. Get the last 100 days of the scaled data to start our predictions
# Note: We use data_test_scaler which is already scaled between 0 and 1
curr_input = data_test_scaler[len(data_test_scaler)-100:].reshape(1, -1)
temp_input = list(curr_input)
temp_input = temp_input[0].tolist()

# 2. Predict the next 30 days recursively
lst_output = []
n_steps = 100
i = 0
future_days = 30 # Change this if you want to predict more/fewer days

while(i < future_days):
    if(len(temp_input) > 100):
        curr_input = np.array(temp_input[1:])
        curr_input = curr_input.reshape(1, -1)
        curr_input = curr_input.reshape((1, n_steps, 1))
        yhat = model.predict(curr_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        lst_output.extend(yhat.tolist())
        i = i + 1
    else:
        curr_input = curr_input.reshape((1, n_steps, 1))
        yhat = model.predict(curr_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i = i + 1

# 3. Convert predictions back to original prices (Dollars)
lst_output = scaler.inverse_transform(lst_output)

# LOGIC FOR NEXT TRADING DAY 
today = date.today()
weekday_num = today.weekday() # 0=Monday, 1=Tuesday, ... 4=Friday, 5=Saturday, 6=Sunday

if weekday_num >= 4: # If it is Friday(4), Saturday(5), or Sunday(6)
    next_day_text = "Monday"
else:
    # If it is Mon-Thu, the next trading day is just tomorrow
    next_date = today + timedelta(days=1)
    next_day_text = next_date.strftime("%A") # %A gives the full name (e.g., "Tuesday")

st.subheader(f'Predicted Price for {next_day_text}')


# 5. Create a graph for the future predictions
fig5 = plt.figure(figsize=(10, 6))

# Create dummy time steps for plotting
# The last real day is '100', so future starts at '101'
day_new = np.arange(1, 101)
day_pred = np.arange(101, 101 + future_days)

# We grab the last 100 days of REAL data to show the connection
real_data_last_100 = scaler.inverse_transform(data_test_scaler[len(data_test_scaler)-100:])

plt.plot(day_new, real_data_last_100, 'b', label="Past 100 Days (Actual)")
plt.plot(day_pred, lst_output, 'r', label=f"Next {future_days} Days (Predicted)")
plt.title(f'Future Stock Prediction ({future_days} Days)')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig5)


# LIVE ACCURACY DASHBOARD ---
st.divider()  # Adds a divider line
st.subheader("Live Accuracy Dashboard")

# 1. Fetch the absolute latest price from Yahoo Finance
live_data = yf.Ticker(stock).history(period="1d")
latest_price = live_data['Close'].iloc[-1]

# 2. Get your Predicted Price (Day 1 of the future loop)
predicted_next_close = lst_output[0][0]

# 3. Calculate the difference
difference = predicted_next_close - latest_price

# 4. Display nicely with Streamlit Metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Latest Market Price", value=f"${latest_price:.2f}")

with col2:
    st.metric(label="Predicted Next Close", value=f"${predicted_next_close:.2f}")

with col3:
    # Green means the model predicts the price will go UP.
    # Red means the model predicts the price will go DOWN.
    st.metric(label="Expected Change", value=f"${difference:.2f}", delta=f"{difference:.2f}")

st.caption("Note: 'Latest Market Price' is the closing price of the last trading session.")

with st.sidebar.expander("About the Model"):
    st.write(
        "This app uses a **Long Short-Term Memory (LSTM)** neural network. "
        "It is trained on historical data from Yahoo Finance to recognize "
        "price patterns and forecast future trends."
    )
