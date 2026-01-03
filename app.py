import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

# Add 'r' right here v
# model = load_model(r'C:\Users\Taiwo\STOCK\Stock Predictions Model.keras')
model = load_model('Stock Predictions Model.keras')

st.header('Stock Price Prediction App')

stock= st.text_input('Enter Stock Symbol', 'NVDA')
start = '2015-01-01' 
end = '2025-12-20'

data = yf.download(stock, start, end)

st.subheader('Stock Data ')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
# Create a new column named 'test' using bracket notation
data_test = pd.DataFrame(data.Close[int(len(data) * 0.8): len(data)])

from sklearn.preprocessing import MinMaxScaler # type: ignore
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scaler = scaler.fit_transform(data_test)

st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3= plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)


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
# # plt.show()
st.pyplot(fig4)

# # --- FUTURE PREDICTION LOGIC ---

# st.subheader('Prediction for Tomorrow')

# # 1. Get the last 100 days of data from your existing 'data_test' dataframe
# last_100_days = data_test.tail(100)

# # 2. Scale this data (just like we did for the training data)
# last_100_days_scaled = scaler.transform(last_100_days)

# # 3. Reshape it to the format the LSTM model expects: (1 sample, 100 time steps, 1 feature)
# X_future = []
# X_future.append(last_100_days_scaled)
# X_future = np.array(X_future)
# X_future = np.reshape(X_future, (X_future.shape[0], X_future.shape[1], 1))

# # 4. Make the prediction
# predicted_price = model.predict(X_future)

# # 5. Undo the scaling to get the actual price in Dollars
# predicted_price_actual = predicted_price * scale

# # 6. Display the result
# st.write(f"The predicted price for the next trading day is: **${predicted_price_actual[0][0]:.2f}**")

# --- FUTURE PREDICTION LOGIC (GRAPH & PRICE) ---

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

# 4. Display the specific price for "Tomorrow" (Day 1 of prediction)
st.write(f"Predicted Price for Tomorrow: **${lst_output[0][0]:.2f}**")

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