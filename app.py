import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime
import pytz # ADDED THIS FOR TIMEZONE SUPPORT

st.set_page_config(
    page_title="StockVision",
    page_icon="📈",
)

#  HIDE STREAMLIT STYLE 
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}  /* Hides the 3-dot menu at top right */
            footer {visibility: hidden;}     /* Hides the 'Made with Streamlit' footer */
            .stAppDeployButton {display:none;} /* Hides the 'Deploy' button */
            
            /* WE REMOVED 'header {visibility: hidden;}' so the sidebar button stays visible! */
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# DISCLAIMER & LANDING PAGE (ADDED)

if 'disclaimer_accepted' not in st.session_state:
    st.session_state.disclaimer_accepted = False

if not st.session_state.disclaimer_accepted:
    st.title("⚠️ Read Before You Enter")
    
    st.markdown("### 1. Market Time Zone Converter")
    st.write("The US Stock Market operates on New York Time (ET). Use this table to track when the market is open in Nigeria (WAT).")
    
    # Create the Time Zone Table
    time_data = {
        "Trading Session": ["Pre-Market", "Market Open", "Market Close", "After-Hours"],
        "New York Time (ET)": ["4:00 AM", "9:30 AM", "4:00 PM", "4:00 PM - 8:00 PM"],
        "Nigeria Time (WAT)": ["10:00 AM (Activity Starts)", "3:30 PM (Main Session)", "10:00 PM (Session Ends)", "10:00 PM - 2:00 AM"]
    }
    time_df = pd.DataFrame(time_data)
    st.table(time_df)

    st.markdown("---")
    
    st.markdown("### 2. Legal & Risk Disclaimer")
    st.info(
        """
        **Strictly for Educational Purposes:** This application uses Artificial Intelligence to predict stock prices. 
        It is a research project and **not** a financial advisory tool.
        
        * **No Guarantees:** AI predictions are probabilistic and can be wrong.
        * **Zero Liability:** The developer is not responsible for any financial losses incurred based on these numbers.
        * **Data Delay:** Live data is fetched from Yahoo Finance and may have slight delays.
        """
    )
    
    st.markdown("---")

    # The Button to Enter the App
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("✅ I Understand & Accept - Enter App", use_container_width=True):
            st.session_state.disclaimer_accepted = True
            st.rerun() # Reloads the app to show the main content

    st.stop() # THIS STOPS THE MAIN APP FROM LOADING UNTIL BUTTON IS CLICKED


model = load_model('Stock Predictions Model.keras')

st.header('Stock Price Prediction App')
st.sidebar.write("Navigate the financial markets with AI-powered insights.")

stock= st.text_input('Enter Stock Symbol', 'NVDA')
start = '2015-01-01' 
end = date.today().strftime("%Y-%m-%d")

data = yf.download(stock, start, end)


# 2. DATA FIX: IGNORE INCOMPLETE DAY (ADDED)

# Determine time in Nigeria
lagos_tz = pytz.timezone('Africa/Lagos')
now_lagos = datetime.now(lagos_tz)

# If it is before 10 PM (22:00) in Nigeria, delete the last row (today's live data)
if now_lagos.hour < 22:
    if len(data) > 0 and data.index[-1].date() == date.today():
        data = data[:-1]



if data.empty:
    st.error("Invalid Ticker Symbol. Please enter a valid stock (e.g., AAPL, TSLA).")
    st.stop() 

st.subheader('Stock Data ')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.8): len(data)])

from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scaler = scaler.fit_transform(data_test)

st.subheader('Technical Analysis (MA50 vs MA200)')

ma_50_days = data.Close.rolling(50).mean()
ma_200_days = data.Close.rolling(200).mean()

fig1 = plt.figure(figsize=(10,6))
plt.plot(data.Close, 'g', label='Price')
plt.plot(ma_50_days, 'r', label='MA50 (Short Term)')
plt.plot(ma_200_days, 'b', label='MA200 (Long Term)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
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
st.pyplot(fig4)


st.subheader('Future Price Prediction (Next 30 Days)')

curr_input = data_test_scaler[len(data_test_scaler)-100:].reshape(1, -1)
temp_input = list(curr_input)
temp_input = temp_input[0].tolist()

lst_output = []
n_steps = 100
i = 0
future_days = 30 

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

lst_output = scaler.inverse_transform(lst_output)


# 3. LOGIC FOR NEXT TRADING DAY (UPDATED FOR NIGERIA TIME)

current_hour_lagos = now_lagos.hour
today_day_name = now_lagos.strftime("%A")
today_weekday = now_lagos.weekday() 

# If weekday and before 10 PM -> Target is TODAY
if current_hour_lagos < 22 and today_weekday < 5:
    next_day_text = today_day_name
# Otherwise -> Target is NEXT TRADING DAY
else:
    if today_weekday >= 4: # Fri-Sun -> Next is Monday
        next_day_text = "Monday"
    else: # Mon-Thu -> Next is Tomorrow
        next_date = now_lagos + timedelta(days=1)
        next_day_text = next_date.strftime("%A")

st.subheader(f'Predicted Price for {next_day_text}')
#


fig5 = plt.figure(figsize=(10, 6))
day_new = np.arange(1, 101)
day_pred = np.arange(101, 101 + future_days)
real_data_last_100 = scaler.inverse_transform(data_test_scaler[len(data_test_scaler)-100:])

plt.plot(day_new, real_data_last_100, 'b', label="Past 100 Days (Actual)")
plt.plot(day_pred, lst_output, 'r', label=f"Next {future_days} Days (Predicted)")
plt.title(f'Future Stock Prediction ({future_days} Days)')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig5)



#  LIVE ACCURACY DASHBOARD (UPDATED LABEL)

st.divider() 
st.subheader("Live Accuracy Dashboard")

live_data = yf.Ticker(stock).history(period="1d")

if live_data.empty:
    st.warning(f"Could not fetch live data for '{stock}'.")
    st.write(f"THis may be due to limited data on '{stock}'. ")
else:
    latest_price = live_data['Close'].iloc[-1]

    if 'lst_output' in locals() and len(lst_output) > 0:
        predicted_price = lst_output[0][0]
        difference = predicted_price - latest_price
        
        # DYNAMIC LABEL: Matches the header above
        dashboard_label = f"Target Close ({next_day_text})"

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(label="Live Market Price", value=f"${latest_price:.2f}")

        with col2:
            st.metric(label=dashboard_label, value=f"${predicted_price:.2f}")

        with col3:
            st.metric(label="Difference", value=f"${difference:.2f}", delta=f"{difference:.2f}")

    else:
        st.write("Prediction data not available yet.")

st.caption(f"Note: Comparing Live Price vs. The Model's Prediction for {next_day_text}.")
# 

with st.sidebar.expander("About the Model"):
    st.write(
        "This app uses a **Long Short-Term Memory (LSTM)** neural network. "
        "It is trained on historical data from Yahoo Finance to recognize "
        "price patterns and forecast future trends."
    )

    #User Guide(Added)
   
    with st.sidebar.expander("📖 How to Use This App"):
        st.markdown("""
    **Step 1: Check the 'Target Close'**
    * Look at the dashboard. This number is the AI's predicted closing price for the current (or next) trading session.
    
    **Step 2: Compare with 'Live Price'**
    * **Green Difference (+):** The Live Price is *below* the Target. The AI thinks the price might **rise** to meet the target.
    * **Red Difference (-):** The Live Price is *above* the Target. The AI thinks the price might **fall** to meet the target.
    
    **Step 3: Check the Trend (Chart)**
    * Look at the **Technical Analysis** chart.
    * If the **Red Line (MA50)** crosses *above* the **Blue Line (MA200)**, it is often a **Buy Signal** (Golden Cross).
    * If it crosses *below*, it is a **Sell Signal** (Death Cross).
    """)
    
    st.caption("Updated automatically every trading session.")