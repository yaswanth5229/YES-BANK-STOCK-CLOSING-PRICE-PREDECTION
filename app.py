import streamlit as st
import numpy as np
import pandas as pd
import sklearn
from PIL import Image
import pickle

model = pickle.load(open('model.sav', 'rb'))

st.title('Yes Bank Stock Closing Price Prediction')
st.sidebar.header('Data')

# Function
def user_report():
    Open = st.sidebar.slider('Open Stock Price', 1,50, 1)
    High = st.sidebar.slider('Highest Stock Price', 1,50, 1)
    Low = st.sidebar.slider('Lowest Stock Price', 1,50, 1)

    user_report_data = {
        'Open':Open,
        'High':High,
        'Low':Low
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

user_data = user_report()
st.header('Bank Data')
st.write(user_data)

Close = model.predict(user_data)
st.subheader('Stock Closing Prices')
st.subheader(np.round(Close[0], 2))
