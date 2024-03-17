import numpy as np 
import pandas as pd 
import yfinance as yf 
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

model = load_model('C:\personal folder\Major Project\Stock Predictions Model.keras')

st.header('Stock Market Predictor')

stock= st.text_input('Enter Stock Symbol' , 'GOOG')
start = '2012-01-01'
end = '2024-02-14' 

data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test =pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data.train.tail(100)
data_test = pd.concat([pas_100_days , data_train], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(10,8))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

x = []
y=[]
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict*scale
y= y*scale

fig1
    