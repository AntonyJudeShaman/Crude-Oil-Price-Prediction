import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

data = pd.read_csv('fdata.csv', index_col='Date', parse_dates=True)
exogg = pd.read_csv('ndata.csv', index_col='Date', parse_dates=True)

ts = data['Close']

model = ARIMA(endog=ts, exog=exogg, order=(1,1,1))
arima = model.fit()


import pickle

pickle.dump(arima,open('arima.pkl','wb'))