import yfinance as yf
from matplotlib import pyplot as plt
import numpy as np
from hmmlearn import hmm

msft = yf.Ticker("MSFT")

data = msft.history(period="max")


df = yf.Ticker("DJI")
dt = yf.Ticker("^NDX")

price = dt.history(period = "max")
price = dt.history(start ="2017-01-01")

p = price['Close'] - price['Open']

a=p.to_numpy()
a = np.reshape(a,(1036,1))

from hmmlearn.hmm import GaussianHMM
model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000).fit(a)

# Predict the optimal sequence of internal hidden state
hidden_states = model.predict(a)



#########################
import pandas as pd


b = price['Close']
hs = pd.Series(hidden_states)

hs.index = b.index
A=pd.concat([hs,b], axis = 1, ignore_index=True)
A.columns=['A','B']

A.A.plot()
A.B.plot(secondary_y=True)