import yfinance as yf
from matplotlib import pyplot as plt
import numpy as np
from hmmlearn import hmm

msft = yf.Ticker("MSFT")

data = msft.history(period="max")


df = yf.Ticker("DJI")
dt = yf.Ticker("^NDX")

price = dt.history(period = "max")
price = dt.history(start ="2010-01-01")
p = price.iloc[:, 0]






