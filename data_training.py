import numpy as np
import pandas as pd
import os
import concurrent.futures

normalizedPath = 'data\\historical_stock_data\\' # path to savae stock
tickers = pd.read_csv('data\\stock_names.csv')['Ticker'] #gets stock Tickers 


