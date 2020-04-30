import pandas as pd 
import os
from progressbar import ProgressBar

""" path = 'data/historical_stock_data/' """
path = 'data/testing/'
tickers = pd.read_csv('data/stock_names.csv')['Ticker'] #gets stock Tickers 
total = 0
numRight = 0
tickers = tickers[:1]
pbar = ProgressBar()
for tic in pbar(tickers):
    if os.path.exists( path + tic + r'.csv'):
        data = pd.read_csv(path + tic + r'.csv')
        #gets the number of times a stock grows a specific percent over a period of time
        """ for i in range(len(data['high'])-5):
            highPrice = data['high'][i+1:i+5].max()
            currPrice = data['high'][i]
            percent_increase = (highPrice-currPrice)/currPrice 
            if percent_increase > .03:
                numRight = numRight +1
            total = total +1 """
        #gets the number of 1 and zeros in the testing file
        for i in range(len(data)):
            if data['Up or Down'][i] == 1:
                numRight = numRight + 1
            total = total +1
        
#print("percent meeting threshold: ", numRight*100/total, "%")
print("percent 1's: ", numRight*100/total, "%")