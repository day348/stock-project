import pandas as pd 
import os
from progressbar import ProgressBar


def num1sTesting(path,numToCheck=10):
    tickers = pd.read_csv('data/stock_names.csv')['Ticker'] #gets stock Tickers 
    total = 0
    numRight = 0
    tickers = tickers[:numToCheck]
    pbar = ProgressBar()
    for tic in pbar(tickers):
        if os.path.exists( path + tic + r'.csv'):
            data = pd.read_csv(path + tic + r'.csv')
            
            #gets the number of 1 and zeros in the testing file
            for i in range(len(data)):
                if data['1or0'][i] == 1:
                    numRight = numRight + 1
                total = total +1
            
    print("percent 1's: ", numRight*100/total, "%")
    return numRight, total

def percent_change(attributtes,threshold,days_ahead = 1,numToCheck =10):
    path = 'data/historical_stock_data/'
    tickers = pd.read_csv('data/stock_names.csv')['Ticker'] #gets stock Tickers 
    total = 0
    numRight = 0
    if numToCheck >0:
        tickers = tickers[:numToCheck]
    pbar = ProgressBar()
    for tic in pbar(tickers):
        if os.path.exists( path + tic + r'.csv'):
            data = pd.read_csv(path + tic + r'.csv')
            
            #gets the number of times a stock grows a specific percent over a period of time
            for i in range(len(data[attributtes[1]])-days_ahead):
                if threshold > 0:
                    highPrice = data[attributtes[1]][i+1:i+days_ahead].max()
                else:
                    highPrice = data[attributtes[1]][i+1:i+days_ahead].min()
                currPrice = data[attributtes[0]][i]
                percent_increase = (highPrice-currPrice)/currPrice 
                if percent_increase > threshold:
                    numRight = numRight +1
                total = total +1
    print("percent meeting threshold: ", numRight*100/total, "%")
    return numRight, total

#Examples:
""" percent_change(['close','high'],.04,days_ahead=5) 
num1sTesting('data/testing/') """