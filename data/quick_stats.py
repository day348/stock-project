import pandas as pd 
import os
from progressbar import ProgressBar
import numpy as np
import sys
sys.path.append('/Users/jewellday/Documents/OneDrive/Documents/Capstone/stock-project')
import neural_network.NeuralNet as nn
import training.testing as testing
from data import inputsForBackProp as dp
from tqdm import tqdm
from multiprocessing import Pool
import itertools

def num1sTesting(path,numToCheck=10):
    tickers = pd.read_csv('data/stock_names.csv')['Ticker'] #gets stock Tickers 
    total = 0
    numRight = 0
    #set to negative to use all stocks 
    if numToCheck > 0:
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
                if threshold >= 0:
                    highPrice = data[attributtes[1]][i+1:i+1+days_ahead].max()
                else:
                    highPrice = data[attributtes[1]][i+1:i+1+days_ahead].min()
                currPrice = data[attributtes[0]][i]
                percent_increase = (highPrice-currPrice)/currPrice 
                if (percent_increase > threshold) & (threshold >= 0):
                    numRight = numRight +1
                if (percent_increase < threshold) & (threshold < 0):
                    numRight = numRight +1
                total = total +1
    print("percent meeting threshold: ", numRight*100/total, "%")
    return numRight, total


def day_change_check(data,attributtes,threshold,days_ahead = 1):
    #gets the number of times a stock grows a specific percent over a period of time
    for i in range(len(data[attributtes[1]])-days_ahead):
        if threshold >= 0:
            highPrice = data[attributtes[1]][i+1:i+1+days_ahead].max()
        else:
            highPrice = data[attributtes[1]][i+1:i+1+days_ahead].min()
        currPrice = data[attributtes[0]][i]
        percent_increase = (highPrice-currPrice)/currPrice 
        if (percent_increase > threshold) & (threshold >= 0):
            return True
        if (percent_increase < threshold) & (threshold < 0):
            return True
        return False

def testing_network(tic):
    #gets data
    stock_data_normalized = pd.read_csv('data/normalized_data/' + tic + '.csv',index_col='date')
    stock_data = pd.read_csv('data/historical_stock_data/' + tic + '.csv',index_col='date',).tail(1500)
    for date in stock_data_normalized.index[20:]:
        input_data = dp.getInputs(date,stock_data_normalized)
        prediction = network.calculateOutput(input_data,single_input=True)
        if day_change_check(stock_data,['close','high'],.03,4):
            return (prediction,1)
        return (prediction,0)

network = nn.NeuralNet([0,0,7],[41,100,50,1])        
if __name__ == "__main__":
        
    #Examples:
    """ percent_change(['close','high'],.04,days_ahead=5) 
    num1sTesting('data/testing/') """

    weights = np.load('test_results/Great+03/end_weights.npy',allow_pickle=True)
    network = nn.NeuralNet([0,0,7],[41,100,50,1])
    network.weights = weights
    stock_tickers = pd.read_csv('data/stock_names.csv')['Ticker']
        #removes tickers wihtout data
    for tic in stock_tickers:
        if not os.path.exists('data/normalized_data/' + tic + '.csv'):
            stock_tickers = stock_tickers[stock_tickers != tic]

    pool = Pool()
    results = list(tqdm(pool.imap(testing_network,stock_tickers),total=len(stock_tickers)))
    pool.close()
    pool.join()
    print(len(results[0]))
    results = [[i for i in results[j] if i] for j in range(len(results))]
    results = list(itertools.chain.from_iterable(results))
    #grabs data from 2% of stocks
    
    print(len(results))
            