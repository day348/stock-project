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
import matplotlib.pyplot as plt
import statistics

PREDICTION_THRESHOLD = .65
GAIN_THRESHOLD = .03
MINIMUM_PRICE = 0
DAYS_AHEAD = 4
ATTRIBUTES = ['close','high']
PERCENT_STOCKS = .5


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


def day_change_check(data,i,attributtes,threshold,days_ahead = 1):
    if attributtes[1] == 'close':
        highPrice = data[attributtes[1]][i+days_ahead]
    else:
        if threshold >= 0:
            highPrice = data[attributtes[1]][i+1:i+1+days_ahead].max()
        else:
            highPrice = data[attributtes[1]][i+1:i+1+days_ahead].min()
    currPrice = data[attributtes[0]][i]
    percent_increase = (highPrice-currPrice)/currPrice 
    if (percent_increase > threshold) & (threshold >= 0):
        return True, percent_increase
    if (percent_increase < threshold) & (threshold < 0):
        return True,percent_increase
    return False,percent_increase

def testing_network(tic):
    #gets data
    weights = np.load('test_results/Great+03/end_weights.npy',allow_pickle=True)
    network = nn.NeuralNet([0,0,7],[41,100,50,1])
    network.weights = weights
    stock_data_normalized = pd.read_csv('data/normalized_data/' + tic + '.csv',index_col='date').tail(1520)
    stock_data = pd.read_csv('data/historical_stock_data/' + tic + '.csv',index_col='date',).tail(1520)
    stock_data = stock_data[stock_data_normalized.index[0]:]
    results = [0]*len(stock_data_normalized.index[20:-DAYS_AHEAD])
    i=0
    for date in stock_data_normalized.index[20:-DAYS_AHEAD]:
        input_data = dp.getInputs(date,stock_data_normalized)
        prediction = network.calculateOutput(input_data,single_input=True)
        #disregards low priced stocks
        if stock_data.loc[date,'close'] > MINIMUM_PRICE:
            percent_change = ((stock_data.loc[stock_data.index[20+i+DAYS_AHEAD],'close'] - stock_data.loc[date,'close']) / stock_data.loc[date,'close'])*100
            crossed_threshold, high = day_change_check(stock_data,20+i,ATTRIBUTES,GAIN_THRESHOLD,days_ahead=DAYS_AHEAD,)
            if crossed_threshold:
                results[i]=(prediction,1,percent_change,high)
            else:   
                """ if (percent_change > .03) & (prediction >.8):
                    print('high')
                    print('5 day close price:',stock_data.loc[stock_data.index[i+5],'close'])
                    print('compare price:', stock_data.loc[date,'close'])
                elif (percent_change < -.5)& (prediction >.8):
                    print('low')
                    print('5 day close price:',stock_data.loc[stock_data.index[i+5],'close'])
                    print('compare price:', stock_data.loc[date,'close']) """
                if percent_change > 3:
                    print('date:', date)
                    print('five day date:', stock_data.index[20+i+DAYS_AHEAD])
                    print('current days price: ', stock_data.loc[date,'close'])
                    print('five day close: ', stock_data.loc[stock_data.index[20+i+DAYS_AHEAD],'close'])
                    print('day change high:',high)
                """ if percent_change < -3:
                    print('date:', date)
                    print('five day date:', stock_data.index[20+i+DAYS_AHEAD])
                    print('current days price: ', stock_data.loc[date,'close'])
                    print('five day close: ', stock_data.loc[stock_data.index[20+i+DAYS_AHEAD],'close'])
                    print('day change high:',high)
                    print('percent change: ', percent_change) """
                results[i] = (prediction,0,percent_change,high)
        i = i+ 1

    return [j for j in results if type(0) != type(j)] 
   
if __name__ == "__main__":
        
    save_string = 'prediction threshold: ' + str(PREDICTION_THRESHOLD) + '\n'
    save_string = save_string + 'gain threshold: ' + str(GAIN_THRESHOLD) + '\n'
    save_string = save_string + 'minimum price: ' + str(MINIMUM_PRICE) + '\n'
    save_string = save_string + 'days ahead: ' + str(DAYS_AHEAD) + '\n'
    save_string = save_string + 'attributes: ' + str(ATTRIBUTES) + '\n'
    save_string = save_string + '\nSTATS:\n\n'

    #gets network and tickers
    weights = np.load('test_results/Great+03/end_weights.npy',allow_pickle=True)
    network = nn.NeuralNet([0,0,7],[41,100,50,1])
    network.weights = weights
    stock_tickers = pd.read_csv('data/stock_names.csv')['Ticker']
        #removes tickers wihtout data
    for tic in stock_tickers:
        if not os.path.exists('data/normalized_data/' + tic + '.csv'):
            stock_tickers = stock_tickers[stock_tickers != tic]
    stock_tickers = stock_tickers.sample(frac=PERCENT_STOCKS)

    #collects data
    print('collecting data...')
    pool = Pool()
    results = list(tqdm(pool.imap(testing_network,stock_tickers),total=len(stock_tickers)))
    pool.close()
    pool.join()
    results = list(itertools.chain.from_iterable(results))
    

    #grabs number predicted right
    results = [i for i in results if type(1) != type(i)]
    numRight = len([i for i in results if ((i[1] == 1)&((i[0] > PREDICTION_THRESHOLD)))])
    total = len([i for i in results if ((i[0] > PREDICTION_THRESHOLD) )])

    #gets failure stats
    failure_gains = [i[2] for i in results if (((i[0] > PREDICTION_THRESHOLD) ))]
    failure_mean = np.mean(failure_gains)
    print(len(failure_gains))
    
    #wrties out string
    save_string = save_string + 'total predicted right to boom: ' + str(numRight) + '\n'
    save_string = save_string + 'total predicted to boom: ' + str(total) + '\n'
    save_string = save_string + 'percent right: ' + str(numRight/total) + '\n'
    save_string = save_string + 'percent meeting threshold in general: ' + str(len([i for i in results if i[1] == 1])/len(results)) + '\n'
    save_string = save_string + 'average false positive ' + str(DAYS_AHEAD) + ' day gain: ' + str(failure_mean) + '% ' + '\n'
    save_string = save_string + 'average 5 day gain not meeting threshold: ' + str(statistics.mean([i[2] for i in results if i[1] == 0]))

    #creates new folder
    folder_created = False
    statsnum = -1
    while not folder_created:
        statsnum = statsnum + 1
        try:
            stats_folder = 'data/stats/' + str(statsnum)
            os.mkdir(stats_folder)
            stats_folder = stats_folder +'/'
            folder_created = True
        except FileExistsError:
            pass
    
    #saves string
    print('\n')
    print(save_string)
    overview_file = open(stats_folder + 'overview_string.txt', "w")
    overview_file.write(save_string)
    overview_file.close()

    #gets plots
    plt.hist(failure_gains,bins=np.arange(-10,10,.25), rwidth= .9)
    plt.xlabel(str(DAYS_AHEAD) + ' Day Close Price % Gain')
    plt.title('False Positive Close Prices')
    plt.text(-4,0.3,'mean: ' + str(failure_mean))
    plt.savefig(stats_folder+'false_positive_gains_hist.png')
    plt.close()
    #histogram of overal gains
    overal_gains = [i[2] for i in results] #if i[2]< GAIN_THRESHOLD*100]
    print(len(overal_gains))
    plt.hist(overal_gains,bins= np.arange(-10,10,.25),rwidth= .9)
    plt.xlabel(str(DAYS_AHEAD) + ' Day Close Price % Gain')
    plt.title('Close Prices Under Threshold')
    plt.text(-4,0.3,'mean: ' + str(statistics.mean(overal_gains)))
    plt.savefig(stats_folder + str(DAYS_AHEAD) +  'day_gain_true_hist.png')
    plt.close()
    #line graph
    prediction_averages = []
    x=[]
    for j in range(80):
        range_values = [i[2] for i in results if ((i[0]*100 >j) & (i[0]*100 <= j+1))]
        if (range_values != []) :
            mean = statistics.mean(range_values)
            if(mean < 20):
                prediction_averages.append(mean)
                x.append(j)
            else:
                prediction_averages.append(25)
                x.append(j)
    best_fit = np.polyfit(x,prediction_averages,1)
    best_fit = np.poly1d(best_fit)
    plt.scatter(x,prediction_averages)
    plt.plot(x,best_fit(x))
    plt.ylabel('Average ' + str(DAYS_AHEAD) + ' Day % Gain ')
    plt.xlabel('Prediction Output')
    plt.title('% Gain vs Prediction')
    plt.savefig(stats_folder+ str(DAYS_AHEAD) + 'dayclose_scatter.png')