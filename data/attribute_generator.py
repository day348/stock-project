import numpy as np
import pandas as pd
import os
import concurrent.futures
#calculates the various attributes from the raw price data
def complete_attributes(data):
    
    #simple attributes
    data['range'] = data['high'] - data['low']
    data['average'] = (data['high'] + data['low'] + data['close'] ) / 3
    data['single_day_change'] = data['close'] - data['open']

    #differnce between the last 2 day's closes
    dtd = [0]
    for i in range(len(data)-1):
        dtd = dtd + [data.iloc[i+1]['close']-data.iloc[i]['close']]
    data['day_to_day_change'] = dtd

    # 12 day moving averages + maybe MACD and bollinger Band
    ema = [data.iloc[0]['average']]
    for i in range(len(data)-1):
        currPrice = float(data.iloc[i+1]['average'])
        oldEma = float(ema[i])
        ema = ema + [currPrice*.15 + oldEma*.85]
    data['12 day ema'] = ema

    #26 day exponential moving average
    ema = [data.iloc[0]['average']]
    for i in range(len(data)-1):
        currPrice = float(data.iloc[i+1]['average'])
        oldEma = float(ema[i])
        ema = ema + [currPrice*.075 + oldEma*.925]
    data['26 day ema'] = ema

    #ema for volume
    volAvg = [data.iloc[0]['volume']]
    for i in range(len(data)-1):
        currVolume = float(data.iloc[i+1]['volume'])
        oldVolAvg = float(volAvg[i])
        volAvg = volAvg + [currVolume*.05 + oldVolAvg*.95]
    data['volume ema'] = volAvg

    #52 week volume average
    volAvg = []
    for i in range(len(data)): 
        if i == 0:
            volAvg = [data.iloc[0]['volume']]
        elif  i <= 365:
            volAvg = volAvg + [np.average(data.iloc[:i]['volume'])]
        else:
            volAvg = volAvg + [np.average(data.iloc[i-365:i]['volume'])]
    data['52 week volume average'] = volAvg

    #52 day high
    high52 = []
    for i in range(len(data)): 
        if i == 0:
            high52 = [data.iloc[0]['high']]
        elif  i <= 52:
            high52 = high52 + [max(data.iloc[:i]['high'])]
        else:
            high52 = high52 + [max(data.iloc[i-52:i]['high'])]
    data['52 day high'] = high52

    #52 week high
    high52 = []
    for i in range(len(data)): 
        if i == 0:
            high52 = [data.iloc[0]['high']]
        elif  i <= 365:
            high52 = high52 + [max(data.iloc[:i]['high'])]
        else:
            high52 = high52 + [max(data.iloc[i-365:i]['high'])]
    data['52 week high'] = high52


    #52 day low
    low52 = []
    for i in range(len(data)): 
        if i == 0:
            low52 = [data.iloc[0]['low']]
        elif  i <= 52:
            low52 = low52 + [min(data.iloc[:i+1]['low'])]
        else:
            low52 = low52 + [min(data.iloc[i-52:i]['low'])]
    data['52 day low'] = low52

    #52 week low
    low52 = []
    for i in range(len(data)): 
        if i == 0:
            low52 = [data.iloc[0]['low']]
        elif  i <= 365:
            low52 = low52 + [min(data.iloc[:i+1]['low'])]
        else:
            low52 = low52 + [min(data.iloc[i-365:i]['low'])]
    data['52 week low'] = low52

    #52 day average
    avg52 = []
    for i in range(len(data)): 
        if i == 0:
            avg52 = [data.iloc[0]['close']]
        elif  i <= 52:
            avg52 = avg52 + [np.average(data.iloc[:i+1]['close'])]
        else:
            avg52 = avg52 + [np.average(data.iloc[i-52:i]['close'])]
    data['52 day average'] = avg52

    #52 week average
    avg52 = []
    for i in range(len(data)): 
        if i == 0:
            avg52 = [data.iloc[0]['close']]
        elif  i <= 365:
            avg52 = avg52 + [np.average(data.iloc[:i+1]['close'])]
        else:
            avg52 = avg52 + [np.average(data.iloc[i-365:i]['close'])]
    data['52 week average'] = avg52

    #52 day standard deviation
    std = []
    for i in range(len(data)): 
        if i == 0:
            std = [0]
        elif  i <= 52:
            std = std + [np.std(data.iloc[:i+1]['close'])]
        else:
            std = std + [np.std(data.iloc[i-52:i]['close'])]
    data['52 day standard deviation'] = std

    #52 week standard deviation
    std = []
    for i in range(len(data)): 
        if i == 0:
            std = [0]
        elif  i <= 365:
            std = std + [np.std(data.iloc[:i+1]['close'])]
        else:
            std = std + [np.std(data.iloc[i-365:i]['close'])]
    data['52 week standard deviation'] = std
    return data

#goes through and completes the attributes for the raw data 

path = 'historical_stock_data\\' #sets location to save files

    
def update_stock(tic):
    if os.path.exists( path + tic + r'.csv'):
        data = pd.read_csv(path + tic + r'.csv')
        if len(data.columns) < 7:
            print('start: ' + tic)
            data = complete_attributes(data)
            data.to_csv(path + tic + r'.csv', index = False)
            print('end: ' + tic)
    return(1)

if __name__ == "__main__":
    tickers = pd.read_csv('data/stock_names.csv')['Ticker'] #gets stock Tickers 
    #runs the update stock tic method for each ticker
    executor = concurrent.futures.ProcessPoolExecutor(10)
    futures = [executor.submit(update_stock, tic) for tic in tickers]
    concurrent.futures.wait(futures)


