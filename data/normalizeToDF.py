import numpy as np
import pandas as pd
import os
import concurrent.futures
import tqdm
from multiprocessing import Pool

def percentChange(low,high, value):
    if high-low!=0 :
        div = (value - low)/(high - low)
        return div
    else:
        return .5
def transform(vals):
    v=[[]]
    alpha=len(vals[0])-1
    while(alpha>0):
        v.append([])
        alpha-=1
    for k in range(0,len(vals[0])):
        for j in range(0,len(vals)):
            v[k].append(vals[j][k])
    return v
def normalize(path, tic):
    if os.path.exists( path + tic + r'.csv'):
        print('start: ' + tic)
        data = pd.read_csv(path + tic + r'.csv')
        low= [0]*len(data['low'])
        high= [0]*len(data['low'])
        average= [0]*len(data['low'])
        volume= [0]*len(data['low'])
        close= [0]*len(data['low'])
        openn= [0]*len(data['low'])
        rangee= [0]*len(data['low'])
        twelveDay= [0]*len(data['low'])
        twentySixDay= [0]*len(data['low'])
        volumeEMA= [0]*len(data['low'])
        singleDay= [0]*len(data['low'])
        dayToDay= [0]*len(data['low'])
        fiftyTwoDayHigh= [0]*len(data['low'])
        fiftyTwoWeekHigh= [0]*len(data['low'])
        fiftyTwoDayLow= [0]*len(data['low'])
        fiftyTwoWeekLow= [0]*len(data['low'])
        fiftyTwoWeekAverage= [0]*len(data['low'])
        fiftyTwoDayStandDev= [0]*len(data['low'])
        fiftyTwoWeekStandDev= [0]*len(data['low'])
        for i in range(10,len(data['high'])):
            priceLow = data['52 week low'][i]
            priceHigh = data['52 week high'][i]
            if(i <= 5):
                volLow = data['volume'][0:i].min()
                volHigh = data['volume'][0:i].max()
                rangeLow = data['range'][0:i].min()
                rangeHigh = data['range'][0:i].max()
                sdayLow = data['single_day_change'][0:i].min()
                sdayHigh = data['single_day_change'][0:i].max()
                ddayLow = data['day_to_day_change'][0:i].min()
                ddayHigh = data['day_to_day_change'][0:i].max()
                deviationLow = data['52 week standard deviation'][0:i].min()
                deviationHigh = data['52 week standard deviation'][0:i].max()
                ddeviationLow = data['52 day standard deviation'][0:i].min()
                ddeviationHigh = data['52 day standard deviation'][0:i].max()
            else:
                volLow = data['volume'][i-5:i].min() 
                volHigh = data['volume'][i-5:i].max()
                rangeLow = data['range'][i-5:i].min()
                rangeHigh = data['range'][i-5:i].max()
                sdayLow = data['single_day_change'][i-5:i].min()
                sdayHigh = data['single_day_change'][i-5:i].max()
                ddayLow = data['day_to_day_change'][i-5:i].min()
                ddayHigh = data['day_to_day_change'][i-5:i].max()
                deviationLow = data['52 week standard deviation'][i-5:i].min()
                deviationHigh = data['52 week standard deviation'][i-5:i].max()
                ddeviationLow = data['52 day standard deviation'][i-5:i].min()
                ddeviationHigh = data['52 day standard deviation'][i-5:i].max()
            low[i]=percentChange(priceLow,priceHigh, data['low'][i])
            high[i]=percentChange(priceLow,priceHigh, data['high'][i])
            average[i]=percentChange(priceLow,priceHigh, data['average'][i])
            close[i]=percentChange(priceLow,priceHigh, data['close'][i])
            openn[i]=percentChange(priceLow,priceHigh, data['open'][i])
            rangee[i]=percentChange(rangeLow,rangeHigh, data['range'][i])
            twelveDay[i]=percentChange(priceLow,priceHigh, data['12 day ema'][i])
            twentySixDay[i]=percentChange(priceLow,priceHigh, data['26 day ema'][i])
            volume[i]=percentChange(volLow,volHigh,data['volume'][i])
            volumeEMA[i]=percentChange(volLow,volHigh,data['volume ema'][i])
            singleDay[i]=percentChange(sdayLow,sdayHigh, data['single_day_change'][i])
            dayToDay[i]=percentChange(ddayLow,ddayHigh, data['day_to_day_change'][i])
            fiftyTwoDayHigh[i]=percentChange(priceLow,priceHigh, data['52 day high'][i])
            fiftyTwoWeekHigh[i]=percentChange(priceLow,priceHigh, data['52 week high'][i])
            fiftyTwoDayLow[i]=percentChange(priceLow,priceHigh, data['52 day low'][i])
            fiftyTwoWeekLow[i]=percentChange(priceLow,priceHigh, data['52 week low'][i])
            fiftyTwoWeekAverage[i]=percentChange(priceLow,priceHigh, data['52 week average'][i])
            fiftyTwoDayStandDev[i]=percentChange(ddeviationLow,ddeviationHigh, data['52 day standard deviation'][i])
            fiftyTwoWeekStandDev[i]=percentChange(deviationLow,deviationHigh, data['52 week standard deviation'][i])
    else:
        return -1
    listOfDates= data['date']
    listOfVols= data['52 week volume average']
    listOfAverages= data['52 day average']
    vals= [low, high, average, volume, close, openn, rangee, twelveDay, twentySixDay,volumeEMA, singleDay, dayToDay,fiftyTwoDayHigh,fiftyTwoWeekHigh,fiftyTwoDayLow,fiftyTwoWeekLow,fiftyTwoWeekAverage,fiftyTwoDayStandDev,fiftyTwoWeekStandDev]  
    v=transform(vals)
    rows=listOfDates
    cols=["low", "high", "average", "volume", "close", "open", "range", "twelveDay", "twentySixDay","volumeEMA", "singleDay", "dayToDay","fiftyTwoDayHigh","fiftyTwoWeekHigh","fiftyTwoDayLow","fiftyTwoWeekLow","fiftyTwoWeekAverage","fiftyTwoDayStandDev","fiftyTwoWeekStandDev"] 
    dataframe=pd.DataFrame(data=v, index=rows, columns=cols)
    return dataframe


getPath = 'data/historical_stock_data/'
savePath = 'data/normalized_data/'

def update_stock(tic):
    data = normalize(getPath, tic)
    if not isinstance(data, int):
        data.to_csv(savePath + tic + r'.csv', index = True)
        #print('end: ' + tic)
    return(1)

if __name__ == "__main__":
    tickers = pd.read_csv('data/stock_names.csv')['Ticker'] #gets stock Tickers 
    executor = concurrent.futures.ProcessPoolExecutor(10)
    #runs the update stock tic method for each ticker
    #update_stock('A')
    pool = Pool()
    results = list(tqdm.tqdm(pool.imap(update_stock, tickers), total=len(tickers)))
    pool.close()
    pool.join()