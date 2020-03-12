import numpy as np
import pandas as pd
import os
import concurrent.futures


def percentChange(base, value):
    if(base!=0):
        diff=value-base
        div=diff/base
        div+=1
        return div
    else:
        raise ZeroDivisionError
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
        for i in range(0,len(data['high'])):
            base=data['52 day average'][i]
            baseVol= data['52 week volume average'][i]
            low[i]=percentChange(base, data['low'][i])
            high[i]=percentChange(base, data['high'][i])
            average[i]=percentChange(base, data['average'][i])
            close[i]=percentChange(base, data['close'][i])
            openn[i]=percentChange(base, data['open'][i])
            rangee[i]=percentChange(base, data['range'][i])
            twelveDay[i]=percentChange(base, data['12 day ema'][i])
            twentySixDay[i]=percentChange(base, data['26 day ema'][i])
            volume[i]=percentChange(baseVol,data['volume'][i])
            volumeEMA[i]=percentChange(baseVol,data['volume ema'][i])
            singleDay[i]=percentChange(base, data['single_day_change'][i])
            dayToDay[i]=percentChange(base, data['day_to_day_change'][i])
            fiftyTwoDayHigh[i]=percentChange(base, data['52 day high'][i])
            fiftyTwoWeekHigh[i]=percentChange(base, data['52 week high'][i])
            fiftyTwoDayLow[i]=percentChange(base, data['52 day low'][i])
            fiftyTwoWeekLow[i]=percentChange(base, data['52 week low'][i])
            fiftyTwoWeekAverage[i]=percentChange(base, data['52 week average'][i])
            fiftyTwoDayStandDev[i]=percentChange(base, data['52 day standard deviation'][i])
            fiftyTwoWeekStandDev[i]=percentChange(base, data['52 week standard deviation'][i])
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


getPath = 'data\\historical_stock_data\\'
savePath = 'data\\normalized_data\\'

def update_stock(tic):
    data = normalize(getPath, tic)
    if not isinstance(data, int):
        data.to_csv(savePath + tic + r'.csv', index = True)
        print('end: ' + tic)
    return(1)

if __name__ == "__main__":
    tickers = pd.read_csv('data\\stock_names.csv')['Ticker'] #gets stock Tickers 
    print(tickers.iloc[1])
    executor = concurrent.futures.ProcessPoolExecutor(10)
    #runs the update stock tic method for each ticker
    futures = [executor.submit(update_stock, tic) for tic in tickers]
    concurrent.futures.wait(futures)