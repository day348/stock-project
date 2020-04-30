import numpy as np
import pandas as pd
import os
import concurrent.futures
from random import random

#Pass path and tic of normalized data
def splitCSVData(path,tic):
    if os.path.exists( path + tic + r'.csv'):
        datav = pd.read_csv(path + tic + r'.csv', index_col=False)
        x= len(datav)
        rows=[r for r in range(0,x)]
        cols=["date", "dayToDay"] 
        dataframe=pd.DataFrame(data=datav, index=rows, columns=cols)
        dataframeForTraining=pd.DataFrame(columns=cols)
        dataframeForTesting=pd.DataFrame(columns=cols)
        for k in range(0,x):
            val= random()
            row=dataframe.iloc[[k]]
            if(val<.3):
                dataframeForTesting=dataframeForTesting.append(row,ignore_index=True)
            else:
                dataframeForTraining=dataframeForTraining.append(row,ignore_index=True)
        return [dataframeForTesting, dataframeForTraining]

def splitCSVDataUPDown(path,tic):
    if os.path.exists( path + tic + r'.csv'):
        data = pd.read_csv('data/historical_stock_data/' + tic + r'.csv')

        datav = pd.read_csv(path + tic + r'.csv', index_col=False)
        x= len(datav)
        rows=[r for r in range(0,x)]
        cols=["date", "dayToDay"] 
        dataframe=pd.DataFrame(data=datav, index=rows, columns=cols)
        cols=["date", "Up or Down"] 
        dataframeForTraining=pd.DataFrame(columns=cols)
        dataframeForTesting=pd.DataFrame(columns=cols)
        for k in range(0,x):
            val= random()
            highPrice = data['high'][k+1:k+5].max()
            currPrice = data['high'][k]
            percent_increase = (highPrice-currPrice)/currPrice 
            price=dataframe.iloc[k,1]
            date = dataframe.iloc[k,0]
            if(percent_increase > 0.03 ):
                price = 1
            else:
                price = 0
            df2 = pd.DataFrame({"date": [date],"Up or Down": [price]}, columns=cols)
            if(val<.3):
                dataframeForTesting=dataframeForTesting.append(df2,ignore_index=True)
            else:
                dataframeForTraining=dataframeForTraining.append(df2,ignore_index=True)
        return [dataframeForTesting, dataframeForTraining]

#Pass path and tic to normalized data
def exportToCSVTestingAndTraining(path,tic):
    if os.path.exists(path + tic + r'.csv'):
        dataSplit=splitCSVDataUPDown(path,tic)
        testing=dataSplit[0].tail(1500)
        training=dataSplit[1].tail(1500)
        testing.to_csv(r'data/testing/' + tic + r'.csv',  index = False)
        training.to_csv(r'data/training/' + tic + r'.csv', index = False) 
        print(tic)


if __name__ == "__main__":
    tickers = pd.read_csv('data/stock_names.csv')['Ticker'] #gets stock Tickers 
    print(tickers)
    executor = concurrent.futures.ProcessPoolExecutor(20)
    #runs the update stock tic method for each ticker
    futures = [executor.submit(exportToCSVTestingAndTraining,'data/historical_stock_data/' ,tic,) for tic in tickers]
    concurrent.futures.wait(futures)
    """ exportToCSVTestingAndTraining('data/historical_stock_data/','A') """
    #TO DO: make into function that inputs test folder, attributte, and threshold