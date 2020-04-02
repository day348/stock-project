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
        cols=["date", "close"] 
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
                
#Pass path and tic to normalized data
def exportToCSVTestingAndTraining(path,tic):
    if os.path.exists(path + tic + r'.csv'):
        dataSplit=splitCSVData(path,tic)
        testing=dataSplit[0].tail(3500)
        training=dataSplit[1].tail(3500)
        testing.to_csv(r'data\\testing\\' + tic + r'.csv',  index = False)
        training.to_csv(r'data\\training\\' + tic + r'.csv', index = False) 


if __name__ == "__main__":
    tickers = pd.read_csv('data\\stock_names.csv')['Ticker'] #gets stock Tickers 
    executor = concurrent.futures.ProcessPoolExecutor(10)
    #runs the update stock tic method for each ticker
    futures = [executor.submit(exportToCSVTestingAndTraining,'data\\normalized_data\\' ,tic,) for tic in tickers]
    concurrent.futures.wait(futures)