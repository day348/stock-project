import numpy as np
import pandas as pd
import os
import concurrent.futures
from random import random
import quick_stats as stats


TESTINGFOLDER = 'data/testing/testing-01/'
TRAININGFOLDER = 'data/training/training-01/'
ATTRIBUTES = ['close','close']
THRESHOLD = -.01
DAYSAHEAD = 2
NUMSTOCKS = -1

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
        cols=["date", ATTRIBUTES[0]] 
        dataframe=pd.DataFrame(data=datav, index=rows, columns=cols)
        cols=["date", "1or0"] 
        dataframeForTraining=pd.DataFrame(columns=cols)
        dataframeForTesting=pd.DataFrame(columns=cols)
        for k in range(0,x):
            val= random()
            date = dataframe.iloc[k,0]
            if len(ATTRIBUTES) > 1:
                if THRESHOLD >= 0:
                    highPrice = data[ATTRIBUTES[1]][k+1:k+1+DAYSAHEAD].max()
                else:
                    highPrice = data[ATTRIBUTES[1]][k+1:k+1+DAYSAHEAD].min()
                currPrice = data[ATTRIBUTES[0]][k]
                percent_increase = (highPrice-currPrice)/currPrice 
                if(THRESHOLD >= 0):
                    if(percent_increase > THRESHOLD ):
                        price = 1
                    else:
                        price = 0
                    df2 = pd.DataFrame({"date": [date],"1or0": [price]}, columns=['date',"1or0"])
                else:
                    if(percent_increase < THRESHOLD ):
                        price = 1
                    else:
                        price = 0
                    df2 = pd.DataFrame({"date": [date],"1or0": [price]}, columns=['date',"1or0"])
            else:
                price=dataframe.iloc[k,1]
                df2 = pd.DataFrame({"date": [date],ATTRIBUTES[0]: [price]}, columns=['date',ATTRIBUTES[0]])

            #seperates to testing and training
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
        if (len(testing) >100) & (len(training) >100):
            try:
                testing.to_csv(TESTINGFOLDER + tic + r'.csv',  index = False)
                training.to_csv(TRAININGFOLDER + tic + r'.csv', index = False) 
            except FileNotFoundError:
                print('creating new folder')
                os.mkdir(TESTINGFOLDER[:-1])
                os.mkdir(TRAININGFOLDER[:-1])
                testing.to_csv(TESTINGFOLDER + tic + r'.csv',  index = False)
                training.to_csv(TRAININGFOLDER + tic + r'.csv', index = False) 

        print(tic)

def printTestStatistics():
    stats.num1sTesting(TESTINGFOLDER)
    if NUMSTOCKS > 100:
        stats.percent_change(ATTRIBUTES,THRESHOLD,numToCheck=NUMSTOCKS)
    else:
        stats.percent_change(ATTRIBUTES,THRESHOLD,days_ahead=DAYSAHEAD,numToCheck=NUMSTOCKS)

if __name__ == "__main__":
    tickers = pd.read_csv('data/stock_names.csv')['Ticker'] #gets stock Tickers 
    if NUMSTOCKS > 0:
        tickers = tickers[:NUMSTOCKS]
    print('num stocks: ', len(tickers))
    print('threshold: ',THRESHOLD)
    print('testing folder: ', TESTINGFOLDER)
    print('training folder: ',TRAININGFOLDER )
    executor = concurrent.futures.ProcessPoolExecutor(20)
    #exportToCSVTestingAndTraining('data/historical_stock_data/' ,tickers[0])
    #runs the update stock tic method for each ticker
    futures = [executor.submit(exportToCSVTestingAndTraining,'data/historical_stock_data/' ,tic,) for tic in tickers]
    concurrent.futures.wait(futures)
    #TO DO: make into function that inputs test folder, attributte, and threshold
    printTestStatistics()
