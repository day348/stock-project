import numpy as np
import pandas as pd
import os
import concurrent.futures
from random import random

#Pass path and tic of normalized data
def splitCSVData(path,tic):
    if os.path.exists( path + tic + r'.csv'):
        datav = pd.read_csv(path + tic + r'.csv', index_col=False)
        x= len(data)
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
        testing=dataSplit[0]
        training=dataSplit[1]
        testing.to_csv(path + r'testing' + tic + r'.csv',  index = False)
        training.to_csv(path + r'training' + tic + r'.csv', index = False) 