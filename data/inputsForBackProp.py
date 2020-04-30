import pandas as pd
import numpy as np
from progressbar import ProgressBar
import time
import tqdm
from multiprocessing import Pool
import tqdm

def getInputs(date, inputData): 
    listOfDates= list(inputData.index)
    listOfNums=[x for x in range(0,len(listOfDates))]
    datesToNums= dict(zip(listOfDates, listOfNums))
    new_dict = dict([(value, key) for key, value in datesToNums.items()]) 
    try:
        beginningIndex= datesToNums[date]
    except:
        return -1
    endIndex=beginningIndex-20
    if(endIndex<0):
        return -1
    relevantDates=[]
    for i in range(endIndex, beginningIndex):
        relevantDates.append(new_dict[i])
    output=[]
    for i in relevantDates[-1:]:
        output.append(inputData['close'][i])
        output.append(inputData['range'][i])
        """ output.append(inputData['singleDay'][i])
        output.append(inputData['dayToDay'][i]) """
        if i == relevantDates[-1]:
            output.append(inputData['low'][i])
            output.append(inputData['high'][i])
            """ output.append(inputData['average'][i])
            output.append(inputData['open'][i]) """
            output.append(inputData['volume'][i])
            output.append(inputData['twelveDay'][i])
            output.append(inputData['twentySixDay'][i])
            """ output.append(inputData['volumeEMA'][i])
            output.append(inputData['fiftyTwoDayHigh'][i])
            output.append(inputData['fiftyTwoWeekHigh'][i])
            output.append(inputData['fiftyTwoDayLow'][i])
            output.append(inputData['fiftyTwoWeekLow'][i])
            output.append(inputData['fiftyTwoWeekAverage'][i])
            output.append(inputData['fiftyTwoDayStandDev'][i])
            output.append(inputData['fiftyTwoWeekStandDev'][i]) """
    return np.array(output)
    
#given a set of tickers it returns two dictionaries one mapping 
#a stock to its input vectors and the other mapping a stock to 
#its ouput values
def inputsForBackProp(tics):
    inputters=[]
    outputters=[]
    for tic in [tics]:
        #gets the dates and the assosiated close values
        output_values = pd.read_csv('data/training/' + tic + '.csv')
        #creates an array of input vectors for a given stock and the training days
        input_values = [0]*len(output_values.index)
        data = pd.read_csv('data/normalized_data/' + tic + '.csv')
        data = data.set_index('date')
        i = 0
        for date in output_values.date:
            input = getInputs(date,data)
            #catches error if not enough previous days
            if type(input) ==type(-1):
                output_values = output_values[output_values.date != date]
                input_values = input_values[:-1]
            else:
                input_values[i] = input
                i = i + 1
            

        inputters.append(input_values)
        outputters.append(output_values['Up or Down'].to_numpy())
    #map tics to their respective lists of inputs and outputs
    input_dict=dict(zip([tics], inputters))
    output_dict=dict(zip([tics],outputters))
    #return list of both dicts
    return [input_dict, output_dict]
    
#given a set of tickers it returns two dictionaries one mapping 
#a stock to its input vectors and the other mapping a stock to 
#its ouput values
def inputsForTesting(tics):
    inputters=[]
    outputters=[]
    for tic in [tics]:
        #gets the dates and the assosiated close values
        output_values = pd.read_csv('data/testing/' + tic + '.csv')
        #creates an array of input vectors for a given stock and the training days
        input_values = [0]*len(output_values.index)
        data = pd.read_csv('data/normalized_data/' + tic + '.csv')
        data = data.set_index('date')
        i = 0
        for date in output_values.date:
            input = getInputs(date,data)
            #catches error if not enough previous days
            if type(input) ==type(-1):
                output_values = output_values[output_values.date != date]
                input_values = input_values[:-1]
            else:
                input_values[i] = input
                i = i + 1
            

        inputters.append(input_values)
        outputters.append(output_values['Up or Down'].to_numpy())
    #map tics to their respective lists of inputs and outputs
    input_dict=dict(zip([tics], inputters))
    output_dict=dict(zip([tics],outputters))
    #return list of both dicts
    return [input_dict, output_dict]

def load_inputs(tickers, testing = False):
    #get input training dictionaries 
    #spawns a process for each stock data that needs to be loaded 
    pool = Pool()
    if not testing:
        results = list(tqdm.tqdm(pool.imap(inputsForBackProp, tickers), total=len(tickers)))
    else:
        results = list(tqdm.tqdm(pool.imap(inputsForTesting, tickers), total=len(tickers)))
    pool.close()
    pool.join()
    inputs = {}
    outputs = {}
    for i in range(len(results)):
        inputs.update(results[i][0])
        outputs.update(results[i][1])
    return inputs, outputs