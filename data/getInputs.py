import numpy as np

def getInputs(tic, date, inputData): 
    listOfDates= list(inputData.index)
    listOfNums=[x for x in range(0,len(listOfDates))]
    datesToNums= dict(zip(listOfDates, listOfNums))
    new_dict = dict([(value, key) for key, value in datesToNums.items()]) 
    beginningIndex= datesToNums[date]
    endIndex=beginningIndex-20
    if(endIndex<0):
        return -1
    relevantDates=[]
    for i in range(endIndex, beginningIndex):
        relevantDates.append(new_dict[i])
    output=[]
    for i in relevantDates:
        output.append(inputData['low'][i])
        output.append(inputData['high'][i])
        output.append(inputData['average'][i])
        output.append(inputData['volume'][i])
        output.append(inputData['close'][i])
        output.append(inputData['open'][i])
        output.append(inputData['range'][i])
        output.append(inputData['twelveDay'][i])
        output.append(inputData['twentySixDay'][i])
        output.append(inputData['volumeEMA'][i])
        output.append(inputData['singleDay'][i])
        output.append(inputData['dayToDay'][i])
        output.append(inputData['fiftyTwoDayHigh'][i])
        output.append(inputData['fiftyTwoWeekHigh'][i])
        output.append(inputData['fiftyTwoDayLow'][i])
        output.append(inputData['fiftyTwoWeekLow'][i])
        output.append(inputData['fiftyTwoWeekAverage'][i])
        output.append(inputData['fiftyTwoDayStandDev'][i])
        output.append(inputData['fiftyTwoWeekStandDev'][i])
    return np.array(output)
    