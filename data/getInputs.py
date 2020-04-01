def getInputs(tic, date, globe, datesToNums):
    listOfDates= list(vals.index)
    listOfNums=[x for x in range(0,2889)]
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
        output.append(globe['low'][i])
        output.append(globe['high'][i])
        output.append(globe['average'][i])
        output.append(globe['volume'][i])
        output.append(globe['close'][i])
        output.append(globe['open'][i])
        output.append(globe['range'][i])
        output.append(globe['twelveDay'][i])
        output.append(globe['twentySixDay'][i])
        output.append(globe['volumeEMA'][i])
        output.append(globe['singleDay'][i])
        output.append(globe['dayToDay'][i])
        output.append(globe['fiftyTwoDayHigh'][i])
        output.append(globe['fiftyTwoWeekHigh'][i])
        output.append(globe['fiftyTwoDayLow'][i])
        output.append(globe['fiftyTwoWeekLow'][i])
        output.append(globe['fiftyTwoWeekAverage'][i])
        output.append(globe['fiftyTwoDayStandDev'][i])
        output.append(globe['fiftyTwoWeekStandDev'][i])
    return np.array(output)
    