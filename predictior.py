import pandas as pd
import numpy as np
import neural_network.NeuralNet as nn

RAW_DATA_FOLDER = 'data/historical_stock_data/'
WEIGHTS_FILE_PATH = 'test_results/Great+03/end_weights.npy'

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
    for i in relevantDates[-20:]:
        output.append(inputData['close'][i])
        if i == relevantDates[-1]:
            output.append(inputData['range'][i])
            output.append(inputData['singleDay'][i])
            output.append(inputData['dayToDay'][i])
            output.append(inputData['low'][i])
            output.append(inputData['high'][i])
            output.append(inputData['average'][i])
            output.append(inputData['open'][i])
            if inputData['dayToDay'][i] > 0:
                output.append(1)
            else:
                output.append(0)
            if inputData['twelveDay'][i]- inputData['twelveDay'][relevantDates[-2]] > 0:
                output.append(1)
            else:
                output.append(0)
            if inputData['twentySixDay'][i]- inputData['twentySixDay'][relevantDates[-2]] > 0:
                output.append(1)
            else:
                output.append(0)
            output.append(inputData['volume'][i] - inputData['volume'][relevantDates[-2]])
            output.append(inputData['twelveDay'][i]- inputData['twelveDay'][relevantDates[-2]])
            output.append(inputData['twentySixDay'][i]- inputData['twentySixDay'][relevantDates[-2]])
            output.append(inputData['volumeEMA'][i])
            output.append(inputData['fiftyTwoDayHigh'][i])
            output.append(inputData['fiftyTwoWeekHigh'][i])
            output.append(inputData['fiftyTwoDayLow'][i])
            output.append(inputData['fiftyTwoWeekLow'][i])
            output.append(inputData['fiftyTwoWeekAverage'][i])
            output.append(inputData['fiftyTwoDayStandDev'][i])
            output.append(inputData['fiftyTwoWeekStandDev'][i])
    return np.array(output)
def percentChange(low,high, value):
    if high-low!=0 :
        div = (value - low)/(high - low)
        """ if div < 0:
            print("high: ", high, " low: ", low, " value", value)
            raise Exception """
        return 2*div-1
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
def normalize(data):
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
    for i in range(len(data['high'])):
        if(i <= 10):
            lowIndex = 0
        else:
            lowIndex = i-10

        priceLow = data['low'][lowIndex:i+1].min()
        priceHigh = data['high'][lowIndex:i+1].max()
        volLow = data['volume'][lowIndex:i+1].min()
        volHigh = data['volume'][lowIndex:i+1].max()
        volemaLow = data['volume ema'][lowIndex:i+1].min()
        volemaHigh = data['volume ema'][lowIndex:i+1].max()
        rangeLow = data['range'][lowIndex:i+1].min()
        rangeHigh = data['range'][lowIndex:i+1].max()
        sdayLow = data['single_day_change'][lowIndex:i+1].min()
        sdayHigh = data['single_day_change'][lowIndex:i+1].max()
        ddayLow = data['day_to_day_change'][lowIndex:i+1].min()
        ddayHigh = data['day_to_day_change'][lowIndex:i+1].max()
        deviationLow = data['52 week standard deviation'][lowIndex:i+1].min()
        deviationHigh = data['52 week standard deviation'][lowIndex:i+1].max()
        ddeviationLow = data['52 day standard deviation'][lowIndex:i+1].min()
        ddeviationHigh = data['52 day standard deviation'][lowIndex:i+1].max()
        pwLow = data['52 week low'][lowIndex:i+1].min()
        pwHigh = data['52 week high'][lowIndex:i+1].max()
        emaLow = data['26 day ema'][lowIndex:i+1].min()
        emaHigh = data['26 day ema'][lowIndex:i+1].max()

        low[i]=percentChange(priceLow,priceHigh, data['low'][i])
        high[i]=percentChange(priceLow,priceHigh, data['high'][i])
        average[i]=percentChange(priceLow,priceHigh, data['average'][i])
        close[i]=percentChange(priceLow,priceHigh, data['close'][i])
        openn[i]=percentChange(priceLow,priceHigh, data['open'][i])
        rangee[i]=percentChange(rangeLow,rangeHigh, data['range'][i])
        twelveDay[i]=percentChange(priceLow,priceHigh, data['12 day ema'][i])
        twentySixDay[i]=percentChange(priceLow,priceHigh, data['26 day ema'][i])
        volume[i]=percentChange(volLow,volHigh,data['volume'][i])
        volumeEMA[i]=percentChange(volemaLow,volemaHigh,data['volume ema'][i])
        singleDay[i]=percentChange(sdayLow,sdayHigh, data['single_day_change'][i])
        dayToDay[i]=percentChange(ddayLow,ddayHigh, data['day_to_day_change'][i])
        fiftyTwoDayHigh[i]=percentChange(pwLow,pwHigh, data['52 day high'][i])
        fiftyTwoWeekHigh[i]=percentChange(pwLow,pwHigh, data['52 week high'][i])
        fiftyTwoDayLow[i]=percentChange(pwLow,pwHigh, data['52 day low'][i])
        fiftyTwoWeekLow[i]=percentChange(pwLow,pwHigh, data['52 week low'][i])
        fiftyTwoWeekAverage[i]=percentChange(pwLow,pwHigh, data['52 week average'][i])
        fiftyTwoDayStandDev[i]=percentChange(ddeviationLow,ddeviationHigh, data['52 day standard deviation'][i])
        fiftyTwoWeekStandDev[i]=percentChange(deviationLow,deviationHigh, data['52 week standard deviation'][i])
        """ if low[i] < 0:
            print("high: ", priceHigh, " low: ", priceLow, " value", data['low'][i], " output: ", low[i])
            raise Exception """
    listOfDates= data['date']
    listOfVols= data['52 week volume average']
    listOfAverages= data['52 day average']
    vals= [low, high, average, volume, close, openn, rangee, twelveDay, twentySixDay,volumeEMA, singleDay, dayToDay,fiftyTwoDayHigh,fiftyTwoWeekHigh,fiftyTwoDayLow,fiftyTwoWeekLow,fiftyTwoWeekAverage,fiftyTwoDayStandDev,fiftyTwoWeekStandDev]  
    v=transform(vals)
    rows=listOfDates
    cols=["low", "high", "average", "volume", "close", "open", "range", "twelveDay", "twentySixDay","volumeEMA", "singleDay", "dayToDay","fiftyTwoDayHigh","fiftyTwoWeekHigh","fiftyTwoDayLow","fiftyTwoWeekLow","fiftyTwoWeekAverage","fiftyTwoDayStandDev","fiftyTwoWeekStandDev"] 
    dataframe=pd.DataFrame(data=v, index=rows, columns=cols)
    return dataframe

def prediction_calc(tic,data=None):
    if data == None:
        data = pd.read_csv(RAW_DATA_FOLDER + tic + '.csv').tail(30)
    data.index = range(len(data))
    data = normalize(data)
    date = data.index[-1]
    print(date)
    inputs = getInputs(date,data)
    weights = np.load(WEIGHTS_FILE_PATH,allow_pickle=True)
    network = nn.NeuralNet([0,0,7],[41,100,50,1])
    network.weights = weights
    prediction = network.calculateOutput(inputs,single_input=True)
    if prediction > .5:
        confidence = ((prediction-.5)/.4)*100
        return (1,confidence)
    else:
        confidence = np.sqrt((.5-prediction)/.4)*100
        return (0,confidence)
    

print(prediction_calc('MSFT'))