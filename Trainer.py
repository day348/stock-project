import neural_network.NeuralNet as nn
import pandas as pd 
import os
from data.getInputs import getInputs
#setup
NODES_PER_LAYER = [380,100,1]
ACTIVATION_FUNCTIONS = [7,7]
"""
0 : TanH        1 : arcTan      2 : Elu
3 : Identity    4 : LeakyRelu   5 : RandomRelu  
6 : Relu        7 : Sigmoid     8 : SoftPlus
9 : Step
"""
if __name__ == "__main__":



    #create neural network
    network = nn.NeuralNet(ACTIVATION_FUNCTIONS, NODES_PER_LAYER)
    #get tickers 
    stock_tickers = pd.read_csv('data\\stock_names.csv')['Ticker']
    for tic in stock_tickers:
        if not os.path.exists('data\\training\\' + tic + '.csv'):
            stock_tickers = stock_tickers[stock_tickers != tic]


    #run this for every stock 
    #store in the 
    
    #get input vectors for a specific stock 
    tic = stock_tickers[0]

    #gets the dates and the assosiated close values
    output_values = pd.read_csv('data\\training\\' + tic + '.csv')
    print(len(output_values))
    #creates an array of input vectors for a given stock and the training days
    input_values = [0]*len(output_values.index)
    data = pd.read_csv('data\\normalized_data\\' + tic + '.csv')
    data = data.set_index('date')
    for i in range(len(output_values.index)):
        date = output_values.iloc[i]['date']
        input = getInputs(tic,date,data)
        #catches error if not enough previous days
        # if input == -1:
        #     output_values.drop(date)
        input_values[i] = input

        
    network.backProp(input_values,output_values['close'].to_numpy())

 