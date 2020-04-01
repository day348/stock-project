import neural_network.NeuralNet as nn
import pandas as pd 
import os

#setup
NODES_PER_LAYER = [10,100,1]
ACTIVATION_FUNCTIONS = [7,7]
"""
0 : TanH        1 : arcTan      2 : Elu
3 : Identity    4 : LeakyRelu   5 : RandomRelu  
6 : Relu        7 : Sigmoid     8 : SoftPlus
9 : Step
"""
#create neural network
network = nn.NeuralNet(ACTIVATION_FUNCTIONS, NODES_PER_LAYER)
#get tickers 
stock_tickers = pd.read_csv('data\\stock_names.csv')['Ticker']
for tic in stock_tickers:
    if not os.path.exists('data\\normalized_data\\' + tic + '.csv'):
        stock_tickers = stock_tickers[stock_tickers != tic]

#get input vectors for a specific stock 
tic = stock_tickers[0]

#gets the dates and the assosiated close values
output_values = pd.read_csv('data\\normalized_data\\' + tic + '.csv')
#creates an array of input vectors for a given stock and the training days
input_values = [0]*len(output_values.index)
for i in range(len(output_values.index)):
    date = output_values.index[i]
    input = get_inputs(tic,date)
    #catches error if not enough previous days
    if input == -1:
        output_values.drop(date)
    input_values[i] = get_inputs(tic, date)


 

def get_inputs(tic, date):
    pass