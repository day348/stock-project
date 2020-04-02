import neural_network.NeuralNet as nn
from data.inputsForBackProp import inputsForBackProp
from multiprocessing import Pool
from progressbar import ProgressBar
import tqdm
import sys
import time
import pandas as pd 
import os

#setup
NODES_PER_LAYER = [380,100,100,50,20,1]
ACTIVATION_FUNCTIONS = [7,7,7,4,4]
NUM_ITERATIONS = 1
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
    #removes tickers wihtout data
    for tic in stock_tickers:
        if not os.path.exists('data\\training\\' + tic + '.csv'):
            stock_tickers = stock_tickers[stock_tickers != tic]
        if not os.path.exists('data\\normalized_data\\' + tic + '.csv'):
            stock_tickers = stock_tickers[stock_tickers != tic]
      
    print("loading training data")
    #get input training dictionaries 
    loading_time = -time.time()
    #spawns a process for each stock data that needs to be loaded 
    pool = Pool()
    results = list(tqdm.tqdm(pool.imap(inputsForBackProp, stock_tickers), total=len(stock_tickers)))
    pool.close()
    pool.join()
    inputs = {}
    outputs = {}
    for i in range(len(results)):
        inputs.update(results[i][0])
        outputs.update(results[i][1])
    loading_time = loading_time+time.time()
    print("time spent loading: ", loading_time)

    print()
    print("Begining Training with ", NUM_ITERATIONS, " iterations")
    print("\tnodes per layer: ", NODES_PER_LAYER)
    print("\tactivation functions: ", ACTIVATION_FUNCTIONS)
    print()

    print("time test on AAC")
    print(len(inputs['AAC']), len(outputs['AAC']))
    backPropTime = -time.time()
    network.backProp(inputs['AAC'], outputs['AAC'])
    backPropTime = backPropTime + time.time()
    print('time taken: ', backPropTime)


    start_error = 0
    end_error = 0
    pbar = ProgressBar()
    for i in pbar(range(NUM_ITERATIONS)):
        error = 0
        for tic in stock_tickers:
            print(tic,len(inputs[tic]), len(outputs[tic]))
            backPropTime = -time.time()
            error = error + network.backProp(inputs[tic], outputs[tic])[0]
            backPropTime = backPropTime + time.time()
            print('time taken: ', backPropTime)

        #error editors and prints
        print('prediction error on iteration ', i, ': ', error)
        if i == 0: 
            start_error = error
        elif i ==NUM_ITERATIONS -1:
            end_error = error
        