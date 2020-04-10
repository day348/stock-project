import neural_network.NeuralNet as nn
from data.inputsForBackProp import inputsForBackProp
from data.inputsForBackProp import inputsForTesting
from multiprocessing import Pool
from progressbar import ProgressBar 
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import sys
import time
import pandas as pd 
import os

#setup
NODES_PER_LAYER = [380,100,50,50,20,1]
ACTIVATION_FUNCTIONS = [4,4,4,4,3]
NUM_ITERATIONS = 10
LEARN_RATE = 10

"""
0 : TanH        1 : arcTan      2 : Elu
3 : Identity    4 : LeakyRelu   5 : RandomRelu  
6 : Relu        7 : Sigmoid     8 : SoftPlus
9 : Step
"""

if __name__ == "__main__":
    #counters
    numPredictions = 0
    training_errors = [0]*NUM_ITERATIONS
    testing_errors = None
    start_weights = None
    end_weights = None
    time_per_stock = 0
    time_per_iteration = 0
    overview_string = ''

    #create neural network
    network = nn.NeuralNet(ACTIVATION_FUNCTIONS, NODES_PER_LAYER)
    #get tickers 
    stock_tickers = pd.read_csv('data/stock_names.csv')['Ticker']
    #removes tickers wihtout data
    for tic in stock_tickers:
        if not os.path.exists('data/training/' + tic + '.csv'):
            stock_tickers = stock_tickers[stock_tickers != tic]
        if not os.path.exists('data/normalized_data/' + tic + '.csv'):
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
    #print updates
    print("time spent loading: ", loading_time)
    print()
    print("Begining Training with ", NUM_ITERATIONS, " iterations")
    print("\tnodes per layer: ", NODES_PER_LAYER)
    print("\tactivation functions: ", ACTIVATION_FUNCTIONS)
    print("\tnumber of stocks: ", len(stock_tickers))
    print()

    #this loop trains the neural network
    start_weights = network.weights
    pbar = ProgressBar()
    for i in pbar(range(NUM_ITERATIONS)):
        training_errors[i] = 0
        time_taken = 0
        for tic in stock_tickers:
            #print(tic,len(inputs[tic]), len(outputs[tic]))
            backPropTime = -time.time()
            #back propigation call
            try:
                training_errors[i] = training_errors[i] + (network.backProp(inputs[tic], outputs[tic], learnRate=LEARN_RATE)[0]/len(outputs[tic]))
            except:
                print("failed on iteration ", i, ' on stock ', tic)
                print(network.weights)
            backPropTime = backPropTime + time.time()
            #debugger test prints
            """ prediction = network.calculateOutput(inputs[tic][-10])[-1][-1][0]
            goal = outputs[tic][-10]
            print('time taken: ', backPropTime)
            print('recent prediction, actual: ',goal*100,'% prediction:', prediction*100, '%') """
            #counters
            time_taken = time_taken + backPropTime
        #counters
        time_per_iteration = time_taken + time_per_iteration
        training_errors[i] = (training_errors[i] / len(stock_tickers.index))*100
        #error editors and prints
        print('average prediction error on iteration', i+1, ':',training_errors[i], '%')
    time_per_iteration = time_per_iteration / NUM_ITERATIONS
    time_per_stock = time_per_iteration / len(stock_tickers)
    end_weights = network.weights




    #gets testing data
    print("\nloading testing data")
    #get input testing dictionaries 
    loading_time = -time.time()
    #spawns a process for each stock data that needs to be loaded 
    pool = Pool()
    results = list(tqdm.tqdm(pool.imap(inputsForTesting, stock_tickers), total=len(stock_tickers)))
    pool.close()
    pool.join()
    inputs = {}
    outputs = {}
    for i in range(len(results)):
        inputs.update(results[i][0])
        outputs.update(results[i][1])
    print('\nStarting testing\n')
    # This loop tests the neural network
    testing_errors = [0]*len(stock_tickers)
    j = -1
    pbar = ProgressBar()
    for tic in stock_tickers:
        j = j+1
        for i in range(len(inputs[tic])):
            prediction = network.calculateOutput(inputs[tic][i],single_input=True)
            goal = outputs[tic][i]
            testing_errors[j] = testing_errors[j] + np.abs(prediction - goal)
        testing_errors[j] = testing_errors[j]*100/len(inputs[tic])
        #print('Error for ',tic, ': ', testing_errors[j]*100, '%')
    print('total testing average error: ', np.average(testing_errors), '%')
    avg_test_error = np.average(testing_errors)
    temp = pd.DataFrame({'testing errors':testing_errors})
    testing_errors = temp.join(stock_tickers)
    testing_errors.set_index('Ticker')
    print('\nSaving Results...')
    #saves results
    #creates new test folder
    folder_created = False
    test_folder = None
    test_num = 0
    while not folder_created:
        test_num = test_num + 1
        try:
            test_folder = 'test_results/test' + str(test_num)
            os.mkdir(test_folder)
            folder_created = True
        except FileExistsError:
            pass
    #saves start and end wieghts
    np.save(test_folder + '/start_weights',start_weights)
    np.save(test_folder + '/end_weights',end_weights)
    #saves testing and training data 
    np.save(test_folder + '/training_error_progression', training_errors)
    testing_errors.to_csv(test_folder + '/testing_errors.csv')
    #creates and saves overview
    overview_string = overview_string + 'This training aims to predict the day to day change of stocks\n'
    overview_string = overview_string + "\ttraining with " +  str(NUM_ITERATIONS) + " iterations\n"
    overview_string = overview_string + "\tnodes per layer: " + str(NODES_PER_LAYER) + '\n'
    overview_string = overview_string + "\tactivation functions: " +  str(ACTIVATION_FUNCTIONS) + "\n"
    overview_string = overview_string + "\tnumber of stocks: " + str(len(stock_tickers)) + '\n'
    overview_string = overview_string + '\nStatistics:\n'
    overview_string = overview_string + '\tbackprop time per stock= ' + str(time_per_stock) + '\n'
    overview_string = overview_string + '\tbackprop time per iteration = ' + str(time_per_iteration) + '\n'
    overview_string = overview_string + '\tstart training error = ' + str(training_errors[0])+ '%\n'
    overview_string = overview_string + '\tend training error = ' + str(training_errors[-1])+ '%\n'
    overview_string = overview_string + "\taverage testing error = " + str(avg_test_error)+ "%\n"
    #write file
    overview_file = open(test_folder + '/results_overview', "w")
    overview_file.write(overview_string)
    overview_file.close()
    #plot training progression
    plt.plot(range(len(training_errors)), training_errors)
    plt.xlabel("iteration")
    plt.ylabel('% Error')
    plt.savefig(test_folder + '/training_progression.png')
    print('done!')
