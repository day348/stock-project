from progressbar import ProgressBar
import neural_network.NeuralNet as nn
import numpy as np
import pandas as pd

def test(network,inputs,outputs,stock_tickers):
    testing_errors = [0]*len(stock_tickers)
    j = -1
    pbar = ProgressBar()
    for tic in stock_tickers:
        j = j+1
        for i in range(len(inputs[tic])):
            prediction = network.calculateOutput(inputs[tic][i],single_input=True)
            goal = outputs[tic][i]
            testing_errors[j] = testing_errors[j] + np.abs(prediction - goal)
            #TO DO: amount in the right direction

        testing_errors[j] = testing_errors[j]*100/len(inputs[tic])
        #print('Error for ',tic, ': ', testing_errors[j]*100, '%')
    print('total testing average error: ', np.average(testing_errors), '%')
    avg_test_error = np.average(testing_errors)
    temp = pd.DataFrame({'testing errors':testing_errors})
    testing_errors = temp.join(stock_tickers)
    return testing_errors.set_index('Ticker')