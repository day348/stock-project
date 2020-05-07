from progressbar import ProgressBar
import neural_network.NeuralNet as nn
import numpy as np
import pandas as pd

def test(network,inputs,outputs,stock_tickers):
    testing_errors = [0]*len(stock_tickers)
    j = -1
    pbar = ProgressBar()
    counters = [0]*6
    num1 = 0
    num0 = 0
    for tic in stock_tickers:
        j = j+1
        for i in range(len(inputs[tic])):
            try:
                prediction = network.calculateOutput(inputs[tic][i],single_input=True)
                """ if sum(inputs[tic][i]) > 2:
                    prediction = 1
                else:
                    prediction = 0 """
            except:
                print("tic: ",tic)
                print("input:")
                print(inputs[tic][i])
            goal = outputs[tic][i]
            counters[5] = counters[5] + 1
            if goal == 1:
                counters[0] = counters[0] + 1
                if prediction > .5:
                    counters[1] = counters[1] + 1
                    num1 = num1 +1 
                elif prediction < .5:
                    num0 = num0 +1 
                    counters[4] = counters[4] + 1
            else:
                if prediction < .5:
                    counters[3] = counters[3] + 1
                    num0 = num0 +1 
                elif prediction > .5:
                    num1 = num1 +1 
                    counters[2] = counters[2] + 1
            if i == 10:
                pass
                #print(prediction)
            #print("goal: ", goal, " Prediction: ", prediction)
            testing_errors[j] = testing_errors[j] + np.abs(prediction - goal)
            #TO DO: amount in the right direction
            #checks convergenct
        """ if(num0 > len(inputs[tic])*.95 or num1 > len(inputs[tic])*.95 ):
            print("converged to constant solution. num 0:", num0, " num 1: ", num1) """
        if len(inputs) != 0:
            testing_errors[j] = testing_errors[j]*100/len(inputs[tic])
        else:
            testing_errors = -1
        #print('Error for ',tic, ': ', testing_errors[j]*100, '%')
    counters[0] = float(counters[0]) / float(counters[5])
    #print('total testing average error: ', np.average(testing_errors), '%')
    avg_test_error = np.average(testing_errors)
    temp = pd.DataFrame({'testing errors':testing_errors})
    testing_errors = temp.join(stock_tickers)
    #print(counters)
    return testing_errors.set_index('Ticker'), avg_test_error, counters