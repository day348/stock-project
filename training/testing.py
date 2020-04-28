from progressbar import ProgressBar
import neural_network.NeuralNet as nn
import numpy as np
import pandas as pd

def test(network,inputs,outputs,stock_tickers):
    testing_errors = [0]*len(stock_tickers)
    j = -1
    pbar = ProgressBar()
    counters = [0]*6
    for tic in stock_tickers:
        j = j+1
        for i in range(len(inputs[tic])):
            try:
                prediction = network.calculateOutput(inputs[tic][i],single_input=True)
            except:
                print("tic: ",tic)
                print("input:")
                print(inputs[tic][i])
            goal = outputs[tic][i]
            counters[5] = counters[5] + 1
            if goal == 1:
                counters[0] = counters[0] + 1
                if prediction > .75:
                    counters[1] = counters[1] + 1
                elif prediction < .25:
                    counters[4] = counters[4] + 1
            else:
                print("got in")
                if prediction < .25:
                    counters[3] = counters[3] + 1
                elif prediction > .75:
                    counters[2] = counters[2] + 1
            #print("goal: ", goal, " Prediction: ", prediction)
            testing_errors[j] = testing_errors[j] + np.abs(prediction - goal)
            #TO DO: amount in the right direction

        testing_errors[j] = testing_errors[j]*100/len(inputs[tic])
        #print('Error for ',tic, ': ', testing_errors[j]*100, '%')
    counters[0] = float(counters[0]) / float(counters[5])
    print('total testing average error: ', np.average(testing_errors), '%')
    avg_test_error = np.average(testing_errors)
    temp = pd.DataFrame({'testing errors':testing_errors})
    testing_errors = temp.join(stock_tickers)
    print(counters)
    return testing_errors.set_index('Ticker'), avg_test_error, counters