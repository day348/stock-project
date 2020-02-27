import random
import numpy as np
import matrixMult as net
import gradientDescent as gd
import progressbar
import pandas as pd

numIterations = 1000
errorAll = [[0]*24]*91
ra = 0
errorStart = 0;

# progress bar init
bar = progressbar.ProgressBar(maxval=numIterations, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()

for p in range(10,100):
    for q in range(2,25):
        print(q)
        #generates test outputs
        data = [ [ i*j for i in range(4) ] for j in range(4) ] 
        #generates random wieghts
        weights = [[ [ random.randint(-2,2) for i in range(p) ] for j in range(2) ],
                    [ [ random.randint(-2,2) for i in range(q) ] for j in range(p) ],
                    [ [ random.randint(-2,2) for i in range(1) ] for j in range(q) ]]
        #progress bar init
        # bar = progressbar.ProgressBar(maxval=numIterations, \
        #     widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])


        for k in range(numIterations):
            delt = [0] * len(weights)
            error = 0
            for i in range(4):
                for j in range(4):
                    output = net.calculateOutput( [4,4,4], weights, [i,j], 0)
                    deltTemp = gd.gdBackprop(weights, output, net.derivLeakyReLU, net.derivLeakyReLU, (.1/(p+q)), data[i][j])
                    error = error + (gd.errorCalc(data[i][j],output[1][-1][0]) / 16)
                    for l in range(len(delt)):
                        delt[l] = delt[l] + (deltTemp[l] / 16)
            
            # if(k == 0):
            #     print("average error:" )
            #     print(error)
            #     bar.start()
            # bar.update(k+1)
            if(k==(numIterations -1)):
                # bar.finish()
                # print("average error:" )
                # print(error)
                errorAll[p][q] = error
            

            newWeights = weights
            for i in range(len(weights)):
                newWeights[i] = weights[i] + delt[i]
            weights = newWeights

        bar.update(ra+1)
        ra = ra +1
        # print("final")
        # out =[[net.calculateOutput( [4,4,4], weights, [i,j], 0)[1][-1][0] for i in range(4)] for j in range(4)]
        # print(out)
        # print("actual")
        # print(data)
        # print(gd.errorCalc(out[1][2], data[1][2]))

bar.finish()
df = pd.DataFrame.from_records(errorAll)
df.to_csv('test_data\\simpletests_Relu.csv', index = False)
