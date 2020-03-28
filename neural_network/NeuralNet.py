import math
import numpy as np
import random
import activation_functions as af
import time
import concurrent.futures
from multiprocessing import Pool
from functools import partial
from progressbar import ProgressBar

class NeuralNet:
    weights = [[]]
    errorCalc = None
    activation_functions = []
    last_output = None
    last_out_val = None
    #time stats
    lwp = 0
    nderiv = 0
    def __init__(self,activation_funcs, nodes_per_layer):

        #checks gives correct weight info as input
        num_layers = len(nodes_per_layer)
        self.weights =  [np.random.rand(nodes_per_layer[k],nodes_per_layer[k+1])  for k in range(num_layers-1) ]
        
        #checks activation functions
        if len(activation_funcs) == len(self.weights):
            self.activation_functions = activation_funcs
        else:
            print("invalid action functions")
            print(activation_funcs, self.weights)

        #default is error squared
        self.errorCalc = self.sqErrorCalc

    #given an input to the neuralnet calculates the output
    def calculateOutput(self, input, rando = None):
        b=len(self.weights);
        listToBeReturned=[]
        other=[]
        for a in range(0,b):
            listToBeReturned.append(input);
            if(a == 0):
                other.append(input)
            try:
                output= np.dot(self.weights[a].T, input)
                other.append(output)
            except ValueError: 
                print("Matrices were invalid dimensions or...")
            active= self.activation_functions[a]
            input = af.func(active,output,rando = rando)
 
        listToBeReturned.append(input)
        # self.last_output = [listToBeReturned, other]
        # self.last_out_val = other[-1][-1] 
        return [listToBeReturned, other]

    #determines the change of weights for a specific calculation
    def gdBackprop(self, output,learnRate, target):
        numLayers = len(self.weights)
        deltaWeights = [None]*numLayers
        #seperates the last network output to the various components
        pre_nodeValues = output[0][:-1]
        post_nodeValues = output[1][:-1]
        pre_output = output[0][-1][0]
        output = output[1][-1][0]
        #calculates derivative for final output 
        cost_derivative = af.func(self.activation_functions[numLayers-1],[pre_output],deriv=True)
        tempPartials = [np.dot(cost_derivative , self.errorCalc(target, output, deriv= True))]
        for i in range(numLayers):
            layer = -i-1
            #gets the partials for the nodes 
            #gets the weight changes
            self.lwp = self.lwp -time.time()
            deltaWeights[layer]  = -1 * learnRate * self.layerWeightPartials(post_nodeValues[layer], tempPartials)
            self.lwp = self.lwp + time.time()
            #calculates the partials for the node inputs for the next iteration
            if(i != numLayers-1):
                #for hidden layers only
                self.nderiv = self.nderiv -time.time()
                tempPartials = self.nodeDerivatives(pre_nodeValues[layer], self.weights[layer], self.activation_functions[i], tempPartials)
                self.nderiv = self.nderiv +time.time()
        # #updates weights
        # self.weights =[self.weights[i] + deltaWeights[i] for i in range(numLayers)]
        return deltaWeights

    #changes the weights for a given input delta weights
    def changeWeights(self, newWeights):
        self.weights = newWeights

    #helper methods

    #helper methods for gradient descent
    #gets the partial derivatives dE/dw_ij for an individual layers weights
    def layerWeightPartials(self, inputVals, outputDerivatives):

        rows = len(inputVals)
        cols = len(outputDerivatives)
        weightPartials = np.zeros((rows,cols))
        
        for i in range(rows):
            weightPartials[i]  = np.dot(inputVals[i], outputDerivatives)
        # test print statements
        # print("inputVals")
        # print(inputVals)
        # print("outputDerivatives")
        # print(outputDerivatives)
        # print("weight partials")
        # print(weightPartials)
        return weightPartials

    #gets the partial derivatives dE/dn_i for an individual layers nodes
    def nodeDerivatives(self, node_values, output_weights, selector, output_derivatives):
        numNodes = len(node_values)
        nodeParitals = np.zeros(numNodes)

        for i in range(numNodes):
            node_input = [node_values[i]]
            activation_deriv = af.func(selector, node_input, deriv = True)[0]
            nodeParitals[i] = activation_deriv * np.dot(output_weights[i], output_derivatives)

        return nodeParitals

    def nodeDerivativesHelper(self, node_values, output_weights, selector, output_derivatives, j):
        
        for i in range(j*100, j*100+100):
            if i == len(node_values):
                return
            node_input = [node_values[i]]
            activation_deriv = af.func(selector, node_input, deriv = True)[0]
            val = activation_deriv * np.dot(output_weights[i], output_derivatives)

    def backProp(self, inputs, targets, learnRate = 1, iterations=1000):
        return activation_deriv * np.dot(output_weights[i], output_derivatives)

    def backProp(self, inputs, targets, learnRate = 1, iterations=1000):
        if len(targets) != len(inputs):
            print("invalid size combination for inputs and outputs")
        error = 0
        output = None
        calcTime = 0
        gradientTime = 0
        deltaWeights = [0]*len(inputs)


        runTimeNormal = 0
        runTimeOther = 0

        step = int(len(inputs) / 20)
        if len(inputs) % 20 != 0:
            step = step+1
            
        helper = partial(self.calcWeights, inputs, targets, learnRate,step)
        bar = ProgressBar(maxval = iterations)
        for k in range(iterations):

            startTime = time.time()
            # executor = concurrent.futures.ProcessPoolExecutor(5)
            # #runs the update stock tic method for each ticker
            # futures = [executor.submit(helper, tic) for tic in range(len(inputs))]
            # concurrent.futures.wait(futures)
            
            pool = Pool()
            results = pool.map(helper, range(int(len(inputs)/step)+1))
            pool.close()
            pool.join()
            runTimeOther = runTimeOther + time.time() - startTime 

            error = 0
            runTimeNormal = runTimeNormal - time.time()
            for i in range(len(inputs)):
                startTime = time.time()
                output = self.calculateOutput(inputs[i])
                calcTime = calcTime + time.time() -startTime
                startTime = time.time()
                deltaWeights[i] = self.gdBackprop(output,learnRate, targets[i])
                gradientTime = gradientTime + time.time() - startTime
                error = error + self.sqErrorCalc(targets[i],output[-1][-1][-1])/16
                output = output
            runTimeNormal = runTimeNormal + time.time()
            # TO DO: average and update weights together
            

            if k == 0:
                print("start: ", error)   
                bar.start() 
                # print("multi: ", results[0][0])
                # print("single: ", deltaWeights[0])
                # print(deltaWeights)
            bar.update(k)
        bar.finish()

        print("calc time: ", calcTime)
        print("gradient time: ", gradientTime)       
        print("end: ", error)
        print("run time normal: ", runTimeNormal)
        print("runt time parrallel: ", runTimeOther)
        print("lws: ", self.lwp)
        print("nderiv: ", self.nderiv)
        # print(error)
        # print(targets)
        # print(output[-1])

    def calcWeights(self, inputs, targets,learnRate, step,j):
        delta = [0]*step
        for i in range(step):
            if i+j*step >= len(inputs):
                return delta[0:i]
            output = self.calculateOutput(inputs[i+j*step])
            delta[i] = self.gdBackprop(output,learnRate, targets[i+j*step])
        return delta
        

    def test(self, i):
        print("got in", i)
        pass

    def sqErrorCalc(self,target, output, deriv = False):
        if deriv:
            return output - target
        return float(((target - output)**2)/2) 


    def getWeightsAverageKernel(self,deltaWeights,curr):
        avgDelta = curr
        try:
            for i in range(len(curr)):
                avgDelta[i] = self.getWeightsAverageKernel()
        except expression as identifier:
            pass

