import math
import numpy as np
import random
import neural_network.activation_functions as af
import time
import concurrent.futures
from multiprocessing import Pool
from functools import partial

class NeuralNet:
    weights = [[]]
    errorCalc = None
    activation_functions = []
    last_output = None
    last_out_val = None
    #time stats

    #Creates a nural network whoes weights are initialized to random values 
    #@param:    activation_funcs    a list with the activation number code corresponding to 
    #                               each layer of nodes
    #@param:    Nodes_per_layer     a list with the number of nodes per layer at each index
    #                               corresponding to each layer of the network
    def __init__(self,activation_funcs, nodes_per_layer):

        #checks gives correct weight info as input
        num_layers = len(nodes_per_layer)
        self.weights =  [np.random.rand(nodes_per_layer[k],nodes_per_layer[k+1])/(nodes_per_layer[k]*nodes_per_layer[k+1])  for k in range(num_layers-1) ]
        
        #checks activation functions
        if len(activation_funcs) == len(self.weights):
            self.activation_functions = activation_funcs
        else:
            print("invalid action functions")
            print(activation_funcs, self.weights)

        #default is error squared
        self.errorCalc = self.sqErrorCalc

    #given an input to the neuralnet calculates the output
    #   currently does not edit the last output class variable 
    #   so that there are no lock issues with parralizations
    def calculateOutput(self, input, rando = None,single_input = False):
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
        if single_input:
            return other[-1][-1]
        return [listToBeReturned, other]

    #determines the change of weights for a specific calculation
    #   no class variable edits for reason above
    #@param     output  -the last output of each node to the nueral network 
    #           including both the before and after activation function values  
    #@param     learnRate   -the learning rate of the algorithm
    #@param     target      -the value for which the ouput is supposed to predict
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
            deltaWeights[layer]  = -1 * learnRate * self.layerWeightPartials(post_nodeValues[layer], tempPartials)
            #calculates the partials for the node inputs for the next iteration
            if(i != numLayers-1):
                #for hidden layers only
                tempPartials = self.nodeDerivatives(pre_nodeValues[layer], self.weights[layer], self.activation_functions[i], tempPartials)
        # #updates weights
        # self.weights =[self.weights[i] + deltaWeights[i] for i in range(numLayers)]
        return deltaWeights

    #changes the weights for a given input delta weights
    def updateWeights(self, deltaWeights):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + deltaWeights[i]

    #helper methods

    #helper methods for gradient descent
    #gets the partial derivatives dE/dw_ij for an individual layers weights
    #TO DO: get this on the GPU
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
    #TO DO: get this on the GPU
    def nodeDerivatives(self, node_values, output_weights, selector, output_derivatives):
        numNodes = len(node_values)
        nodeParitals = np.zeros(numNodes)

        for i in range(numNodes):
            node_input = [node_values[i]]
            activation_deriv = af.func(selector, node_input, deriv = True)[0]
            nodeParitals[i] = activation_deriv * np.dot(output_weights[i], output_derivatives)

        return nodeParitals

    #This 
    def backProp(self, inputs, targets, learnRate = .01, iterations=1):
        if len(targets) != len(inputs):
            print("invalid size combination for inputs and outputs")
        learnRate = learnRate / len(targets)
        error = 0
        output = None
        start_error =0
        end_error = 0
        #creates a step size that takes the number of inputs and divides by 20
        #to split the input data into approximetly 20 parralizable functions to calcuate
        inputs_per_process = int(len(inputs) / 20)
        if len(inputs) % 20 != 0:
            inputs_per_process = inputs_per_process+1
        helper = partial(self.calcWeights, inputs, targets, learnRate,inputs_per_process)
        for k in range(iterations):
            #this creates generates multiple processes to run a back propogation 
            #for the weights. results holds an array of the calculated weight changes
            #for EACH input 
            pool = Pool()
            num_processes = int(len(inputs)/inputs_per_process)
            if num_processes*inputs_per_process < len(inputs):
                num_processes+1
            # print(num_processes)
            # print(inputs_per_process)
            results = pool.map(helper, range(num_processes))
            pool.close()
            pool.join()
            #this sums the individual weight changes
            error = 0
           
            deltaWeights = results[0][0][0]
            for i in range(len(results)):
                error = error + results[i][1]
                for j in range(len(results[i][0])):
                    for l in range(len(results[i][0][j])):
                        deltaWeights[l] = deltaWeights[l] + results[i][0][j][l]
            # print(self.weights)
            self.updateWeights(deltaWeights)
            if k == 0:
                start_error = error
            elif k == iterations-1: 
                end_error = error
            # print(deltaWeights)
            # print(self.weights)
        return [start_error, end_error]
 

    def calcWeights(self, inputs, targets,learnRate, step,j):
        delta = [0]*step
        errort = 0
        for i in range(step):
            if i+j*step >= len(inputs):
                return delta[0:i]
            output = self.calculateOutput(inputs[i+j*step])
            delta[i] = self.gdBackprop(output,learnRate, targets[i+j*step])
            errort = errort + np.abs(targets[i+j*step]- output[-1][-1][-1])
        return [delta, errort]
        

    def test(self, i):
        print("got in", i)
        pass

    def sqErrorCalc(self,target, output, deriv = False):
        if deriv:
            return output - target
        return float(((target - output)**2)/2) 


