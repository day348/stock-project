import math
import numpy as np
import random
import activation_functions as af
class neuralnet:
    weights = [[]]
    errorCalc = None
    activation_functions = []
    last_output = None

    def __init__(self,activation_funcs ,input_weigths=-1, nodes_per_layer=-1):

        #checks gives correct weight info as input
        if (type(input_weigths) == type(-1)) & (type(nodes_per_layer) == type(-1)):
                print("must either give nodes per layer or input weights")
        elif type(input_weigths) == type(-1):
            num_layers = len(nodes_per_layer)
            self.weights =  [[ [ random.randint(-2,2) for i in range(nodes_per_layer[k+1]) ] for j in range(nodes_per_layer[k]) ] for k in range(num_layers-1) ]
        else:
            self.weights = input_weigths
        #checks activation functions
        if len(activation_funcs) == len(self.weights)+1:
            self.activation_functions = activation_funcs
        else:
            print("invalid action functions")

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
                output= self.matrixMultiplication(self.weights[a], input)
                other.append(output)
            except ValueError: 
                print("Matrices were invalid dimensions or...")
            active= self.activation_functions[a]
            input = af.func(active,output,rando = rando)
 
        listToBeReturned.append(input)
        self.last_output = [listToBeReturned, other] 
        return self.last_output

    #determines the change of weights for a specific calculation
    def gdBackprop(self, learnRate, target):
        numLayers = len(self.weights)
        deltaWeights = [None]*numLayers
        #seperates the last network output to the various components
        pre_nodeValues = self.last_output[0][:-1]
        post_nodeValues = self.last_output[1][:-1]
        pre_output = self.last_output[0][-1][0]
        output = self.last_output[1][-1][0]

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
        #updates weights
        self.weights =[self.weights[i] + deltaWeights[i] for i in range(numLayers)]
        return deltaWeights

    #changes the weights for a given input delta weights
    def changeWeights(self, newWeights):
        self.weights = newWeights



    #helper methods
    #helper method for calculate output
    def matrixMultiplication(self, weights, input):
        outputSize= len(weights[0])
        other= len(weights)
        lenOfInput=len(input)
        output=[]
        bbool=True
        for alpha in range(0, other):
            if(len(weights[alpha])!=outputSize):
                #print(weights[alpha])
                bbool=False
        if(lenOfInput==other and bbool):
            for a in range(0,outputSize):
                sum=0
                for b in range(0, other):
                    sum+=input[b]*weights[b][a]
                output.append(sum)
            return output
        else:
            if(bbool):
                raise ValueError("Dimensions are incompatible")
            else:
                raise ValueError("Weights are not all same length")

    #helper methods for gradient descent
    #gets the partial derivatives dE/dw_ij for an individual layers weights
    def layerWeightPartials(self, inputVals, outputDerivatives):

        rows = len(inputVals)
        cols = len(outputDerivatives)
        weightPartials = np.zeros((rows,cols))
        # print(inputVals)
        # print(outputDerivatives)
        for i in range(rows):
            for j in range(cols):
                weightPartials[i][j]  = inputVals[i] * outputDerivatives[j]
        
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


    def sqErrorCalc(self,target, output, deriv = False):
        if deriv:
            return output - target
        return float(((target - output)**2)/2) 







