import math
import numpy
import random

class neuralnet:
    weights = [[]]
    errorCalc = None
    activation_functions = []

    def __init__(self,activation_funcs, input_weigths=-1, nodes_per_layer=-1 , error_method = sqErrorCalc):

        #checks gives correct weight info as input
        if type(input_weigths) == type(-1) & type(nodes_per_layer) == type(-1):
                print("must either give nodes per layer or input weights")
        elif type(input_weigths) == type(-1):
            num_layers = nodes_per_layer.length
            weights = [[ [ random.randint(-2,2) for i in range(nodes_per_layer[j]) ] for j in range(num_layers) ] * num_layers ]
        else:
            weights = input_weigths

        #checks activation functions
        if activation_funcs.length == weights.length:
            activation_functions = activation_funcs
        else:
            print("invalid action functions")
        #default is error squared
        errorCalc = error_method

        pass



    def calculateOutput(self, activationFunction, matrix, input, rando):
        b=len(matrix);
        listToBeReturned=[]
        other=[]
        for a in range(0,b):
            listToBeReturned.append(input);
            if(a == 0):
                other.append(input)
            try:
                output= matrixMultiplication(matrix[a], input)
                other.append(output)
            except ValueError: 
                print("Matrices were invalid dimensions or...")
            active= activationFunction[a]
            if(active==1):
                input=activationArcTan(output)
            elif(active==2):
                input=activationELU(output,rando)
            elif(active==3):
                input=activationIdentity(output)
            elif(active==4):
                input=activationLeakyReLU(output)
            elif(active==5):
                input=activationRandomReLU(output, rando)
            elif(active==6):
                input=activationReLU(output)
            elif(active==7):
                input=activationSigmoid(output)
            elif(active==8):
                input=activationSoftPlus(output)
            elif(active==9):
                input=activationStep(output)
            else:
                input=activationTanH(output)
        listToBeReturned.append(input)
        return [listToBeReturned, other] 



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
    def activationSigmoid(self, input):
        for a in range(0,len(input)):
            z=input[a]
            z=(-1)*z
            input[a]= 1/(1+math.exp(z))
        return input
    def activationTanH(self,input):
        for a in range(0,len(input)):
            z=input[a]
            input[a]=math.tanh(z)
        return input
    def activationReLU(self,input):
        for a in range(0,len(input)):
            z=input[a]
            if(z<0):
                input[a]=0
        return input
    def derivReLU(self, input):
        for a in range(0,len(input)):
            z=input[a]
            if(z<0):
                input[a]=0
            else:
                input[a]=1
        return input
    def activationLeakyReLU(self, input):
        b=0.01
        for a in range(0,len(input)):
            z=input[a]
            if(z<0):
                input[a]=z*b
        return input
    def derivLeakyReLU(self, input):
        for a in range(0, len(input)):
            z=input[a]
            if(z<0):
                input[a]=0.01
            else:
                input[a]=1
        return input
    def activationRandomReLU(self, input,b):
        for a in range(0,len(input)):
            z=input[a]
            if(z<0):
                input[a]=z*b
        return input
    def derivRandomReLU(self, input, b):
        for a in range(0,len(input)):
            z=input[a]
            if(z<0):
                input[a]=b
            else:
                input[a]=1
        return input
    def activationStep(self, input):
        for a in range(0,len(input)):
            z=input[a]
            if(z<0):
                input[a]=0
            else:
                if(bbool):
                    raise ValueError("Dimensions are incompatible")
                else:
                    raise ValueError("Weights are not all same length")
            return input
    def activationSigmoid(self,input):
        for a in range(0,len(input)):
            z=input[a]
            z=(-1)*z
            input[a]= 1/(1+math.exp(z))
        return input
    def derivSigmoid(self,input):
        z=activationSigmoid(input)
        for a in range(0,len(input)):
            input[a]=z[a]*(1-z[a])
        return input
    def activationTanH(self,input):
        for a in range(0,len(input)):
            z=input[a]
            input[a]=math.tanh(z)
        return input
    def derivTanH(self, input):
        z=activationTanH(input)
        for a in range(0,len(input)):
            input[a]=1-z[a]*z[a]
        return input
    def activationReLU(self,input):
        for a in range(0,len(input)):
            z=input[a]
            if(z<0):
                input[a]=0
        return input
    def derivReLU(self, input):
        for a in range(0,len(input)):
            z=input[a]
            if(z<0):
                input[a]=0
            else:
                input[a]=1
        return input
    def activationLeakyReLU(self, input):
        b=0.01
        for a in range(0,len(input)):
            z=input[a]
            if(z<0):
                input[a]=z*b
        return input
    def derivLeakyReLU(self, input):
        for a in range(0, len(input)):
            z=input[a]
            if(z<0):
                input[a]=0.01
            else:
                input[a]=1
        return input
    def activationRandomReLU(self, input,b):
        for a in range(0,len(input)):
            z=input[a]
            if(z<0):
                input[a]=z*b
        return input
    def derivRandomReLU(self, input, b):
        for a in range(0,len(input)):
            z=input[a]
            if(z<0):
                input[a]=b
            else:
                input[a]=1
        return input
    def activationStep(self, input):
        for a in range(0,len(input)):
            z=input[a]
            if(z<0):
                input[a]=0
            else:
                input[a]=1
        return input
    def derivStep(self, input):
        for a in range(0,len(input)):
            if(input[a]!=0):
                input[a]=0
            else:
                raise ValueError("Not differentiable")
        return input
    def activationArcTan(self, input):
        for a in range(0,len(input)):
            input[a]=numpy.arctan(input[a])
        return input
    def derivArcTan(input):
        for a in range(0,len(input)):
            z=input[a]*input[a]+1
            input[a]=1/z
        return input
    def activationSoftPlus(self, input):
        for a in range(0,len(input)):
            z=input[a]
            b=math.log(1+math.exp(z))
            input[a]=b
        return input
    def derivSoftPlus(self, input):
        return activationSigmoid(input)
    def activationELU(self, input, b):
        for a in range(0,len(input)):
            z=input[a]
            if(z<0):
                q=b*(math.exp(z)-1)
                input[a]=q
        return input
    def derivELU(self, input, b):
        z=activationELU( input, b)
        for a in range(0,len(input)):
            input[a]=z[a]+b
        return input
    def activationIdentity(self, input):
        return input
    def derivIdentity(self, input):
        for a in range(0,len(input)):
            input[a]=1
        return input
    def activationArcTan(self, input):
        for a in range(0,len(input)):
            input[a]=numpy.arctan(input[a])
        return input
    def activationSoftPlus(self, input):
        for a in range(0,len(input)):
            z=input[a]
            b=math.log(1+math.exp(z))
            input[a]=b
        return input
    def activationELU(self, input, b):
        for a in range(0,len(input)):
            z=input[a]
            if(z<0):
                q=b*(math.exp(z)-1)
                input[a]=q
        return input
    def activationIdentity(self, input):
        return input
    def derivIdentity(self, input):
        for a in range(0,len(input)):
            input[a]=1
        return input

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
def nodeDerivatives(self, node_values, output_weights, activation_derivative, output_derivatives):
    numNodes = len(node_values)
    nodeParitals = np.zeros(numNodes)

    for i in range(numNodes):
        nodeParitals[i] = activation_derivative([node_values[i]])[0] * np.dot(output_weights[i], output_derivatives)

    return nodeParitals

#determines the change of weights for a specific calculation
def gdBackprop(self,weights, node_values, activation_derivative, cost_derivative, learnRate, target):
    numLayers = len(weights)
    deltaWeights = [None]*numLayers
    #seperates the input to the various components
    pre_nodeValues = node_values[0][:-1]
    post_nodeValues = node_values[1][:-1]
    pre_output = node_values[0][-1][0]
    output = node_values[1][-1][0]

    #calculates derivative for final output 
    tempPartials = [np.dot(cost_derivative([pre_output]) , errorCalc(target, output, deriv= True))]
    for i in range(numLayers):
        layer = -i-1
        #gets the partials for the nodes 
        #gets the weight changes
        # print(weights[layer])
        deltaWeights[layer]  = -1 * learnRate * layerWeightPartials(post_nodeValues[layer], tempPartials)

        if(i != numLayers-1):
            #for hidden layers
            tempPartials = nodeDerivatives(pre_nodeValues[layer], weights[layer], activation_derivative, tempPartials)


    return deltaWeights


def sqErrorCalc(self,target, output, deriv = False):
    if deriv:
        return [output - target]
    return float(((target - output)**2)/2) 


#test function
# weights = [[[1,2],[3,4]], [[1],[2]]]

# pre_nodeValues = [[1,1], [4,6]]
# post_nodeValues = [[1,1], [16,36]]

# def func1(x):
#     return x
# def func2(x, target):
#     return x-target

# nodeValues = [4,6]
# inputVals = [1,1]

# derivs = [-16,-24]
# output_derivatives = [[-4]]
# def func(x):
#     return 1 

# print(gdBackprop(weights, pre_nodeValues, post_nodeValues, func1, func, 0.5 , 88 , 88, 92))




