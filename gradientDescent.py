import numpy as np

#gets the partial derivatives dE/dw_ij for an individual layers weights
def layerWeightPartials(inputVals, outputDerivatives):

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
def nodeDerivatives(node_values, output_weights, activation_derivative, output_derivatives):
    numNodes = len(node_values)
    nodeParitals = np.zeros(numNodes)

    for i in range(numNodes):
        nodeParitals[i] = activation_derivative([node_values[i]])[0] * np.dot(output_weights[i], output_derivatives)

    return nodeParitals

#determines the change of weights for a specific calculation
def gdBackprop(weights, node_values, activation_derivative, cost_derivative, learnRate, target):
    numLayers = len(weights)
    deltaWeights = [None]*numLayers
    #seperates the input to the various components
    pre_nodeValues = node_values[0][:-1]
    post_nodeValues = node_values[1][:-1]
    pre_output = node_values[0][-1][0]
    output = node_values[1][-1][0]

    #calculates derivative for final output 
    tempPartials = [np.dot(cost_derivative([pre_output]) , errorCalcDeriv(target, output))]
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


def errorCalc(target, output):
    return float(((target - output)**2)/2) 

def errorCalcDeriv(target, input):
    #print(target)
    #print(input)
    return [input - target]

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




