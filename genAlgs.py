import random
import NeuralNet as nn
import statistics as stats

#generates random population amount of random neural networks
#intialize with varying weights for each layer but the same amount
#of layers and nodes per layer
#initialized with random activation functions as well
def genRandom(population, nodes_per_layer):
    neuralNets=[]
    layers=len(nodes_per_layer)
    for p in range(0,population):
        activation_funcs=[]
        for q in range(0,layers):
            funcs.append(random.randInt(0,9))
        net= nn.NeuralNet(activation_funcs,nodes_per_layer)
        neuralNets.append(net)
    return neuralNets
#Initial training of random population before genetic algs do their
#thing
def runBackProp(neuralNets, tics_to_inputs, tics_to_outputs, its=5):
    #get all the tics
    tics=tics_to_inputs.keys()
    #for each neural network
    for a in range(0,len(neuralNets)):
        #choose amount of iterations to initially train neural nets
        for b in range(0,its):
            #for each tic
            for b in tics:
                #back propogate over each tic
                inputs=tics_to_inputs[b]
                outputs=tics_to_inputs[b]
                neuralNets[a].backProp(inputs,outputs)
    return neuralNets

def singleScore(neuralNet, tics_to_outputs, tics_to_inputs):
    scores=[]
    for tic in tics_to_outputs.keys():
        inputVals=tics_to_inputs.values()
        calc=[]
        real=[]
        for p in range(0,len(inputVals)):
            calc.append(neuralNet.calculateOutput(inputVals[p]))
            real.append(tics_to_outputs[tic][p])
        scores.append(fitness(real, calc))
    score=stats.mean(scores)
    return score
def cumulativeScore(neuralNets, tics_to_outputs, tics_to_inputs)
    scoresDictionary={}
    for i in neuralNets:
        singleScore(i,tics_to_outputs, tics_to_inputs)
        scoresDictionary.update({i,score})
    return scoresDictionary
def fitness(actualVals,output):
    mad= 1/meanAbsoluteDev(actualVals, output)
    med= 1/medianAbsoluteDev(actualVals, output)
    ls= 1/leastSquares(actualVals, output)
    cheb= 1/cheby(actualVals, output)
    #sumOfAll=mad+med+ls+cheb
    fitScore=med+mad+cheb+ls
    return fitScore


#roulette wheel selection
def selection(scoresDictionary, amount):
    selection={}
    #normalize the scores
    scoresDictionary2=normalizeScores(scoresDictionary)
    #find sum of all the scores
    valTotal=sum(scoresDictionary2.values())
    #incrementer
    val=0
    #get keys 
    keys=scoresDictionary2.keys()
    #create dictionary for probabilites
    otherDict={}
    for a in keys:
        #get probability of reproduction
        prob=scoresDictionary2[a]/valTotal
        #multiply to 100 for simplicity
        prob=prob*100
        #add to val incrementer to get range of numbers
        prob=prob+val
        #update the dictionary
        otherDict.update{a:val}
        #update the val incrementer
        val=prob
    #loop until we have all those selected to breed
    while(len(selection)<amount)
        #get a random val between 0 and 1 and multiply by 100
        randVal=random.random()*100
        #get the neural networks as keys
        keys2=otherDict.keys()
        #loop until we find the key we should select
        for a in range(0,len(keys2)):
            #get the current and next keys
            keyToSee=keys2[a]
            nextKey=key2[a+1]
            #if our val is less than the val and and the next val is bigger, then break
            if(otherDict[keyToSee]<=randVal and otherDict[nextKey]>randVal):
                break
        #get the original score
        valToSee=scoresDictionary[keyToSee]
        #add the selection
        selection.update({keyToSee: valToSee})
    return selection


def evaluation(population, scoresDictionary):
    while(len(scoresDictionary)>population):
        scores=scoresDictionary.values()
        minScore=min(scores)
        scoresDictionary={key:val for key,val in scoresDictionary if val!=minScore}
    return scoresDictionary


def crossover(neuralNetwork1, neuralNetwork2, nodes_per_layer):
    #get the weights of each neural network
    weights1=neuralNetwork1.weights
    weights2=neuralNetwork2.weights
    #get the activation functions of each neural network
    activation1=neuralNetwork1.activation_funcs
    activation2=neuralNetwork2.activation_funcs
    #get number of activation functions
    actives=len(activation1)
    #get random int for one-point crossover of activation functions
    cross=random.randInt(0,actives)
    activation_funcs=[]
    for a in range(0,actives):
        if a<cross:
            activation_funcs.append(activation1[a])
        else:
            activation_funcs.append(activation2[a])
    #create initial neural net to return
    net= nn.NeuralNet(activation_funcs,nodes_per_layer)
    #get total amount of weights
    totalAmtOfWeights=0
    for a in range(0,len(weights)):
        for b in range(0,len(weights[a])):
            for c in range(0,len(weights[a][b]))
                totalAmtOfWeights++
    #grab two random ints for two-point crossover
    b=0
    c=0
    while(b==c)
        b= random.randInt(0,totalAmtOfWeights)
        c= random.randInt(0,totalAmtOfWeights)
    #find the lower index
    if(b<c):
        first=b
        second=c
    else:
        first=c
        second=b
    counter=0
    #crossover weights to child
    for a in range(0,len(weights)):
        for b in range(0,len(weights[a])):
            for c in range(0,len(weights[a][b]))
                if(counter<first):
                    net.weights[a][b][c]=weights1[a][b][c]
                else if(counter>=first and counter<second):
                    net.weights[a][b][c]=weights2[a][b][c]
                else:
                    net.weights[a][b][c]=weights1[a][b][c]
                counter+=1
    #return child
    return net
def findMostFit(scoresDictionary):
    scores=scoresDictionary.values()
    return max(scores)
def normalizeScores(scoresDictionary):
    maxVal=findMostFit(scoresDictionary)
    for a in scoresDictionary.keys():
        score=scoresDictionary[a]
        norm=score/maxVal
        updater={a: norm}
        scoresDictionary.update(updater)
    return scoresDictionary

def meanAbsoluteDev(right, estimate):
    total=0
    div=len(right)
    for a in range(0,div):
        total+=abs(right[a]-estimate[a])
    return total/div

def medianAbsoluteDev(right, estimate):
    even=False
    listOfDiff=[]
    length= len(right)
    for a in range(length):
        listOfDiff.append(abs(right[a]-estimate[a]))
    listOfDiff=sorted(listOfDiff)
    if(length % 2 == 0):
        val1= listOfDiff[length/2]
        val2= listOfDiff[(length/2)-1]
        summ=val1+val2
        return summ/2
    else:
        index= length/2
        index= index-0.5
        return listOfDiff[index]

def cheby(right, estimate):
    listOfDiff=[]
    for a in range(len(right)):
        listOfDiff.append(abs(right[a]-esimate[a]))
    return max(listOfDiff)


def leastSquares(right, estimate):
    listOfDiff=[]
    total=0
    for a in range(len(right)):
        partial= right[a]-estimate[a]
        square= partial*partial
        total+=square
    return total



    