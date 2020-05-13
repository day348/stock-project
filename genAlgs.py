import random
import NeuralNet as nn
import statistics as stats
import pandas as pd
#generates random population amount of random neural networks
#intialize with varying weights for each layer but the same amount
#of layers and nodes per layer
#initialized with random activation functions as well
#outputs neural nets as a list
def genRandom(population, nodes_per_layer):
    neuralNets=[]
    layers=len(nodes_per_layer)
    for p in range(0,population):
        activation_funcs=[]
        for q in range(0,layers-1):
            #we only want tanh, arctan, or constant activation
            alpha=random.randint(0,2)
            #if 2, change to 3 which is constant
            if(alpha==2):
                activation_funcs.append(3)
            else:
                activation_funcs.append(alpha)
        net= nn.NeuralNet(activation_funcs,nodes_per_layer)
        neuralNets.append(net)
    return neuralNets



#Initial training of random population before genetic algs do their
#thing, pass in list of neural networks and the training dictionaries
#optionally pass in iterations for back propogation to be ran
#outputs updated neural nets 
def runBackProp(neuralNets, tics_to_inputs, tics_to_outputs, its=1):
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

#returns the mean absolute deviation from estimate and right
def meanAbsoluteDev(right, estimate):
    total=0
    div=len(right)
    if(div==0):
        return None
    for a in range(0,div):
        #get the absolute value and add it to the running error total
        total+=abs(right[a]-estimate[a])
    return total/div
#returns the median absolute deviation from estimate and right
def medianAbsoluteDev(right, estimate):
    even=False
    listOfDiff=[]
    length= len(right)
    if(length==0):
        return None
    #grab all the errors
    for a in range(length):
        listOfDiff.append(abs(right[a]-estimate[a]))
    #sort them
    listOfDiff=sorted(listOfDiff)
    #if even length
    if length % 2 == 0:
        val1= listOfDiff[int(length/2)]
        val2= listOfDiff[(int(length/2))-1]
        summ=val1+val2
        return summ/2
    #if odd length
    else:
        index= length/2
        index= int(index-0.5)
        return listOfDiff[index]
#get the maximum error (ie Chebyshev criteria-minimizing the maximum error)
def cheby(right, estimate):
    listOfDiff=[]
    for a in range(len(right)):
        listOfDiff.append(abs(right[a]-estimate[a]))
    return max(listOfDiff)

#get the least squares value of right and estimate
def leastSquares(right, estimate):
    listOfDiff=[]
    total=0
    for a in range(len(right)):
        #find the difference
        partial= right[a]-estimate[a]
        #square it
        square= partial*partial
        #add it to the total
        total+=square
    return total


#takes in the actual values and what the neural net outputs
#returns the fitness score for the neural net
def fitness(actualVals,output):
    mad= meanAbsoluteDev(actualVals, output)
    #med= medianAbsoluteDev(actualVals, output)
    #ls= 1/leastSquares(actualVals, output)
    #sumOfAll=mad+med+ls
    #fitScore=mad+cheb+ls
    if(mad==None):
        return None
    mad=1/mad
    return mad




#Pass in single neural network and training data dictionaries
#outputs a single fitness score for the neural net
def singleScore(neuralNet, tics_to_outputs, tics_to_inputs):
    scores=[]
    #for every ticker, we grab 4 random values to try and predict
    for tic in tics_to_outputs.keys():
        calc=[]
        real=[]
        #for 4 times
        for p in range(0,4):
            #if we have ample data at this tic
            if(len(tics_to_inputs[tic])>5):
                #get a random index
                a=random.randint(0,len(tics_to_inputs[tic])-1)
                #get the real and calculate the neural net output
                calc.append(neuralNet.calculateOutput(tics_to_inputs[tic][a], single_input=True))
                real.append(tics_to_outputs[tic][a])
        if(fitness(real,calc)!=None):
            #append the fitness
            scores.append(fitness(real, calc))
    #return the mean of all the fitnesses across every ticker
    score=stats.mean(scores)
    return score

#Pass in list of neural networks and the training data dictionaries
#returns a dictionary of the neural networks and their scores 
def cumulativeScore(neuralNets, tics_to_outputs, tics_to_inputs):
    scoresDictionary={}
    for i in neuralNets:
        score=singleScore(i,tics_to_outputs, tics_to_inputs)
        scoresDictionary.update({i:score})
    return scoresDictionary

#returns the highest fitness score from the score dictionary
def findMostFit(scoresDictionary):
    scores=scoresDictionary.values()
    return max(scores)

#Takes a scores dictionary and normalizes them to values from
#0 to 1 and returns said dictionary
def normalizeScores(scoresDictionary):
    maxVal=findMostFit(scoresDictionary)
    for a in scoresDictionary.keys():
        score=scoresDictionary[a]
        norm=score/maxVal
        updater={a: norm}
        scoresDictionary.update(updater)
    return scoresDictionary

#roulette wheel selection
#input the scores dictionary of neural nets with their scores
#and the amount of parents you want
#returns a dictionary of the to be parent neural nets and their fitness scores
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
    return selection#roulette wheel selection
#input the scores dictionary of neural nets with their scores
#and the amount of parents you want
#returns a dictionary of the to be parent neural nets and their fitness scores
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
    #create list of tuples
    otherList=[]
    for a in keys:
        #get probability of reproduction
        prob=scoresDictionary2[a]/valTotal
        #multiply to 100 for simplicity
        prob=prob*100
        #add to val incrementer to get range of numbers
        prob=prob+val
        #create tuple
        tup=(a,prob)
        #update the list
        otherList.append(tup)
        #update the val incrementer
        val=prob
    #loop until we have all those selected to breed
    while(len(selection)<amount):
        #get a random val between 0 and 1 and multiply by 100
        randVal=random.random()*100
        for a in range(0,len(otherList)):
            #get the value
            val=otherList[a][1]
            if(val>randVal):
                break
        #get the original score and neural network
        keyToSee=otherList[a][0]
        valToSee=scoresDictionary[keyToSee]
        #add the selection
        selection.update({keyToSee: valToSee})
    return selection



#Turns the inputted one dimensional matrix to a three dimensional 
#matrix with nodes_per_layer dimensions
def turnOneToThree(listInOne, nodes_per_layer):
    threeDList=[]
    counter=0
    for a in range(0,len(nodes_per_layer)-1):
        twoDList=np.random.rand(nodes_per_layer[a],nodes_per_layer[a+1])
        for b in range(0, nodes_per_layer[a]):
            oneDList=np.zeros(nodes_per_layer[a+1])
            for c in range(0,nodes_per_layer[a+1]):
                oneDList[c]=listInOne[counter]
                counter+=1
            twoDList[b]=oneDList
        threeDList.append(twoDList)
    return threeDList


#Turns the inputted three dimensional matrix into a single list
def turnThreeToOne(matrix):
    listToReturn=[]
    for a in range(0,len(matrix)):
        for b in range(0,len(matrix[a])):
            for c in range(0, len(matrix[a][b])):
                listToReturn.append(matrix[a][b][c])
    return listToReturn




#This function takes in a single neural network
#Mutates its weights (not activation functions)
#and returns the mutated neural net
#inversion mutation
def mutateThatJohn(neuralNetToMutate, nodes_per_layer):
    #don't mutate the activation functions
    activation_funcs=neuralNetToMutate.activation_functions
    #creates new neural network
    neuralNetToReturn= nn.NeuralNet(activation_funcs,nodes_per_layer)
    weightsOf=neuralNetToMutate.weights
    #turn weights to one dimension
    weights1d= turnThreeToOne(weightsOf)
    #get total amount of weights
    totalAmtOfWeights=len(weights1d)
    b=0
    c=0
    #loop until we get two different indices
    while c==b:
        b= random.randint(0,totalAmtOfWeights)
        c= random.randint(0,totalAmtOfWeights)
    #grab sublist from those indices
    if b<c:
        subList=weights1d[b:c]
    else:
        subList=weights1d[c:b]
    #reverse the list
    subList.reverse()
    #replace elements in the list with reversed list
    if b<c:
        weights1d[b:c]=subList
    else:
        weights1d[c:b]=subList
    #need it back in three dimensions
    weightsBack=turnOneToThree(weights1d, nodes_per_layer)
    #set mutated weights to new neural net
    neuralNetToReturn.weights=weightsBack
    #return mutated neural net
    return neuralNetToReturn






#inputs two neural networks and a list of the nodes per layer 
#and crosses over the activation functions and weights of the
#two neural networks and returns the network that is their child
def crossover(neuralNetwork1, neuralNetwork2, nodes_per_layer):
    #get the weights of each neural network
    weights1=neuralNetwork1.weights
    weights2=neuralNetwork2.weights
    #get the activation functions of each neural network
    activation1=neuralNetwork1.activation_functions
    activation2=neuralNetwork2.activation_functions
    #get number of activation functions
    actives=len(activation1)
    #get random int for one-point crossover of activation functions
    cross=random.randint(0,actives)
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
    for a in range(0,len(weights1)):
        for b in range(0,len(weights1[a])):
            for c in range(0,len(weights1[a][b])):
                totalAmtOfWeights+=1
    #grab two random ints for two-point crossover
    b=0
    c=0
    while b==c:
        b= random.randint(0,totalAmtOfWeights)
        c= random.randint(0,totalAmtOfWeights)
    #find the lower index
    if b<c:
        first=b
        second=c
    else:
        first=c
        second=b
    counter=0
    #crossover weights to child
    for a in range(0,len(weights1)):
        for b in range(0,len(weights1[a])):
            for c in range(0,len(weights1[a][b])):
                if counter<first:
                    net.weights[a][b][c]=weights1[a][b][c]
                elif counter>=first and counter<second:
                    net.weights[a][b][c]=weights2[a][b][c]
                else:
                    net.weights[a][b][c]=weights1[a][b][c]
                counter+=1
    #return child
    return net

#Takes in a dictionary of those selected to be parents
#Takes in a dictionary of all the neural nets and their scores
#Takes in a population threshold (optionally takes in proportion to mutate)
#Returns list of mutated to pass to next generation
def mutateThoseJohns(population, selectionDictionary, scoresDictionary, nodes_per_layer, mutate=0.09):
    #get the list of neural networks
    listOfPotential=list(scoresDictionary.keys())
    #get the list of the selected neural networks
    listOfSelected=list(selectionDictionary.keys())
    #find the length of the list of potential
    totalLength=len(listOfPotential)
    #set amount selected to 0
    amountForMutate=0
    #set amount to select to the proportion of the population needed to
    #be mutated
    amountToMutate=population*mutate
    #list of mutated
    listOfMutated=[]
    #loop until all have been selected for mutation
    while(amountForMutate<amountToMutate):
        #get a random index to select
        randomIndex=random.randint(0,totalLength-1)
        #select it
        neuralNetToMutate=listOfPotential[randomIndex]
        #if it was not already selected
        if(neuralNetToMutate not in listOfSelected):
            #mutate it
            neuralNetToMutate=mutateThatJohn(neuralNetToMutate, nodes_per_layer)
            #add it to the list
            listOfMutated.append(neuralNetToMutate)
            #increment
            amountForMutate+=1
    #return list
    return listOfMutated


#generates next generation with population as the threshold
#value for amount in the next generation and scoresDictionary that
#maps neural nets to their fitness scores.
#Returns list of next generation of neural nets
def evaluation(population, selection, scoresDictionary, nodes_per_layer):
    #start list of next population
    nextPopulation=[]
    #grab the parents we will be using
    parents=list(selection.keys())
    #mutate some of the population and append them for the next generation
    nextPopulation.extend(mutateThoseJohns(population, selection, scoresDictionary, nodes_per_layer))
    #append parents to the next population
    nextPopulation.extend(parents)
    #get list of neural networks
    thelist1 = [(k, v) for k, v in scoresDictionary.items()] 
    #find the best neural net
    best=findMostFitNeuralNet(thelist1)
    nextPopulation.append(best[0])
    #loop until we have a full next population
    while len(nextPopulation)<population: 
        #parent indices
        par1=0
        par2=0
        #until two different parents
        while par1==par2:
            par1=random.randint(0,len(parents)-1)
            par2=random.randint(0,len(parents)-1)
        #grab the parents
        parent1=parents[par1]
        parent2=parents[par2]
        #create the child
        child= crossover(parent1, parent2, nodes_per_layer)
        #append the child
        nextPopulation.append(child)
    return nextPopulation




#Executes iterations worth of generations of neural networks
def letsDoSomeGeneticAlgorithms(nodes_per_layer, population, tics_to_outputs, tics_to_inputs, amount=0.3,iterations=100):
    #create initial population of neural networks
    neuralNets=genRandom(population, nodes_per_layer)
    #get amount of parents wanted
    parents=amount*population
    #set loop condition
    i=0
    #after iterations amount of iterations
    while i<iterations:
        #get the scores of the neural networks
        scoresDictionary=cumulativeScore(neuralNets, tics_to_outputs, tics_to_inputs)
        #grab parents for next generation
        selectionDictionary=selection(scoresDictionary, parents)
        #create next population
        nextPop=evaluation(population, selectionDictionary, scoresDictionary, nodes_per_layer)
        #set that to neuralNets
        neuralNets=nextPop
        #increment i
        i+=1
    #return the neural network loop
    return neuralNets

#how many did we get right in direction
def directionPred(actual,predicted):
    right=0;
    wrong=0;
    for i in range(0,len(actual)):
        for j in range(0,len(actual[i])):
            if(predicted[i][j]>0 and actual[i][j]>0):
                right+=1
            elif(predicted[i][j]<0 and actual[i][j]<0):
                right+=1
            elif(predicted[i][j]==0 and actual[i][j]==0):
                right+=1
            elif(actual[i][j]==0):
                #don't count as wrong because there was no direction
            else:
                wrong+=1
    return [right,wrong]

#see what the error was on average
def averageError(actual, predicted):
    error=0
    ticker=0
    for i in range(0,len(actual)):
        for j in range(0,len(actual[i])):
            val=actual[i][j]-predicted[i][j]
            if(val<0):
                val=val*(-1)
            error+=val
            ticker+=1
    return error/ticker
#grab all the best predictions for the testing data
def bestPredictions(best):
    actual=[]
    predicted=[]
    for tic in outputsForTesting[0].keys():
        calc=[]
        real=[]
        for p in range(0,len(outputsForTesting[0][tic])):
            calc.append(best[0].calculateOutput(outputsForTesting[0][tic][p], single_input=True))
            real.append(outputsForTesting[1][tic][p])
        actual.append(real)
        predicted.append(calc)
    return [actual, predicted]
#average out the output for the whole population and see how that does 
#as a prediction
from statistics import mean
def wholePopAveragePredictions(final):
    actual=[]
    predicted=[]
    for tic in outputsForTesting[0].keys():
        calc=[]
        real=[]
        for p in range(0,len(outputsForTesting[0][tic])):
            temp=[]
            #for every neural net
            for k in range(0,len(final)):
                temp.append(final[k].calculateOutput(outputsForTesting[0][tic][p], single_input=True))
            #get the mean of all the neural net predictions
            calc.append(mean(temp))
            real.append(outputsForTesting[1][tic][p])
        actual.append(real)
        predicted.append(calc)
    return [actual, predicted]

#grab every ticker
tics=[]
for i in range(0,len(frame['Ticker'])):
    tics.append(frame['Ticker'][i])
#get random tickers to test
def getRandomTics(n):
    i=0
    ret=[]
    while(i<n):
        #get random index
        k=random.randint(0,len(tics)-1)
        ret.append(tics[k])
        i+=1
    return ret

#function to export npy file
from tempfile import TemporaryFile
def exportNeuralNet(best, name):
    path= name + '.npy';
    np.save(path,best[0].weights)