import math  
import numpy
#general activation function usage
#takes which function calling and input value

"""
0 : TanH
1 : arcTan
2 : Elu
3 : Identity
4 : LeakyRelu
5 : RandomRelu
6 : Relu
7 : Sigmoid
8 : SoftPlus
9 : Step

"""
#general activation function usage
#takes which function calling and input value

def func(selector, output, rando=1, deriv = False):
    #activation functions and all their derivatives
    def activationSigmoid( input):
        for a in range(0,len(input)):
            z=input[a]
            z=(-1)*z
            input[a]= 1/(1+math.exp(z))
        return input
    def activationTanH(input):
        for a in range(0,len(input)):
            z=input[a]
            input[a]=math.tanh(z)
        return input
    def activationReLU(input):
        for a in range(0,len(input)):
            z=input[a]
            if(z<0):
                input[a]=0
        return input
    def derivReLU( input):
        for a in range(0,len(input)):
            z=input[a]
            if(z<0):
                input[a]=0
            else:
                input[a]=1
        return input
    def activationLeakyReLU( input):
        b=0.01
        for a in range(0,len(input)):
            z=input[a]
            if(z<0):
                input[a]=z*b
        return input
    def derivLeakyReLU( input):
        for a in range(0, len(input)):
            z=input[a]
            if(z<0):
                input[a]=0.01
            else:
                input[a]=1
        return input
    def activationRandomReLU( input,b):
        for a in range(0,len(input)):
            z=input[a]
            if(z<0):
                input[a]=z*b
        return input
    def derivRandomReLU( input, b):
        for a in range(0,len(input)):
            z=input[a]
            if(z<0):
                input[a]=b
            else:
                input[a]=1
        return input
    """ def activationStep( input):
        for a in range(0,len(input)):
            z=input[a]
            if(z<0):
                input[a]=0
            else:
                if(bbool):
                    raise ValueError("Dimensions are incompatible")
                else:
                    raise ValueError("Weights are not all same length")
            return input """
    def activationSigmoid(input):
        for a in range(0,len(input)):
            z=input[a]
            if z == 0:
                z = .00001
            z=(-1)*z
            try:
                input[a]= 1/(1+math.exp(z))
            except: 
                print("overloading ouput to Sigmoid")
                input[a] = 0
        return input
    def derivSigmoid(input):
        z=activationSigmoid(input)
        for a in range(0,len(input)):
            input[a]=z[a]*(1-z[a])
        return input
    def activationTanH(input):
        for a in range(0,len(input)):
            z=input[a]
            input[a]=math.tanh(z)
        return input
    def derivTanH( input):
        z=activationTanH(input)
        for a in range(0,len(input)):
            input[a]=1-z[a]*z[a]
        return input
    def activationReLU(input):
        for a in range(0,len(input)):
            z=input[a]
            if(z<0):
                input[a]=0
        return input
    def derivReLU( input):
        for a in range(0,len(input)):
            z=input[a]
            if(z<0):
                input[a]=0
            else:
                input[a]=1
        return input
    def activationLeakyReLU( input):
        b=0.01
        for a in range(0,len(input)):
            z=input[a]
            if(z<0):
                input[a]=z*b
        return input
    def derivLeakyReLU( input):
        for a in range(0, len(input)):
            z=input[a]
            if(z<0):
                input[a]=0.01
            else:
                input[a]=1
        return input
    def activationRandomReLU( input,b):
        for a in range(0,len(input)):
            z=input[a]
            if(z<0):
                input[a]=z*b
        return input
    def derivRandomReLU( input, b):
        for a in range(0,len(input)):
            z=input[a]
            if(z<0):
                input[a]=b
            else:
                input[a]=1
        return input
    def activationStep( input):
        for a in range(0,len(input)):
            z=input[a]
            if(z<0):
                input[a]=0
            else:
                input[a]=1
        return input
    def derivStep( input):
        for a in range(0,len(input)):
            z=input[a]
            if(z<0):
                input[a]=-1
            else:
                input[a]=1
            """ if(input[a]!=0):
                input[a]=0
            else:
                raise ValueError("Not differentiable") """
        return input
    def activationArcTan( input):
        for a in range(0,len(input)):
            input[a]=numpy.arctan(input[a])
        return input
    def derivArcTan(input):
        for a in range(0,len(input)):
            z=input[a]*input[a]+1
            input[a]=1/z
        return input
    def activationSoftPlus( input):
        for a in range(0,len(input)):
            z=input[a]
            b=math.log(1+math.exp(z))
            input[a]=b
        return input
    def derivSoftPlus( input):
        return activationSigmoid(input)
    def activationELU( input, b):
        for a in range(0,len(input)):
            z=input[a]
            if(z<0):
                q=b*(math.exp(z)-1)
                input[a]=q
        return input
    def derivELU( input, b):
        z=activationELU( input, b)
        for a in range(0,len(input)):
            input[a]=z[a]+b
        return input
    def activationIdentity( input):
        return input
    def derivIdentity( input):
        for a in range(0,len(input)):
            input[a]=1
        return input
    def activationArcTan( input):
        for a in range(0,len(input)):
            input[a]=numpy.arctan(input[a])
        return input
    def activationSoftPlus( input):
        for a in range(0,len(input)):
            z=input[a]
            b=math.log(1+math.exp(z))
            input[a]=b
        return input
    def activationELU( input, b):
        for a in range(0,len(input)):
            z=input[a]
            if(z<0):
                q=b*(math.exp(z)-1)
                input[a]=q
        return input
    def activationIdentity( input):
        return input
    def derivIdentity( input):
        for a in range(0,len(input)):
            input[a]=1
        return input

    if deriv == False:
        if(selector==1):
            input= activationArcTan(output)
        elif(selector==2):
            input=activationELU(output,rando)
        elif(selector==3):
            input=activationIdentity(output)
        elif(selector==4):
            input=activationLeakyReLU(output)
        elif(selector==5):
            input=activationRandomReLU(output, rando)
        elif(selector==6):
            input=activationReLU(output)
        elif(selector==7):
            input=activationSigmoid(output)
        elif(selector==8):
            input=activationSoftPlus(output)
        elif(selector==9):
            input=activationStep(output)
        else:
            input=activationTanH(output)
        return input
    else:
        if(selector==1):
            input= derivArcTan(output)
        elif(selector==2):
            input=derivELU(output,rando)
        elif(selector==3):
            input=derivIdentity(output)
        elif(selector==4):
            input=derivLeakyReLU(output)
        elif(selector==5):
            input=derivRandomReLU(output, rando)
        elif(selector==6):
            input=derivReLU(output)
        elif(selector==7):
            input=derivSigmoid(output)
        elif(selector==8):
            input=derivSoftPlus(output)
        elif(selector==9):
            input=derivStep(output)
        else:
            input=derivTanH(output)
        return input
