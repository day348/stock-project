import NeuralNet as nn  
import numpy as np

x = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]]*50)
y = np.array([[0, 1, 1, 0]*50])


net = nn.NeuralNet([7,7,7],[3,1000,100,1])
if __name__ == "__main__":
    net.backProp(x,y[0],1,1)



