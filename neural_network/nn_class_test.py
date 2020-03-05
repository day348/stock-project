import NeuralNet as nn  
import numpy as np

x = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
y = np.array([[0, 1, 1, 0]])


net = nn.NeuralNet([7,7],[3,4,1])
net.backProp(x,y[0],1,10000)



