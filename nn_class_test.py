import neural_network.NeuralNet as nn  
import numpy as np

x = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
y = np.array([0, 1, 1, 0])


net = nn.NeuralNet([7,7],[3,100,1])
if __name__ == "__main__":
    error_change = net.backProp(x,y,learnRate = 1,iterations = 100)
    for i in range(4):
        print(net.calculateOutput(x[i])[-1][-1])
    print(error_change)

