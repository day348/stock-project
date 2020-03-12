import numpy as np
import time

class NeuralNetwork:
    def __init__(self):
        np.random.seed(10) # for generating the same results
        self.wij   = np.random.rand(3,50) # input to hidden layer weights
        self.wjk   = np.random.rand(50,1) # hidden layer to output weights
        
    def sigmoid(self, x, w):
        z = np.dot(x, w)
        return 1/(1 + np.exp(-z))
    
    def sigmoid_derivative(self, x, w):
        return self.sigmoid(x, w) * (1 - self.sigmoid(x, w))
    
    def gradient_descent(self, x, y, iterations):
        calcTime =0
        gradientTime = 0
        for i in range(iterations):
            start_time = time.time()
            Xi = x
            Xj = self.sigmoid(Xi, self.wij)
            yhat = self.sigmoid(Xj, self.wjk)
            calcTime = calcTime + time.time() -start_time
            start_time = time.time()
            # gradients for hidden to output weights
            g_wjk = np.dot(Xj.T, (y - yhat) * self.sigmoid_derivative(Xj, self.wjk))
            # gradients for input to hidden weights
            g_wij = np.dot(Xi.T, np.dot((y - yhat) * self.sigmoid_derivative(Xj, self.wjk), self.wjk.T) * self.sigmoid_derivative(Xi, self.wij))
            # update weights
            gradientTime = gradientTime + time.time() -start_time
            self.wij += g_wij
            self.wjk += g_wjk
        print("calc time: ", calcTime)
        print("gradient time: ", gradientTime)
        print('The final prediction from neural network are: ')
        print(yhat)
if __name__ == '__main__':
    neural_network = NeuralNetwork()
    print('Random starting input to hidden weights: ')
    print(neural_network.wij)
    print('Random starting hidden to output weights: ')
    print(neural_network.wjk)
    X = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    y = np.array([[0, 1, 1, 0]]).T
    neural_network.gradient_descent(X, y, 10000)