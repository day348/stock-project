from neuralnet import neuralnet 

print("testing good class decleration")
nn = neuralnet([3,3,3], nodes_per_layer = [5,3,1])
print(nn.weights)
output = nn.calculateOutput([1,2,3,4,5])
print(output)
print(nn.gdBackprop(.1,5))
print(nn.weights)


