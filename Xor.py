from dense import Dense
from activation import Activation
from activationfunction import *
from losss import *
import numpy as np
X = np.reshape([[0,0],[0,1],[1,0],[1,1]],(4,2,1))
Y = np.reshape([[0],[1],[1],[0]],(4,1,1))
network = [Dense(2,3),Tanh(),Dense(3,1),Tanh()]
epoch = 10000
learning_rete = 0.1
for e in range(epoch):
    error = 0
    for x,y in zip(X,Y):
        output = x
        for layer in network:
            output = layer.forward(output)
        error+=mse(y,output)
        grad = mse_prime(y,output)
        for layer in reversed(network):
            grad = layer.backward(grad,learning_rete)
    error/=len(x)
    print('%d/%d,error = %f' % (e+1,epoch,error))
