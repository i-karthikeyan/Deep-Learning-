from layer import Layer
import numpy as np
class Dense(Layer):
    def __init__(self,input_size,output_size):
        self.weight = np.random.randn(output_size,input_size)
        self.bias = np.random.randn(output_size,1)
    def forward(self, input):
        self.input = input
        return np.dot(self.weight,self.input)+self.bias
    def backward(self, output_gradient, learning_rate):
        weight_gradient = np.dot(output_gradient,self.input.T)
        input_gradient = np.dot(self.weight.T,output_gradient)
        self.weight -= learning_rate*weight_gradient
        self.bias -= learning_rate* output_gradient
        return input_gradient