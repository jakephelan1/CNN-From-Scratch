import numpy as np
from layers import Layer

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        self.input = input
        batch_size = input.shape[0]
        return np.reshape(input, (batch_size, *self.output_shape))
    
    def backward(self, output_gradient, learning_rate):
        batch_size = output_gradient.shape[0]
        return np.reshape(output_gradient, (batch_size, *self.input_shape))