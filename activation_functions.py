import numpy as np
import backward_linear_layer

class Sigmoid:
    def __init__(self):
        self.x = None

    def forward(self,x):
        y = 1 / (1 + np.exp(-x))

        self.x = y

        return y

    def backward(self,x):

        grad =  np.exp(-x)/((np.exp(-x)+1)**2)

        return grad


class Relu:
    def __init__(self):
        self.x = None

    def forward(self,x):
        y = (np.maximum(0, x))
        self.x = y

        return y

    def backward(self,x):
        grad = (x > 0)

        return grad


class TanH:
    def __init__(self):
        self.x = None

    def forward(self,x):
        y = (np.exp(1)**x - np.exp(1)** - x) / \
            (np.exp(1)**x + np.exp(1)** - x)

        self.x = y

        return y

    def backward(self,x):
       grad =  ((4 * np.exp(1) ** (2 * x)) /
         ((np.exp(1) ** (2 * x)) + 1) ** 2)

       return grad


