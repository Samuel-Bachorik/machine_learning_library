from linear_layer import Linearlayer
from activation_functions import *


class Model:
    def __init__(self):

        self.layers = [
                    Linearlayer(784, 784, bias=True),
                    Relu(),
                    Linearlayer(784, 1024, bias=True),
                    Relu(),
                    Linearlayer(1024, 2048, bias=True),
                    Relu(), 
                    Linearlayer(2048, 1024, bias=True),
                    Relu(),
                    Linearlayer(1024, 784, bias=True),
                    Relu(),
                    Linearlayer(784, 10, bias=True)
        ]

    def forward(self,x):
        x = self.layers[0].forward(x)
        x = self.layers[1].forward(x)
        x = self.layers[2].forward(x)
        x = self.layers[3].forward(x)
        x = self.layers[4].forward(x)
        x = self.layers[5].forward(x)
        x = self.layers[6].forward(x)
        x = self.layers[7].forward(x)
        x = self.layers[8].forward(x)
        x = self.layers[9].forward(x)
        x = self.layers[10].forward(x)


        return x
