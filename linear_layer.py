import numpy as np
 
class Linearlayer:
    def __init__(self, in_features, out_feauters, bias= True):

        self.in_features    = in_features
        self.out_features   = out_feauters
        self.b              = bias

        #Initialize weight with xavier uniform
        limit       = np.sqrt(2 / float(in_features + out_feauters))
        self.weight = np.random.normal(0.0, limit, size=(in_features, out_feauters))

        # Store self output from layer for backpropagation chain rule
        self.x = None

        #Check if bias is true
        if self.b is True:
            self.bias = (np.zeros(out_feauters)).astype(float)

    def forward(self, input):
        x, y = input.shape
        if y != self.in_features:
            raise Exception(f'Wrong input features. Please use input tensor with {str(self.in_features)} input features')

        y = np.matmul(input, self.weight)

        if self.b:
            y = y + self.bias

        self.x = y

        return y
