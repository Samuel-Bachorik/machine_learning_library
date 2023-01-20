import numpy
import numpy as np
from loss_functions import *
from optimizers import *

class backward_linear:
    def __init__(self, optimizer, model):

        self.optimizer = optimizer
        print(self.optimizer)
        if self.optimizer != "sgd" and self.optimizer != "adam":
            raise Exception("Wrong optimizer selected use 'adam' or 'sgd'")

        if optimizer == "adam":
            print("You are using Adam optimizer \n")
            self.adam_optimizers = []
            for i in range(len(model.layers)):
                if model.layers[i].__class__.__name__ == "Linearlayer":
                    self.adam_optimizers.append(Adam())

                else:
                    self.adam_optimizers.append(None)

        if self.optimizer == "sgd":
            self.sgd = SGD()


    def run_backward(self,model, y_pred , y, lr, x, epoch, loss):

        first_grad = loss.backward(y, y_pred)

        for i in range(len(model.layers) -1, -1, -1):
            actual_layer = model.layers[i].__class__.__name__

            if actual_layer == "Linearlayer":
                if i > 0:
                    input = model.layers[i-1].x.transpose()

                elif i ==0:
                    input = x.transpose()

                x_grad = numpy.matmul(first_grad, model.layers[i].weight.transpose())
                w_grad = numpy.matmul(input, first_grad)
                b_grad = numpy.sum(first_grad, axis=0)


                if self.optimizer == "adam":
                    model = self.adam_optimizers[i].Adam_foward(model,i, w_grad, b_grad, epoch, lr, bias=model.layers[i].b)

                if self.optimizer == "sgd":
                    model = self.sgd.forward_SGD(model, i, w_grad, b_grad, lr, bias = model.layers[i].b)

                first_grad = x_grad

            if actual_layer == "Relu" or actual_layer == "TanH" or actual_layer == "Sigmoid":
                first_grad = first_grad * model.layers[i].backward(model.layers[i-1].x)

        return model
