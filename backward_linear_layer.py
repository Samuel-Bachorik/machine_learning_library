import numpy
import numpy as np
from loss_functions import *
from optimizers import *

class backward_linear:
    def __init__(self, optimizer, model):

        self.optimizer = optimizer
        print(self.optimizer)
        
        #Check if choosen optimizer is right
        if self.optimizer != "sgd" and self.optimizer != "adam":
            raise Exception("Wrong optimizer selected use 'adam' or 'sgd'")
 
        if optimizer == "adam":
            print("You are using Adam optimizer \n")
            self.adam_optimizers = []
            # Create Adam optimizers for every layer in model
            for i in range(len(model.layers)):
                if model.layers[i].__class__.__name__ == "Linearlayer":
                    self.adam_optimizers.append(Adam())

                else:
                    # if not linear layer, append nothing
                    self.adam_optimizers.append(None)

        if self.optimizer == "sgd":
            print("You are using SGD optimizer \n")
            self.sgd = SGD()


    def run_backward(self,model, y_pred , y, lr, x, epoch, loss):
        # Calculate derivative of loss function
        first_grad = loss.backward(y, y_pred)
        
        # Loop trought model layers
        for i in range(len(model.layers) -1, -1, -1):
            actual_layer = model.layers[i].__class__.__name__

            if actual_layer == "Linearlayer":
                # If we are not in the begginig of model, input needs to be output from previous layer
                if i > 0:
                    input = model.layers[i-1].x.transpose()
                    
                # If we are at the beginig of model input needs to be x
                elif i ==0:
                    input = x.transpose()
                
                # Compute gradients
                x_grad = numpy.matmul(first_grad, model.layers[i].weight.transpose())
                w_grad = numpy.matmul(input, first_grad)
                b_grad = numpy.sum(first_grad, axis=0)

                # Update weights 
                if self.optimizer == "adam":
                    model = self.adam_optimizers[i].Adam_foward(model,i, w_grad, b_grad, epoch, lr, bias=model.layers[i].b)

                if self.optimizer == "sgd":
                    model = self.sgd.forward_SGD(model, i, w_grad, b_grad, lr, bias = model.layers[i].b)

                first_grad = x_grad
            
            #If activation layer, compute gradient of actual activation
            if actual_layer == "Relu" or actual_layer == "TanH" or actual_layer == "Sigmoid":
                first_grad = first_grad * model.layers[i].backward(model.layers[i-1].x)

        return model
