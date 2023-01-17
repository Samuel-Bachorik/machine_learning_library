import numpy
import numpy as np
from loss_functions import *


class SGD:
    def forward_SGD(self, model, index, dw, db, lr, bias = True):
        #Weights
        w_new = model.layers[index].weight - lr * dw
        model.layers[index].weight = w_new

        #Bias
        if bias:
            b_new = model.layers[index].bias - lr * db
            model.layers[index].bias = b_new

        return model



class Adam:
    def __init__(self):

        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0

    def Adam_foward(self, model ,index, dw, db, epoch, lr, bias = True, betas=(0.9, 0.999), eps=1e-08):

        #Weight

        self.m_dw   = betas[0] * self.m_dw + (1 - betas[0]) * dw
        self.v_dw   = betas[1] * self.v_dw + (1 - betas[1]) * (dw ** 2)

        mdw_corr    = self.m_dw / (1 - np.power(betas[0], epoch + 1))
        vdw_corr    = self.v_dw / (1 - np.power(betas[1], epoch + 1))

        model.layers[index].weight -= (lr / (np.sqrt(vdw_corr + eps))) * mdw_corr

        #Bias

        if bias:
            self.m_db   = betas[0] * self.m_db + (1 - betas[0]) * db
            self.v_db   = betas[1] * self.v_db + (1 - betas[1]) * (db ** 2)

            mdb_corr    = self.m_db / (1 - np.power(betas[0], epoch + 1))
            vdb_corr    = self.v_db / (1 - np.power(betas[1], epoch + 1))

            model.layers[index].bias -= (lr / (np.sqrt(vdb_corr + eps))) * mdb_corr


        return model

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
                    model = self.adam_optimizers[i].Adam_foward(model,i, w_grad, b_grad, epoch, lr, bias=False)

                if self.optimizer == "sgd":
                    model = self.sgd.forward_SGD(model, i, w_grad, b_grad, lr, bias = True)


                first_grad = x_grad

            if actual_layer == "Relu" or actual_layer == "TanH" or actual_layer == "Sigmoid":
                first_grad = first_grad * model.layers[i].backward(model.layers[i-1].x)

        return model