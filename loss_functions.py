import numpy as np
 
class MSE_loss:
    def forward(self, y, y_pred):
        loss = ((y - y_pred) ** 2).mean()

        return loss

    def backward(self,y, y_pred):
        N = y.shape[0]
        grad = -2 * (y - y_pred) / N

        return grad


class CrossEntropyLoss:

    def softmax(self,x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, y, y_pred):

        softmax = self.softmax(y_pred)
        loss = -np.log(softmax[range(len(y)), y])

        return loss.mean()


    def backward(self,y, y_pred):
        softmax = self.softmax(y_pred)

        softmax[range(len(y)), y] -= 1.0
        softmax /= len(y)

        grad = softmax

        return grad

