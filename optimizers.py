import numpy as np

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
