import numpy as np

class save_load:

    def load_weights(self,model,filename):
        data = np.load(str(filename))
        lst = data.files

        counter = 0
        for i in range(len(model.layers)):
            if model.layers[i].__class__.__name__ == "Linearlayer":
                model.layers[i].weight = data[lst[counter]]
                counter+=1

                if model.layers[i].b:
                    model.layers[i].bias = data[lst[counter]]
                    counter+=1


    def save_weights(self,model,filename):

        model_weights = {}
        counter = 0
        for i in range(len(model.layers)):
            if model.layers[i].__class__.__name__ == "Linearlayer":
                model_weights[str(counter)] = model.layers[i].weight
                counter+=1

                if model.layers[i].b:
                    model_weights[str(counter)] = model.layers[i].bias
                    counter+=1


        np.savez_compressed(str(filename), **model_weights)
