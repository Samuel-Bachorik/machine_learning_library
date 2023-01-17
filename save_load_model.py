import numpy as np

class save_load:

    def load_weights(self,model,filename):

        data = np.load(str(filename))
        lst = data.files

        counter = 0

        for i in range(len(lst)):

            if model.layers[counter].__class__.__name__ == "Linearlayer":
                model.layers[counter].weight = data[lst[i]]

                if model.layers[counter].b:
                    model.layers[counter].bias = data[lst[i+1]]

                    i+=1

            if counter == len(model.layers)-1:
                break

            counter += 1



    def save_weights(self,model,filename):

        model_weights = {}

        for i in range(len(model.layers)):
            if model.layers[i].__class__.__name__ == "Linearlayer":
                model_weights[str(i)] = model.layers[i].weight

                if model.layers[i].b:
                    model_weights[str(i+1)] = model.layers[i].bias

                    i+=1

        np.savez_compressed(str(filename), **model_weights)

