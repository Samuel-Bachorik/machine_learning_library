import numpy
from model import Model
from dataset_loader import ImagesLoader
from save_load_model import *


testing_paths = []
#Append all testing folders
for z in range(10):
    testing_paths.append("C:/Users/Samuel/PycharmProjects/Deep_learning_library/MNIST/MNIST - JPG - testing/{}/".format(z))

if __name__ == '__main__':
    loader = ImagesLoader(512)
    dataset = loader.get_dataset(testing_paths, training=False)
    model = Model()

    save = save_load()

    save.load_weights(model, "WEIGHTS_BIG_SGD.npz")


    def test_model():
            good_predictions = 0
            
            for images, labels in (zip(*dataset)):
                images = images.reshape(512, 784)

                y = model.forward(images)
                y = numpy.argmax(y, axis=1)

                difference = numpy.equal(y, labels)

                for i in difference:
                    if i == True:
                        good_predictions+=1

            print("Accuracy is - ",(good_predictions)/100,"% and model missed", 10000- good_predictions,"of 10 000 images")

    test_model()
