from model import Model
from backward_linear_layer import *
from dataset_loader import ImagesLoader
from loss_functions import *
from save_load_model import *

training_paths = []

#Append all training folders
for z in range(10):
    training_paths.append("C:/Users/Samuel/PycharmProjects/Deep_learning_library/MNIST/MNIST - JPG - training/{}/".format(z))


if __name__ == '__main__':

    batch_size = 256
    loader = ImagesLoader(batch_size)
    dataset = loader.get_dataset(training_paths, training=False)

    lr = 0.01

    model = Model()
    backward = backward_linear("sgd",model)

    lossum, num_epoch = 0, 5

    criterion = CrossEntropyLoss()

    print("Training started...")
    count = 0
    for epoch in range(num_epoch):
        for images, labels in (zip(*dataset)):

            images = images.reshape(batch_size, 784)

            y_pred = model.forward(images)

            loss = criterion.forward(labels, y_pred)

            lossum += loss

            model = backward.run_backward(model, y_pred, labels, lr, images, epoch, criterion)


        print(numpy.argmax(y_pred[epoch]))
        print(labels[epoch])


        print("Average loss for epoch ", epoch + 1, " - ", lossum / len(dataset[0]))
        lossum = 0

    save = save_load()
    save.save_weights(model,"WEIGHTS")

    save.load_weights(model, "WEIGHTS.npz")



