from model import Model
from backward_linear_layer import *
from dataset_loader import ImagesLoader
from loss_functions import *
from save_load_model import *
from graphing_class import CreateGraph
training_paths = []

#Append all training folders
for z in range(10):
    training_paths.append("C:/Users/Samuel/PycharmProjects/Deep_learning_library/MNIST/MNIST - JPG - training/{}/".format(z))


if __name__ == '__main__':

    batch_size = 512
    loader = ImagesLoader(batch_size)
    dataset = loader.get_dataset(training_paths, training=False)

    lr = 0.001

    model    = Model()
    backward = backward_linear("sgd",model)

   num_epoch = 50

    #Define loss function
    criterion = CrossEntropyLoss()

    batch_count_train = (60000 + batch_size) // batch_size
    loss_chart = CreateGraph(batch_count_train, "MNIST Crossentropy loss")

    print("Training started...")
    count = 0
    for epoch in range(num_epoch):
        for images, labels in (zip(*dataset)):

            images = images.reshape(batch_size, 784)

            y_pred = model.forward(images)

            loss = criterion.forward(labels, y_pred)

            loss_chart.num_for_avg += loss

            model = backward.run_backward(model, y_pred, labels, lr, images, epoch, criterion)

        loss_chart.count(epoch)

    # Save model weights after training
    save = save_load()
    save.save_weights(model,"WEIGHTS_BIG_SGD")
