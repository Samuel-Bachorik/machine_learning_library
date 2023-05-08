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
    
    batch_size  = 512
    loader      = ImagesLoader(batch_size)
    dataset     = loader.get_dataset(training_paths, training=False)

    lr          = 0.001
    model       = Model()
    backward    = backward_linear("adam", model)

    num_epoch    = 50

    #Define loss function
    criterion   = CrossEntropyLoss()
    
    #Class for showing loss on graph
    batch_count_train   = (60000 + batch_size) // batch_size
    loss_chart          = CreateGraph(batch_count_train, "MNIST Crossentropy loss")

    print("Training started...")
    count = 0
    #Epoch loop
    for epoch in range(num_epoch):
        #Batch loop
        for images, labels in (zip(*dataset)):
            # Reshape from (batch,1,28,28) to (batch,784) for linear layer
            images = images.reshape(batch_size, 784)
            # Push images to model
            y_pred = model.forward(images)
            # Compute loss
            loss = criterion.forward(labels, y_pred)
            # Graph adding
            loss_chart.num_for_avg += loss
            #Run backward
            model = backward.run_backward(model, y_pred, labels, lr, images, epoch, criterion)

        loss_chart.count(epoch)
 
    # Save model weights after training
    save = save_load()
    save.save_weights(model,"WEIGHTS_BIG_SGD")
