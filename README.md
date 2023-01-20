# Fully automatic backpropagation algorithm trough various linear layers, activation functions and loss function

   - How to change weights of the model to get better results? The right question is - what are derivatives of weights with respect to loss function?<br/>

   - Backpropagation alorithm implemented in this library will give you answer to this question. <br/>

   - Build your model, put your data inside, run backward, save model weights, load model weights and test trained model in real life<br/>
# Tests on hand written digits - MNIST
   - Library tests were performed on the MNIST dataset, which contains 60,000 training images and 10,000 test images.

### The course of the crossentropy loss function during training
This graph represents model loss optimized with `Adam` optimizer, after 50 epochs with Adam we are getting average loss for epoch about 0.0003<br/>
<br/>
![MNIST Crossentropy loss ADAM](https://user-images.githubusercontent.com/61843287/213419750-538c88e2-0ba9-4f74-9bbb-8b709b22a03a.jpg)<br/>
This graph represents model loss optimized with basic `SGD` optimizer, with SGD we can not get less than 0.3859<br/>
<br/>
![MNIST Crossentropy loss SGD](https://user-images.githubusercontent.com/61843287/213420893-b8baa676-530a-4411-b1fe-3b234f162585.jpg)<br/>


### Measuring accuracy of trained models

After sucessfull training, the accuracy of the model was tested on 10,000 test images that the model had never seen before.
   - Model acuraccy optimized with `Adam`optimizer is 99.79 % and model missed only 21 of 10 000 images
   - Model acuraccy optimized with `SGD` optimizer is 92.31 % and model missed 769 of 10 000 images



# Theory
...
## Linear layer
...
## Model
...
## Stochastic gradient descent
...
## Adam optimizer
...
## Loss functions
...
## Activation functions
...
## Training loop
...

