# Fully automatic backpropagation algorithm trough various linear layers, activation functions and loss function

How to change weights of the model to get better results? The right question is - what are derivatives of weights with respect to loss function?<br/>

Backpropagation alorithm implemented in this library will give you answer to this question. <br/>

Build your model, put your data inside, run backward, save model weights, load model weights and test trained model in real life<br/>
# Tests on MNIST
### The course of the crossentropy loss function
Here is model loss optimized with `Adam` optimizer, with Adam we are getting average loss for epoch about 0.0003<br/>
<br/>
![MNIST Crossentropy loss ADAM](https://user-images.githubusercontent.com/61843287/213419750-538c88e2-0ba9-4f74-9bbb-8b709b22a03a.jpg)<br/>
Here is model loss optimized with basic `SGD` optimizer, with SGD we can not get less than 0.3859<br/>
<br/>
![MNIST Crossentropy loss SGD](https://user-images.githubusercontent.com/61843287/213420893-b8baa676-530a-4411-b1fe-3b234f162585.jpg)<br/>


### Measuring accuracy of trained models



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

