# Fully automatic backpropagation algorithm implemeted in numpy trough various linear layers, activation functions and loss function

   - What are derivatives of weights with respect to loss function? How to update model's weights to get better predictions?<br/>

   - Backpropagation alorithm implemented in this library will give you answer to this questions. <br/>

   - Build your model, put your data inside, run backward, save model weights, load model weights and test trained model in real life<br/>
# Tests on hand written digits - MNIST
   - Library tests were performed on the MNIST dataset, which contains 60,000 training images and 10,000 test images.

### The course of the crossentropy loss function during training
This graph represents model loss optimized with `Adam` optimizer, after 50 epochs with Adam we are getting average loss for epoch about 0.0003<br/>
<br/>
![MNIST Crossentropy loss ADAM](https://user-images.githubusercontent.com/61843287/213419750-538c88e2-0ba9-4f74-9bbb-8b709b22a03a.jpg)<br/>
Next graph represents model loss optimized with basic `SGD` optimizer, with SGD we can not get less than 0.3859<br/>
<br/>
![MNIST Crossentropy loss SGD](https://user-images.githubusercontent.com/61843287/213420893-b8baa676-530a-4411-b1fe-3b234f162585.jpg)<br/>


### Measuring accuracy of trained models 

After sucessfull training, the accuracy of the model was tested on 10,000 test images that the model had never seen before.
   - Model acuraccy optimized with `Adam`optimizer is 99.79 % and model missed only 21 of 10 000 images
   - Model acuraccy optimized with `SGD` optimizer is 92.31 % and model missed 769 of 10 000 images



# Theory
   - Example of forward and backward on three layer computational graph
   - MSE loss at the end of model
   - Chart and equations are made with lucidchart

   
   


![image](https://user-images.githubusercontent.com/61843287/213939301-6e8e1942-fe19-489d-95a7-d1c792ad7061.png)
<br/>
- Next image is forward and backward computation corresponding to previous computational graph
- SGD optimalization of two weights

![image](https://user-images.githubusercontent.com/61843287/214009611-91d0e47c-0b30-4989-be0a-a53deb223caa.png)


## Training process explained

![training loop](https://user-images.githubusercontent.com/61843287/213724773-14531b68-46d0-4d46-b841-e0c352e3ce50.JPG)
Chart made with lucidchart


## Linear layer
   - Linear layer applies a linear transformation to the incoming data x
   - x, W and B are tensors 
   - T = transposed matrix


![image](https://user-images.githubusercontent.com/61843287/213730494-7d6dbb1b-74d9-49b6-91e6-47cd0d224de0.png)

## Stochastic gradient descent
Intiution behind optimizer Adam- https://www.geeksforgeeks.org/intuition-of-adam-optimizer/
