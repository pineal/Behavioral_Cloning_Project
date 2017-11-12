# Behavioral Cloning Project

# Behavioral Cloning

The goals / steps of this project are the following:

- Use the simulator to collect data of good driving behavior
- Build, a convolution neural network in Keras that predicts steering angles from images
- Train and validate the model with a training and validation set
- Test that the model successfully drives around track one without leaving the road
- Summarize the results with a written report
## Rubric Points

**Are all required files submitted?**
All required files are in the [GitHub repository](https://github.com/pineal/Behavioral_Cloning_Project), with model.py, drive.py, model.h5, writeup report and video.mp4.

The codes are fully functional in AWS EC2 instance and my local machine (Mac OSX) with carnd-term1 miniconda environment.

**Is the code usable and readable?**
The code is relatively simple, however still readable and of course usable. 

**Has an appropriate model architecture been employed for the task?**

Firstly I applied LeNet model in this project, but the result tends not very good. In consequence, I used NVIDIA model, which leading a good result. 

NVIDIA CNN architecture consists of 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully connected layers. (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)

![Fig1: LeNet architecture](https://www.researchgate.net/profile/Haohan_Wang/publication/282997080/figure/fig10/AS:305939199610894@1449952997905/Figure-10-Architecture-of-LeNet-5-one-of-the-first-initial-architectures-of-CNN.png)


In addition to it, added one cropping layer to reduce the size of image, augment data and speed up a little training process. A dropout layer after convolutional layers to reduce overfit. 



![NVIDIA CNN architecture. ](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png)


**Has an attempt been made to reduce overfitting of the model?**
Two attempts made to reduce overfitting: 

1. Split 20% training data for validation data in fitting process.
2. Add an additional dropout layer with 
3. Other preprocessing might also helpful like flipping, normalizing

**Have the model parameters been tuned appropriately?**
An Adam optimizer is used when compiling the configures, thus there is no learning rate needed to be tuned. 

**Is the training data chosen appropriately?**
I have run the training mode in simulator for more than 3 laps, carefully tried my best to keep the car in the center of road.
Moreover, I have retrained at some curve on the track that failed at previous tries to collect more data to train. 

**Is the solution design documented?**

At prototyping stage, I started with a very naive layer network and sample data provided by the course, the car crashed into water soon.

A second try was applying LeNet model and train by myself. Data was collected from only one lap run on simulator. Result was still not good: it drove well for a certain distance and stocked in hill. LeNet is not good enough for training in this complex secernio. 

Then I applied the NVIDIA model architecture. It ran a lot better, however it still stuck on the roadside at very late of the trip. I added some image augmentation like flipping the images, combining left, center and right cameras images and cropping them. Also, I collected more train data by reproducing some sharp turns, and the result is getting much better. 

**Is the model architecture documented?**

In the end, the architecture used here is Nvidia network, the architecture of Nvidia network has been illustrated and disccussed in the above section.

Here is the training result with 10 epochs, displaying the loss rate in each iteration. 


    Train on 33801 samples, validate on 8451 samples
    Epoch 1/10
    33801/33801 [==============================] - 88s - loss: 0.0213 - val_loss: 0.0183
    Epoch 2/10
    33801/33801 [==============================] - 87s - loss: 0.0174 - val_loss: 0.0164
    Epoch 3/10
    33801/33801 [==============================] - 87s - loss: 0.0159 - val_loss: 0.0169
    Epoch 4/10
    33801/33801 [==============================] - 87s - loss: 0.0146 - val_loss: 0.0144
    Epoch 5/10
    33801/33801 [==============================] - 87s - loss: 0.0137 - val_loss: 0.0145
    Epoch 6/10
    33801/33801 [==============================] - 87s - loss: 0.0127 - val_loss: 0.0143
    Epoch 7/10
    33801/33801 [==============================] - 87s - loss: 0.0118 - val_loss: 0.0135
    Epoch 8/10
    33801/33801 [==============================] - 87s - loss: 0.0111 - val_loss: 0.0136
    Epoch 9/10
    33801/33801 [==============================] - 87s - loss: 0.0102 - val_loss: 0.0133
    Epoch 10/10
    33801/33801 [==============================] - 87s - loss: 0.0095 - val_loss: 0.0132

**Is the creation of the training dataset and training process documented?**

Pictures from left, center and right cameras were all used. The angles measurement have been added some correction manually.

Images have been augmented by flipping and cropping and shuffling, they were normalized in the Nvidia network as well.

After pre-processing, training data size increased to 42252. By splitting them, 33801 of samples were remained as training data, 8451 samples became validation data. 

**Is the car able to navigate correctly on test data?**
The car can finish whole track on the simulator.

