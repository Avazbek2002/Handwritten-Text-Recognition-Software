# HandWritten Text Recognition Software

The project is aimed for mainly learning how different kinds of Neural Networks work.
The ***input*** picture must be in the size of ***128x32***.
The ***output*** is in the form of a ***string***.

## Model Overview
> The model basically consists of neural networks of different types.
![image](https://user-images.githubusercontent.com/64166521/146567736-09b93582-eef4-49de-a13b-4b0781023823.png)

> - 5 layers of Convolutional Neural Network (CNN) followed by ReLu and MaxPool operation.
> - 2 layers of Recurrent Neural Network (RNN)
> - 1 layer of dilated CNN
> - dense layer
> - CTC loss and CTC decode function

***The detailed explanation of every layers are in my [Project Documentation](file:///C:/Users/avazb/Desktop/Handwritten%20text%20classification%20software.pdf)***

## References
Scheidl, Harald. “Build a Handwritten Text Recognition System Using TensorFlow.” Medium, 9 Aug. 2020, [link to the website](https://towardsdatascience.com/build-a-handwritten-text-recognition-system-using-tensorflow-2326a3487cd5)
