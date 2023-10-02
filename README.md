# Face-mask-detection-vgg19
We are going to build a face mask detector using pretrained VGG19 network on imagenet.

We have the .ipynb file which the construction of our model and the realTimeDetection file where everything is integrated there together, our saved model and real time detections.

- We start off by augmenting the data we have using Image Data Generator in keras and specifying the directories of the images.
- Then we fit the model after adding a dense layer for the output and train the model using the data genereted.
- Finally we use openCV to get real time detection of the dace using the haar cascade of the frontal face and combine it with our saved model to check whether the face has a mask on or no.
