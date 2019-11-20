# How to run the webapp


## Overview

This is the code for detecting cat and dog in the image. It uses an object classification model.

## Description

Here, our task is to detect whether a cat or dog is present in an image or not.

There are mainly four case to cover:
 1.) Both cat and dog are present.
 2.) Only cat is present.
 3.) Only dog is present.
 4.) None of the animal is present.
 
 We can use a classification or object detection model for this use case. This code employs an
 object classification convolutional neural network. We train the network and store the weights.
 The network is trained on [this](https://www.kaggle.com/c/dogs-vs-cats)
 dataset which contains images of only cats or dogs (separate).
 
 This model performs bad at the 'first' and the 'last' use case as the dataset doesn't
 contain any image containing both dog and cat or not containing any.
 But, if a suitable dataset is provided, this model performs fine and is faster than the object detection
 algorithm (YOLO).
 In this repo, we also attach the weights which are loaded when webapp is run.
 We preprocess the images and feed into our network. The network gives two outputs 
 (one gives cat's probability and one provides probability of a dog being present)
 and uses binary cross-entropy as loss function with sigmoid (instead of softmax)
 as the activation function at output layer. This is done since the two outputs are
 uncorrelated.

## Dependencies

```sudo pip install -r requirements.txt```

## Usage

Once dependencies are installed, just run this to see it in your browser. 

```python app.py```

That's it! It's serving a saved model from our own network in train.py via Flask on port 5001 of your localhost. 


