# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 13:04:47 2018

@author: krishna raj
"""

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation 
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as k

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        
        if k.image_data_format() == "channels_first":
            inputShape = (depth,width,height)
        #first set of layers    
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        #second set of layers
        
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        #Flattening layer and fully connected layer
        
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("softmax"))
        
        # softmax classifier
        
        model.add(Dense(classes))   #no of nodes is no of output classes
        model.add(Activation("softmax"))
        
        return model
    
    
        
    
