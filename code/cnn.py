from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data, get_next_batch
from manual_convolution import ManualConv2d
from base_model import CifarModel

import os
import tensorflow as tf
import numpy as np
import random
import math

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class CNN(CifarModel):
    def __init__(self, classes):
        """
        This model class will contain the architecture for your CNN that
        classifies images. Do not modify the constructor, as doing so
        will break the autograder. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(CNN, self).__init__()

        # Initialize all hyperparameters
        self.loss_list = []
        self.batch_size = 64
        self.input_width = 32
        self.input_height = 32
        self.image_channels = 3
        self.num_classes = len(classes)

        self.hidden_layer_size = 320

        self.epsilon = 1e-3  # this is used for batch normalization only!

        # Fill the rest of this out!
        self.filters = 32
        self.kernel_size = 3
        self.padding = "same"
        self.pool_size=2

        self.conv_1 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, padding=self.padding)
        self.bn_1   = tf.keras.layers.BatchNormalization()
        self.relu_1 = tf.keras.layers.ReLU()
        self.pool_1 = tf.keras.layers.MaxPooling2D(self.pool_size)
        self.conv_2 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, padding=self.padding)
        self.bn_2   = tf.keras.layers.BatchNormalization()
        self.relu_2 = tf.keras.layers.ReLU()
        self.pool_2 = tf.keras.layers.MaxPooling2D(self.pool_size)
        self.conv_3 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, padding=self.padding)
        self.bn_3   = tf.keras.layers.BatchNormalization()
        self.relu_3 = tf.keras.layers.ReLU()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden_layer_1 = tf.keras.layers.Dense(self.hidden_layer_size,activation='relu')
        self.hidden_layer_2 = tf.keras.layers.Dense(self.hidden_layer_size,activation='relu')
        self.hidden_layer_3 = tf.keras.layers.Dense(self.hidden_layer_size,activation='relu')
        self.output_layer = tf.keras.layers.Dense(self.num_classes, activation=None)

    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, num_classes)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)
        x = inputs
        x = self.conv_1(x)
        x = self.bn_1(x)  
        x = self.relu_1(x)
        x = self.pool_1(x) 
        x = self.conv_2(x)
        x = self.bn_2(x)   
        x = self.relu_2(x)
        x = self.pool_2(x)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu_3(x)
        x = self.flatten(x)
        x = self.hidden_layer_1(x)
        if not is_testing:
            x = tf.nn.dropout(x, rate=0.3)
        x = self.hidden_layer_2(x)
        if not is_testing:
            x = tf.nn.dropout(x, rate=0.3)
        x = self.hidden_layer_3(x) 
        x = self.output_layer(x)
        return x

