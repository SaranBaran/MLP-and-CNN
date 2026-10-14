from __future__ import absolute_import
from preprocess import unpickle, get_next_batch, get_data

import os
import tensorflow as tf
import numpy as np
import random
import math


class ManualConv2d(tf.keras.layers.Layer):
    def __init__(self, filter_shape: list[int], strides: list[int]=[1,1,1,1], padding = "VALID", use_bias = True, trainable=True, *args, **kwargs):
        """
        :param filter_shape: list of [filter_height, filter_width, in_channels, out_channels]
        :param strides: MUST BE [1, 1, 1, 1] - list of strides, with each stride corresponding to each dimension in input
        :param padding: either "SAME" or "VALID", capitalization matters
        """
        super().__init__()

        self.strides = strides
        self.padding = padding

        def get_var(name, shape, trainable):
            return tf.Variable(tf.random.truncated_normal(shape, dtype=tf.float32, stddev=1e-1), name=name, trainable = trainable)

        self.filters = get_var("conv_filters", filter_shape, trainable)
        self.use_bias = use_bias
        if use_bias: self.bias = get_var("conv_bias", [filter_shape[-1]], trainable)
        else: self.bias = None

    def get_weights(self):
        if self.bias is not None: return self.filters, self.bias
        return self.filters

    def set_weights(self, filters, bias=None): 
        self.filters = filters
        if bias is not None: self.bias = bias

    def call(self, inputs):
        """
        :param inputs: tensor with shape [num_examples, in_height, in_width, in_channels]
        """

        #define some useful variables
        num_examples, in_height, in_width, input_in_channels = inputs.shape
        filter_height, filter_width, filter_in_channels, filter_out_channels = self.filters.shape

        # fill out the rest!
        filters = tf.cast(self.filters, tf.float32)
        bias = tf.cast(self.bias, tf.float32) if self.use_bias else None
        if input_in_channels != filter_in_channels:
            raise ValueError("number of input in channels are not the same as the filter’s in channels")
        if self.padding == "SAME":
            pad_total_h = filter_height - 1
            pad_total_w = filter_width - 1
            pad_top = pad_total_h // 2            
            pad_bottom = pad_total_h - pad_top
            pad_left = pad_total_w // 2
            pad_right = pad_total_w - pad_left
            inputs = tf.pad(inputs, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
        out_height = inputs.shape[1] - filter_height + 1
        out_width = inputs.shape[2] - filter_width + 1
        rows = []
        for i in range(out_height):            
            row = []
            for j in range(out_width):         
                patch = inputs[:, i:i + filter_height, j:j + filter_width, :]
                value = tf.tensordot(patch, filters, axes=[[1, 2, 3], [0, 1, 2]])
                row.append(value)
            rows.append(tf.stack(row, axis=1))
        output = tf.stack(rows, axis=1)
        if self.use_bias:
            output = output + bias
        return tf.convert_to_tensor(output, dtype=tf.float32)
