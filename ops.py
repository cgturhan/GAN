# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 14:38:24 2017

@author: pc2
"""

import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops


def conv2d(input_, output_size, k_h = 5, k_w = 5, d_h = 2, d_w = 2, stddev = 0.02, name = "conv2d"):
    
    with tf.variable_scope(name):
        filterc = tf.get_variable("filterc", [k_h, k_w, input_.get_shape()[-1], output_size], initializer = tf.truncated_normal_initializer(stddev = stddev))     
        conv = tf.nn.conv2d(input_, filterc, strides = [ 1, d_h, d_w, 1], padding = "SAME")
        biasc = tf.get_variable('biasc', [output_size], initializer=tf.constant_initializer(0.0))
        conv_result = tf.nn.bias_add(conv, biasc)
        return conv_result
        
def deconv2d(input_, output_shape, k_h = 5, k_w = 5, d_h = 2, d_w = 2, stddev = 0.02, name = "deconv2d" , with_weights = False):
    with tf.variable_scope(name):   
        w = tf.get_variable("filters", [k_h, k_w, output_shape[-1], input_.get_shape()[-1],], initializer = tf.truncated_normal_initializer(stddev = stddev))     
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,strides = [ 1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_weights:
            return deconv, w, biases
        else:
            return deconv
    
def lrelu(input_, leak = 0.2, name = "lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        output_ = f1 * input_ + f2 * abs(input_)        
        return output_
        

def linear(input_, output_size, scope = None, stddev = 0.2, bias_start = 0.0, with_weights = False):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        weights = tf.get_variable("weights", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], initializer = tf.constant_initializer(bias_start))
        
        if with_weights:
            return tf.matmul(input_, weights) + bias, weights, bias
        else:
            return tf.matmul(input_, weights) + bias
            
                            
def fully_connected(input_, output_size, scope = None, stddev = 1.0, with_bias = True):
    shape = input_.get_shape().as_list()       

    with tf.variable_scope(scope or "FC"):
          weight = tf.get_variable("weight", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
          result = tf.matmul(input_, weight)
          
          if with_bias:
              b = tf.get_variable("b", [1, output_size], initializer = tf.random_normal_initializer(stddev = stddev))
              
              result = result + b * tf.ones([shape[0], 1], dtype = tf.float32)
              
          return result
          
def binary_cross_entropy_with_logits(logits, targets, name=None):
    """Computes binary cross entropy given `logits`.
    For brevity, let `x = logits`, `z = targets`.  The logistic loss is
        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))
    Args:
        logits: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `logits`.
    """
    eps = 1e-12
    with ops.op_scope([logits, targets], name, "bce_loss") as name:
        logits = ops.convert_to_tensor(logits, name="logits")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(logits * tf.log(targets + eps) + (1. - logits) * tf.log(1. - targets + eps)))

class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, batch_size, epsilon=1e-5, momentum = 0.1, name="batch_norm"):
        with tf.variable_scope(name) as scope:
            self.epsilon = epsilon
            self.momentum = momentum
            self.batch_size = batch_size

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name=name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        with tf.variable_scope(self.name) as scope:
            self.gamma = tf.get_variable("gamma", [shape[-1]],
                                initializer=tf.random_normal_initializer(1., 0.02))
            self.beta = tf.get_variable("beta", [shape[-1]],
                                initializer=tf.constant_initializer(0.))

            self.mean, self.variance = tf.nn.moments(x, [0, 1, 2])

            return tf.nn.batch_norm_with_global_normalization(
                x, self.mean, self.variance, self.beta, self.gamma, self.epsilon,scale_after_normalization=True)          