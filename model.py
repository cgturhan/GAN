# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 15:57:50 2017

@author: pc2
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
from PIL import Image
import os
from ops import *

class CPPN():
    def __init__(self, x_dim = 32, y_dim = 32, z_dim = 20, batch_size = 1, c_dim = 1, node_size = 32, scale = 10.0):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.batch_size = batch_size
        self.node_size = node_size
        self.scale = scale
       
        self.batch = tf.placeholder(tf.float32, [self.batch_size, self.x_dim, self.y_dim, self.c_dim])
        self.z = tf.placeholder(tf.float32, [self.batch_size,self.z_dim])
        self.x_vec, self.y_vec, self.r_vec = self.coordinates(self.x_dim, self.y_dim, self.scale)
        
        self.x = tf.placeholder(tf.float32, [self.batch_size, None, 1])
        self.y = tf.placeholder(tf.float32, [self.batch_size, None, 1])
        self.r = tf.placeholder(tf.float32, [self.batch_size, None, 1])
        
        self.G = self.generator(x_dim = x_dim, y_dim = y_dim)
        self.init()
        
    def init(self):
        init = tf.initialize_all_variables()
        print ('Initialization is completed!')
        self.sess = tf.Session()
        self.sess.run(init)
        
    def reinit(self):
        init = tf.initialize_all_variables(tf.trainable_variables())
        self.sess.run(init)
    
    def coordinates(self, x_dim = 32, y_dim = 32, scale = 1.0):
        n_points = x_dim * y_dim
        x_range = scale * (np.arange(x_dim)-(x_dim-1)/2.0)/(x_dim-1)/0.5
        y_range = scale * (np.arange(y_dim)-(y_dim-1)/2.0)/(y_dim-1)/0.5
        x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
        y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
        r_mat = np.sqrt(x_mat*x_mat + y_mat*y_mat)
        x_mat = np.tile(x_mat.flatten(), self.batch_size).reshape(self.batch_size, n_points, 1)
        y_mat = np.tile(y_mat.flatten(), self.batch_size).reshape(self.batch_size, n_points, 1)
        r_mat = np.tile(r_mat.flatten(), self.batch_size).reshape(self.batch_size, n_points, 1)
        return x_mat, y_mat, r_mat
        
    def generator(self, x_dim, y_dim, reuse = False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
            
        node_size = self.node_size
        n_points = x_dim * y_dim
        
        z_scaled = tf.reshape(self.z, [self.batch_size, 1, self.z_dim]) *\
                        tf.ones([n_points,1], dtype = tf.float32) * self.scale
                        
        z_unroll = tf.reshape(z_scaled, [self.batch_size * n_points, self.z_dim])
        x_unroll = tf.reshape(self.x, [self.batch_size * n_points, 1])
        y_unroll = tf.reshape(self.y, [self.batch_size * n_points, 1])
        r_unroll = tf.reshape(self.r, [self.batch_size * n_points, 1])
        
        U = self.fully_connected(z_unroll, node_size, 'g_0_z') + \
            self.fully_connected(x_unroll, node_size, 'g_0_x', with_bias = False) + \
            self.fully_connected(y_unroll, node_size, 'g_0_y', with_bias = False) + \
            self.fully_connected(r_unroll, node_size, 'g_0_r', with_bias = False)                
                        
        H = tf.nn.tanh(U)
        for i in range(3):
            H = self.fully_connected(H, node_size, 'g_tanh_'+str(i)) 
            H = tf.nn.tanh(H) 
        output = tf.sigmoid(self.fully_connected(H, self.c_dim, 'g_final'))
        result = tf.reshape(output, [self.batch_size, y_dim, x_dim, self.c_dim])
        return result
        
                                           
    def fully_connected(self, input_, output_size, scope = None, stddev = 1.0, with_bias = True):
        shape = input_.get_shape().as_list()       

        with tf.variable_scope(scope or "FC"):
              weight = tf.get_variable("Weight", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
              result = tf.matmul(input_, weight)
              
              if with_bias:
                  bias = tf.get_variable("Bias", [1, output_size], initializer = tf.random_normal_initializer(stddev = stddev))
                  
                  result = result + bias * tf.ones([shape[0], 1], dtype = tf.float32)
                  
              return result
    
    def generate(self, z=None, x_dim = 32, y_dim = 32, scale = 5.0):
        if z is None:
            z = self.generate_z()
        
        G = self.generator(x_dim = x_dim, y_dim = y_dim, reuse = True)
        x, y, r = self.coordinates(x_dim, y_dim, scale = scale)
        image = self.sess.run(G, feed_dict={self.z: z, self.x: x, self.y: y, self.r: r})
        return image
        
    def generate_z(self):
         z = np.random.uniform(-1.0, 1.0, size = (self.batch_size, self.z_dim)).astype(np.float32)
         return z
    
    def close(self):
        self.sess.close()

class CPPNGAN():
    def __init__(self, x_dim = 32, y_dim = 32, z_dim = 20, d_dim = 32, g_dim = 32, batch_size = 1, c_dim = 1, scale = 10.0,
                 learning_rate_g = 0.01, learning_rate_d = 0.001, beta1 = 0.9, net_size_g = 128, net_depth_g = 4, keep_prob = 1.0, model_name = 'cppngan' ):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.batch_size = batch_size
        self.net_size_g = net_size_g
        self.net_depth_g = net_depth_g
        self.d_dim = d_dim
        self.g_dim = g_dim
        self.learning_rate_d = learning_rate_d
        self.learning_rate_g = learning_rate_g
        self.scale = scale
        self.keep_prob = keep_prob
        self.model_name = model_name
        self.beta1 = beta1
        self.chekpoint_file = 'save/' + 'cppngan'

        
        
        self.batch = tf.placeholder(tf.float32, [self.batch_size, self.x_dim, self.y_dim, self.c_dim])
        #self.z = tf.placeholder(tf.float32, [self.batch_size,self.z_dim])
        
        self.z = tf.random_normal((self.batch_size, self.z_dim), 0, 1, dtype=tf.float32)
        self.x_vec, self.y_vec, self.r_vec = self.coordinates(self.x_dim, self.y_dim, self.scale)
        
        self.x = tf.placeholder(tf.float32, [self.batch_size, None, 1])
        self.y = tf.placeholder(tf.float32, [self.batch_size, None, 1])
        self.r = tf.placeholder(tf.float32, [self.batch_size, None, 1])
        
        
        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(batch_size, name=self.model_name+'_d_bn1')
        self.d_bn2 = batch_norm(batch_size, name=self.model_name+'_d_bn2')
        
        self.G = self.generator(x_dim = x_dim, y_dim = y_dim)
        self.D_real = self.discriminator(self.batch)
        self.D_fake = self.discriminator(self.G, reuse = True)
        
        self.d_loss_real = self.binary_cross_entropy_with_logits(tf.ones_like(self.D_real), self.D_real)
        self.d_loss_fake = self.binary_cross_entropy_with_logits(tf.zeros_like(self.D_fake), self.D_fake)
        self.d_loss = 1.0*(self.d_loss_real + self.d_loss_fake)/ 2.0
        self.g_loss = 1.0*self.binary_cross_entropy_with_logits(tf.ones_like(self.D_fake), self.D_fake)
        
        self.t_vars = tf.trainable_variables()
        self.g_vars = [var for var in self.t_vars if (self.model_name+'_g_') in var.name]
        self.d_vars = [var for var in self.t_vars if (self.model_name+'_d_') in var.name]
        
        # Use ADAM optimizer
        self.g_opt = tf.train.AdamOptimizer(self.learning_rate_g, beta1=self.beta1) \
                      .minimize(self.g_loss, var_list=self.g_vars)

        self.d_opt = tf.train.AdamOptimizer(self.learning_rate_d, beta1=self.beta1) \
                      .minimize(self.d_loss, var_list=self.d_vars)

        
        self.init()
        
    def init(self):
        init = tf.initialize_all_variables()
        print ('Initialization is completed!')
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
        self.saver = tf.train.Saver(tf.all_variables())
        
    def reinit(self):
        init = tf.initialize_all_variables(tf.trainable_variables())
        self.sess.run(init)
    
    def coordinates(self, x_dim = 32, y_dim = 32, scale = 1.0):
        n_points = x_dim * y_dim
        x_range = scale * (np.arange(x_dim)-(x_dim-1)/2.0)/(x_dim-1)/0.5
        y_range = scale * (np.arange(y_dim)-(y_dim-1)/2.0)/(y_dim-1)/0.5
        x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
        y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
        r_mat = np.sqrt(x_mat*x_mat + y_mat*y_mat)
        x_mat = np.tile(x_mat.flatten(), self.batch_size).reshape(self.batch_size, n_points, 1)
        y_mat = np.tile(y_mat.flatten(), self.batch_size).reshape(self.batch_size, n_points, 1)
        r_mat = np.tile(r_mat.flatten(), self.batch_size).reshape(self.batch_size, n_points, 1)
        return x_mat, y_mat, r_mat
        
    def generator(self, x_dim, y_dim, reuse = False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
            
        node_size = self.net_size_g
        n_points = x_dim * y_dim
        
        z_scaled = tf.reshape(self.z, [self.batch_size, 1, self.z_dim]) *\
                        tf.ones([n_points,1], dtype = tf.float32) * self.scale
                        
        z_unroll = tf.reshape(z_scaled, [self.batch_size * n_points, self.z_dim])
        x_unroll = tf.reshape(self.x, [self.batch_size * n_points, 1])
        y_unroll = tf.reshape(self.y, [self.batch_size * n_points, 1])
        r_unroll = tf.reshape(self.r, [self.batch_size * n_points, 1])
        
        U = fully_connected(z_unroll, node_size, self.model_name+'_g_0_z') + \
            fully_connected(x_unroll, node_size, self.model_name+'_g_0_x', with_bias = False) + \
            fully_connected(y_unroll, node_size, self.model_name+'_g_0_y', with_bias = False) + \
            fully_connected(r_unroll, node_size, self.model_name+'_g_0_r', with_bias = False)                
                        
        H = tf.nn.tanh(U)
        for i in range(3):
            H = fully_connected(H, node_size, self.model_name+'_g_tanh_'+str(i)) 
            H = tf.nn.tanh(H) 
        output = tf.sigmoid(fully_connected(H, self.c_dim, self.model_name+'_g_final'))
        result = tf.reshape(output, [self.batch_size, y_dim, x_dim, self.c_dim])
        return result
    
    def discriminator(self, image, reuse = False):
        
        if reuse:
            tf.get_variable_scope().reuse_variables()
            
        h0 = lrelu(conv2d(image, self.d_dim, name = self.model_name+'_d_h0_conv') )
        h1 = lrelu(self.d_bn1(conv2d(h0, self.d_dim*2,  name = self.model_name+'_d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.d_dim*4,  name = self.model_name+'_d_h2_conv')))
        h3 = linear(tf.reshape(h2, [self.batch_size, -1]), 1, self.model_name+'_d_h2_lin')         
        out = tf.nn.sigmoid(h3)    
        return out
        
    def train(self, num_epoch = 100, display_step = 10, checkpoint_step = 10):
        dirname = 'save'
        ckpt = tf.train.get_checkpoint_state(dirname)
        if ckpt:
            self.load_model(dirname)
            print "Model is loaded!"
            
        images, labels = read_dataset()
        n_samples = len(labels)

        
        counter1 = 0
        for epoch in range(num_epoch):
            avg_d_loss = 0.
            avg_g_loss = 0.
            #batch images 
            batch_idxs = len(images) // self.batch_size
            for idx in xrange(0, batch_idxs):
                batch_images = images[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_labels = labels[idx*self.batch_size:(idx+1)*self.batch_size]
                #z = self.generate_z()
                counter2 = 0
                for i in range(4):
                    counter2 += 1
                    _, g_loss = self.sess.run((self.g_opt, self.g_loss), feed_dict = {self.batch: batch_images, self.x: self.x_vec, self.y: self.y_vec, self.r : self.r_vec})#, self.z :z})                  
                    if g_loss < 0.6:
                        break
                d_loss = self.sess.run(self.d_loss, feed_dict = {self.batch:batch_images, self.x: self.x_vec, self.y: self.y_vec, self.r : self.r_vec}) #, self.z :z})
                
                if d_loss > 0.45 and g_loss < 0.8:              
                    for i in range(1):
                        counter2 += 1
                        _, d_loss = self.sess.run((self.d_opt, self.d_loss), feed_dict={self.batch: batch_images, self.x: self.x_vec, self.y: self.y_vec, self.r: self.r_vec})#, self.z:z})
                        if d_loss < 0.6:
                            break

                #return d_loss, g_loss, vae_loss, counter2
                assert( d_loss < 1000000 ) # make sure it is not NaN or Inf
                assert( g_loss < 1000000 ) # make sure it is not NaN or Inf
                
                 # Display logs per epoch step
                if (counter1+1) % display_step == 0:
                    print "Sample:", '%d' % ((idx+1)*self.batch_size), " Epoch:", '%d' % (epoch), \
                        "d_loss=", "{:.4f}".format(d_loss), \
                        "g_loss=", "{:.4f}".format(g_loss), \
                        "n_op=", '%d' % (counter2)
                counter1 += 1
                
                avg_d_loss += d_loss / n_samples * self.batch_size
                avg_g_loss += g_loss / n_samples * self.batch_size
                
            # Display logs per epoch step
            if epoch >= 0:
                print "Epoch:", '%04d' % (epoch), \
                    "avg_d_loss=", "{:.6f}".format(avg_d_loss), \
                    "avg_g_loss=", "{:.6f}".format(avg_g_loss)
            # save model
            if epoch >= 0 and epoch % checkpoint_step == 0:
                checkpoint_path = os.path.join(self.checkpoint_file, 'model.ckpt')
                self.save_model(checkpoint_path, epoch)
                print "model saved to {}".format(checkpoint_path)

        self.save_model(checkpoint_path, 0)     
        
    def test (self):
        dirname = './save' + '/' + self.model_name
        ckpt = tf.train.get_checkpoint_state()
        if ckpt:
            self.load_model(dirname)
            

    
    def generate(self, z=None, x_dim = 32, y_dim = 32, scale = 5.0):
        if z is None:
            z = self.generate_z()
        
        G = self.generator(x_dim = x_dim, y_dim = y_dim, reuse = True)
        gen_x, gen_y, gen_r = self.coordinates(x_dim, y_dim, scale = scale)
        image = self.sess.run(G, feed_dict={self.z: z, self.x: gen_x, self.y: gen_y, self.r: gen_r})
        return image
        
    def generate_z(self):
         z = np.random.uniform(-1.0, 1.0, size = (self.batch_size, self.z_dim)).astype(np.float32)
         return z
         
    def save_model(self, checkpoint_path, epoch):
        """ saves the model to a file """
        self.saver.save(self.sess, checkpoint_path, global_step = epoch)

    def load_model(self, checkpoint_path):
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_path, ckpt_name))
        print "loading model: ",os.path.join(checkpoint_path, ckpt_name)

        #self.saver.restore(self.sess, checkpoint_path+'/'+ckpt.model_checkpoint_path)
        # use the below line for tensorflow 0.7
        #self.saver.restore(self.sess, ckpt.model_checkpoint_path)
    def close(self):
        self.sess.close()
        
class DCGAN():
    def __init__(self, z_dim = 20, img_dim = 32, batch_size = 1, c_dim = 1, g_dim = 32, d_dim = 32,
                 learning_rate_g = 0.001, learning_rate_d = 0.001, beta1 = 0.9, y_dim = None, keep_prob = 1.0, model_name = 'dcgan' ):

        self.z_dim = z_dim
        self.img_dim = img_dim
        self.c_dim = c_dim
        self.y_dim = y_dim
        self.g_dim = g_dim
        self.d_dim = d_dim
        self.batch_size = batch_size
        self.learning_rate_d = learning_rate_d
        self.learning_rate_g = learning_rate_g
        self.keep_prob = keep_prob
        self.model_name = model_name
        self.beta1 = beta1
        self.checkpoint_file = 'save/' + self.model_name 

        
        
        self.batch = tf.placeholder(tf.float32, [None, self.img_dim, self.img_dim, self.c_dim])
        self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')
        self.z = tf.placeholder(tf.float32, [None,self.z_dim])
        
        #self.z = tf.random_normal((self.batch_size, self.z_dim), 0, 1, dtype=tf.float32)
 
        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(batch_size, name=self.model_name+'_d_bn1')
        self.d_bn2 = batch_norm(batch_size, name=self.model_name+'_d_bn2')
        self.d_bn3 = batch_norm(batch_size, name=self.model_name+'_d_bn3')
        
        self.g_bn1 = batch_norm(batch_size, name=self.model_name+'_g_bn1')
        self.g_bn2 = batch_norm(batch_size, name=self.model_name+'_g_bn2')
        self.g_bn3 = batch_norm(batch_size, name=self.model_name+'_g_bn3')
        self.g_bn4 = batch_norm(batch_size, name=self.model_name+'_g_bn4')
        
        self.G = self.generator(y =self.y)
        self.D_real = self.discriminator(self.batch)
        self.D_fake = self.discriminator(self.G, reuse = True)
        
#        self.d_loss = -tf.reduce_mean(tf.log(self.D_real) + tf.log(1. - self.D_fake))
#        self.g_loss = -tf.reduce_mean(tf.log(self.D_fake))
   
        self.d_loss_real = binary_cross_entropy_with_logits(tf.ones_like(self.D_real), self.D_real)
        self.d_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(self.D_fake), self.D_fake)
        self.d_loss = 1.0*(self.d_loss_real + self.d_loss_fake)/ 2.0
        self.g_loss = 1.0*binary_cross_entropy_with_logits(tf.ones_like(self.D_fake), self.D_fake)
        
        self.t_vars = tf.trainable_variables()
        self.g_vars = [var for var in self.t_vars if (self.model_name+'_g_') in var.name]
        self.d_vars = [var for var in self.t_vars if (self.model_name+'_d_') in var.name]
        
        # Use ADAM optimizer
        self.g_opt = tf.train.AdamOptimizer(self.learning_rate_g, beta1=self.beta1) \
                      .minimize(self.g_loss, var_list=self.g_vars)

        self.d_opt = tf.train.AdamOptimizer(self.learning_rate_d, beta1=self.beta1) \
                      .minimize(self.d_loss, var_list=self.d_vars)

        self.init()
        
    def init(self):
        init = tf.initialize_all_variables()
        print ('Initialization is completed!')
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
        self.saver = tf.train.Saver(tf.all_variables())
        
    def reinit(self):
        init = tf.initialize_all_variables(tf.trainable_variables())
        self.sess.run(init)
        
    def generator(self, reuse = False, y=None, n = None):
        if reuse:
            tf.get_variable_scope().reuse_variables()
            
        if n is not None:
            batch_size = n
        else:
            batch_size = self.batch_size
        
        z =  tf.reshape(self.z, [batch_size, self.z_dim])       
        y_ = tf.reshape(y, [batch_size, 1, 1, self.y_dim])
        z = tf.concat(1, [z, y_])  #means axis = 1
        
        z_ = linear(z, 1024 , self.model_name+'_g_0_z')              
        z_ = tf.reshape(z_, [-1, 4, 4, 64]) 
        h0 = tf.nn.relu(self.g_bn1(z_))
        h0 = tf.concat(1, [h0, y_])   
        
        h1 = linear(h0, 128*7*7 , self.model_name+'_g_1_z')  
        h1 = tf.reshape(h1, [-1, 7, 7, 128])
        h2 = tf.nn.relu(self.g_bn2(h1))
        h2 = tf.concat( 3, [h2, y_*tf.ones([batch_size, 7 ,7, self.y_dim])])  #axis = 3
        
        h3 = deconv2d(h2, [batch_size, 14, 14, 64],  name = self.model_name+'_g_h1_deconv')
        h3 = tf.nn.relu(self.g_bn3(self.h3))
        h3 = tf.concat(3, [h3, y_*tf.ones([batch_size, 14 ,14, self.y_dim])])  #axis = 3
        
        h4 = deconv2d(h3, [batch_size, 28, 28, self.c_dim], name = self.model_name+'_g_h2_deconv')
         
        #h3 = tf.nn.relu(self.g_bn4(deconv2d(h2, [self.batch_size, s2, s2, self.g_dim], name = self.model_name+'_g_h3_deconv', with_weights = True)))
        #h4 = deconv2d(h2, [batch_size,  28, 28,  self.c_dim], name = self.model_name+'_g_h4_deconv')

        return tf.nn.tanh(h4)

    
    def discriminator(self, image, y, reuse = False):
        
        if reuse:
            tf.get_variable_scope().reuse_variables()
 
        y_ = tf.reshape(y, tf.pack([self.batch_size, 1, 1, self.y_dim]))
        x = tf.concat(3, [image, y_*tf.ones([self.batch_size, 28, 28, self.y_dim])])
        
        h0 = lrelu(conv2d(x, 64, name = self.model_name+'_d_h0_conv') )
        h0 = tf.reshape(h0, [self.batch_size, -1])
        h0 = tf.concat(1, [h0, y])
        
        h1 = lrelu(self.d_bn1(conv2d(h0, 128,  name = self.model_name+'_d_h1_conv')))
        h1 = tf.reshape(h1, [self.batch_size, -1])
        h1 = tf.concat(1, [h1, y])
        
        h2 = self.d_bn2(linear(tf.reshape(h1, [-1,7*7*128 ]), 1, self.model_name+'_d_h2_lin'))         
        h2 = tf.concat(1, [h2, y])
        out = tf.nn.sigmoid(h2)   
        
        return out
        
    def train(self, num_epoch = 100, display_step = 100, checkpoint_step = 10):
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_file)
        if ckpt:
            self.load_model(self.checkpoint_file)
            print "Model is loaded!"
            
        images, labels = read_dataset()
        n_samples = len(labels)
   
        counter1 = 0
        for epoch in range(num_epoch):

            #batch images 
            batch_idxs = len(images) // self.batch_size
            for idx in xrange(0, batch_idxs):
                batch_images = images[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_labels = labels[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_ys = Onehot(batch_labels, self.y_dim)
                z = self.generate_z()
                for i in range(2):
                    _, g_loss_train = self.sess.run((self.g_opt, self.g_loss), feed_dict = {self.batch: batch_images, self.z:z, self.y : batch_ys })#, self.z :z})                  
                d_loss_train = self.sess.run(self.d_loss, feed_dict = {self.batch: batch_images,self.z:z})
                
                if d_loss_train > 0.45 and g_loss_train < 0.8:              
                    _, d_loss_train = self.sess.run((self.d_opt, self.d_loss),  feed_dict={self.batch: batch_images,self.z:z})    
                #_, d_loss_train = self.sess.run((self.d_opt, self.d_loss), feed_dict={self.batch: batch_images,self.z:z})
                
                assert( d_loss_train < 1000000 ) # make sure it is not NaN or Inf
                assert( g_loss_train < 1000000 ) 
                
                 # Display logs per epoch step
                if (counter1+1) % display_step == 0:
                    print "Epoch", '%d' % (epoch) ,\
                        "Step:" , '%d' %(counter1+1), \
                        "Sample:", '%d' % ((idx+1)*self.batch_size), \
                        "d_loss=", "{:.4f}".format(d_loss_train), \
                        "g_loss=", "{:.4f}".format(g_loss_train)
                if  (counter1+1) % 100 == 0:
                    g_sample, y_sample = self.generate(img_dim = self.img_dim, n = 16)
                    plot(g_sample, name = 'Step'+str(counter1+1))
                    
                counter1 += 1
            # save model
            if epoch >= 0 and epoch % checkpoint_step == 0:
                checkpoint_path = os.path.join(self.checkpoint_file , 'model.ckpt')
                self.save_model(checkpoint_path, epoch)
                print "model saved to {}".format(checkpoint_path)

        self.save_model(checkpoint_path, 0)     
        
    def test (self):
        dirname = './save' + '/' + self.model_name
        ckpt = tf.train.get_checkpoint_state(dirname)
        if ckpt:
            self.load_model(dirname)
            image_data = self.generate(n = 16)
            plot(image_data)
    
    def generate(self, z=None, img_dim = 32, n=None, y = None):
        if z is None:
            z = self.generate_z(n)
        if y is None:
            y = np.random.randint(self.y_dim, size=[n])
        G = self.generator(reuse = True, n=n)
        image = self.sess.run(G,  feed_dict={self.z:z, self.y : Onehot(y,n)})
        return image, y
        
    def generate_z(self, n=None):
        if n is None:
            z = np.random.uniform(-1.0, 1.0, size = (self.batch_size, self.z_dim)).astype(np.float32)
        else:
            z = np.random.uniform(-1.0, 1.0, size = (n, self.z_dim)).astype(np.float32)
        return z
         
    def save_model(self, checkpoint_path, epoch):
        """ saves the model to a file """
        self.saver.save(self.sess, checkpoint_path, global_step = epoch)

    def load_model(self, checkpoint_path):
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_path, ckpt_name))
        print "loading model: ",os.path.join(checkpoint_path, ckpt_name)

        #self.saver.restore(self.sess, checkpoint_path+'/'+ckpt.model_checkpoint_path)
        # use the below line for tensorflow 0.7
        #self.saver.restore(self.sess, ckpt.model_checkpoint_path)
    def close(self):
        self.sess.close()
    
def show_image(image_data):
    plt.subplot(1,1,1)
    y_dim = image_data.shape[0]
    x_dim = image_data.shape[1]
    c_dim = image_data.shape[2]
    if c_dim > 1:
        plt.imshow(image_data, interpolation = 'nearest')
    else:
        plt.imshow(image_data.reshape(y_dim, x_dim), cmap='Greys', interpolation='nearest')
    plt.axis('off')
    plt.show()
    
def save_image(image_data, filename):
    img_data = np.array(1-image_data)
    y_dim = image_data.shape[0]
    x_dim = image_data.shape[1]
    c_dim = image_data.shape[2]
    if c_dim > 1:
      img_data = np.array(img_data.reshape((y_dim, x_dim, c_dim))*255.0, dtype=np.uint8)
    else:
      img_data = np.array(img_data.reshape((y_dim, x_dim))*255.0, dtype=np.uint8)
    im = Image.fromarray(img_data)
    im.save(filename)
              

def read_dataset(data_dir = './data/MNIST/', one_hot = False):
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)
        
    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0)
       
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    
    if one_hot:    
        y_vec = np.zeros((len(y), None), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i,y[i]] = 1.0
            
        return X/255.,y_vec
    else:
        return X/255, y

def plot(samples, save = False, name = None):
    x_dim = samples.shape[1]
    y_dim = samples.shape[2]
    nsample = samples.shape[0]
    fig_size = int(np.sqrt(nsample))
    fig = plt.figure(figsize=(fig_size, fig_size))
    gs = gridspec.GridSpec(fig_size, fig_size)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(x_dim, y_dim), cmap = 'gray')
    plt.show()
    if save:
        samples_file = 'samples/' + 'mnist/' + self.model_name 
        fig.savefig(os.path.join(samples_file, name, '.png'))
        
def Onehot(X, n=None, negative_class=0.):
    X = np.asarray(X).flatten()
    if n is None:
        n = np.max(X) + 1
    Xoh = np.ones((len(X), n)) * negative_class
    Xoh[np.arange(len(X)), X] = 1.
    return Xoh

        
#cppngan = CPPNGAN(x_dim = 28, y_dim = 28, batch_size=100, z_dim = 100, scale = 1.0)
#cppngan.train(num_epoch = 50)
#cppngan.test()

dcgan = DCGAN(img_dim = 28, g_dim = 28, d_dim = 28, batch_size=256, z_dim = 100, y_dim = 10)
dcgan.train(num_epoch = 50)
#dcgan.test()

#sample = Sampler()  
#image_data = sample.cppn.generate()[0]
#sample.show_image(image_data)                      

image_data = dcgan.generate(n = 1, y = 8)[0]
show_image(image_data)   


z_dim = 100
z = np.random.uniform(-1.0, 1.0, size = (1, z_dim)).astype(np.float32)                     
#image_data2 = sample.cppngan.generate(z = z)[0]
image_data2 = dcgan.generate(z = z, n = 1)[0]
show_image(image_data2)                    
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        