################################################################################
#Michael Guerzhoy and Davi Frossard, 2016
#AlexNet implementation in TensorFlow, with weights
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

from numpy import *
import os
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import cPickle
import scipy.io as sio


import tensorflow as tf

from datagenerator import ImageDataGenerator

# Load the bvlc alexnet model.
net_data = load("bvlc_alexnet.npy").item()

# You should fine-tune the base model first, and then you can load base model's
# snapshot for the initialization of fc-7 and fc-8 layers. Then, you can train your
# noise model by updating these weights.
snapshot = cPickle.load(open("base_snapshot.pkl"))
fc7W = tf.Variable(snapshot["fc7W"])
fc8W = tf.Variable(snapshot["fc8W"])
fc7b = tf.Variable(snapshot["fc7b"])
fc8b = tf.Variable(snapshot["fc8b"])

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3)
        kernel_groups = tf.split(kernel, group, 3)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])

def weight_decay_penalty(weights, penalty):
    return penalty * sum([tf.nn.l2_loss(w) for w in weights])

# Learning params
learning_rate = 0.1
num_epochs = 250
batch_size = 128
weight_penalty = 0.1

# Network params
dropout_rate = 0.5
num_classes = 10

# Define the placeholders.
train_x = zeros((1, 227,227,3)).astype(float32)
xdim = train_x.shape[1:]

x = tf.placeholder(tf.float32, (None,) + xdim)
y = tf.placeholder(tf.float32, [None, num_classes])

#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1W = tf.Variable(net_data["conv1"][0])
conv1b = tf.Variable(net_data["conv1"][1])
conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2W = tf.Variable(net_data["conv2"][0])
conv2b = tf.Variable(net_data["conv2"][1])
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)


#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3W = tf.Variable(net_data["conv3"][0])
conv3b = tf.Variable(net_data["conv3"][1])
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4W = tf.Variable(net_data["conv4"][0])
conv4b = tf.Variable(net_data["conv4"][1])
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)


#conv5
#conv(3, 3, 256, 1, 1, group=2, name='conv5')
k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
conv5W = tf.Variable(net_data["conv5"][0])
conv5b = tf.Variable(net_data["conv5"][1])
conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv5 = tf.nn.relu(conv5_in)

#maxpool5
#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#fc6
#fc(4096, name='fc6')
fc6W = tf.Variable(net_data["fc6"][0])
fc6b = tf.Variable(net_data["fc6"][1])
fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
# fc6 = tf.nn.dropout(fc6, dropout_rate)

#fc7
#fc(4096, name='fc7')
# Use this one while fine-tuning the base model
# fc7W = tf.Variable(tf.truncated_normal([4096, 4096], stddev=0.1))
# fc7b = tf.Variable(tf.constant(1.0, shape=[4096]))

# Use this one while fine-tuning the noise model
fc7W = tf.Variable(snapshot["fc7W"])
fc7b = tf.Variable(snapshot["fc7b"])
fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)
# fc7 = tf.nn.dropout(fc7, dropout_rate)

#fc8
#fc(1000, relu=False, name='fc8')
# Use this one while fine-tuning the base model
# fc8W = tf.Variable(tf.truncated_normal([4096, num_classes], stddev=0.1))
# fc8b = tf.Variable(tf.constant(1.0, shape=[num_classes]))

# Use this one while fine-tuning the noise model
fc8W = tf.Variable(snapshot["fc8W"])
fc8b = tf.Variable(snapshot["fc8b"])
fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)


#prob
#softmax(name='prob'))
prob = tf.nn.softmax(fc8)

# ************** Noise layer start ****************
# prob = tf.nn.dropout(prob, 0.5)
W_noise = tf.Variable(tf.convert_to_tensor(np.eye(num_classes), dtype=tf.float32))
linear_layer = tf.matmul(prob, W_noise)
# ************** Noise layer end ****************

# Compute the loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=linear_layer, labels=y))

# Add weight decay penalty
# loss = loss + weight_decay_penalty([W_noise], weight_penalty)

# Use this one while fine-tuning the base model
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, var_list=[fc8W, fc8b, fc7W, fc7b])

# Use this one while fine-tuning the noise model
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, var_list=[W_noise, fc8W, fc8b, fc7W, fc7b])

# Calculate the accuracy
correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Write summaries to writer to visualize the loss graph on TensorBoard
training_summary = tf.summary.scalar("training_loss", loss)
validation_summary = tf.summary.scalar("validation_loss", loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

saver = tf.train.Saver()
writer = tf.summary.FileWriter('log', sess.graph)

# Initialize the data generator for the training and validation sets
train_generator = ImageDataGenerator('train.txt',
                                     horizontal_flip = False, shuffle = True)
val_generator = ImageDataGenerator('val.txt', shuffle = False)

# Get the number of training steps per epoch
train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)

for i in range(num_epochs):

    # Fine-tune the network, get training loss as summary
    batch_xs, batch_ys = train_generator.next_batch(batch_size)
    _, train_summary, loss_train = sess.run([optimizer, training_summary, loss], feed_dict={x: batch_xs, y: batch_ys})
    writer.add_summary(train_summary, i)

    # Compute training accuracy
    train_acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})

    # Get validation loss and accuracy
    batch_tx, batch_ty = val_generator.next_batch(batch_size)
    val_acc, val_summary, loss_val = sess.run([accuracy, validation_summary, loss], feed_dict={x: batch_tx, y: batch_ty})
    writer.add_summary(val_summary, i)

    print "i=", i
    print "Validation:", val_acc
    print "Train:", train_acc

    # Save snapshots for each step
    snapshot = {}
    snapshot["fc7W"] = sess.run(fc7W)
    snapshot["fc8W"] = sess.run(fc8W)
    snapshot["fc7b"] = sess.run(fc7b)
    snapshot["fc8b"] = sess.run(fc8b)
    cPickle.dump(snapshot, open("new_snapshot" + str(i) + ".pkl", "w"))

    # Reset the file pointer of the image data generator
    if i % train_batches_per_epoch == 0:
        train_generator.reset_pointer()

writer.close()
train_file.close()
val_file.close()

################################################################################
# Testing

# val_generator = ImageDataGenerator('test.txt', shuffle = False)
#
# test_acc = 0.
# test_count = 0
#
# batch_tx, batch_ty = val_generator.next_batch(200) # Since there are 200 test images in total.
# acc, val_summary, loss_val = sess.run([accuracy, validation_summary, loss], feed_dict={x: batch_tx, y: batch_ty})
#
# print ("Test accuracy: " + str(acc))
