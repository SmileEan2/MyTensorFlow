# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 15:13:44 2017

@author: Ean2
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(r'C:\Users\Ean2\GitHub\LearnNote\TensorFlow\OriginData',
                                  one_hot=True)
sess = tf.InteractiveSession()

# Variables
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Predicted Class and Loss Function
sess.run(tf.global_variables_initializer())
y = tf.matmul(x, W) + b
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Train the Model
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for _ in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    
# Evaluate the Model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images,
                               y_: mnist.test.labels}))