# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 16:47:35 2017

@author: Ean2
"""

import os 
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载mnist_inference.py 中定义的常量和前向传播的函数
import mnist_inference

# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30001
MOVING_AVERAGE_DECAY = 0.99

# 模型保存的路径和文件名
MODEL_SAVE_PATH = r"C:\Users\Ean2\GitHub\LearnNote\TensorFlow\Model"
MODEL_NAME = "model.ckpt"

def train(mnist):
    # 定义输入输出 placeholder
    x = tf.placeholder(tf.float32, 
                       [BATCH_SIZE,
                        mnist_inference.IMAGE_SIZE,
                        mnist_inference.IMAGE_SIZE,
                        mnist_inference.NUM_CHANNELS],
                       name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE],
                        name='y-input')
    
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    # 直接使用mnist_inference.py 中定义的前向传播过程
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)
    
    # 和5.2.1小节样例中类似地定义损失函数、学习率、晃动平均操作及训练过程
    variable_averages = tf.train.ExponentialMovingAverage(
                        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.argmax(y_,1), logits=y)    # 特别注意 labels=tf.argmax(y_,1)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
                    LEARNING_RATE_BASE,
                    global_step,
                    # decay_steps 完整使用一遍训练数据所需要迭代的轮数
                    # 总训练样本数除以每一个batch中的训练样本数
                    mnist.train.num_examples / BATCH_SIZE, 
                    LEARNING_RATE_DECAY,
                    staircase = False)
    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
                 .minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train') 
    # 初始化 TensorFlow 持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        
        # 在训练过程中不在测试模型在验证数据上的表现，验证和测试的过程将会有
        # 一个独立的程序完成
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                  mnist_inference.IMAGE_SIZE,
                                  mnist_inference.IMAGE_SIZE,
                                  mnist_inference.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: reshaped_xs, y_:ys})
            # 每1000轮保存一次模型
            if i % 1000 == 0:
                print("After %d train step(s), loss on training "
                      "batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                           global_step=global_step)

def main(argv=None):
    mnist_file = r'C:\Users\Ean2\GitHub\LearnNote\TensorFlow\OriginData'
    mnist = input_data.read_data_sets(mnist_file, one_hot=True)
    train(mnist)

if __name__== '__main__':
    tf.app.run()