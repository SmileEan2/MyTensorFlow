# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 19:09:21 2017

@author: Ean2
"""

import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
 
import mnist_inference
import mnist_train
 
# 每10秒加载一次最新的模型，并在测试数据上测试最新的模型正确率
EVAL_INTERVAL_SECS = 10
 
def evaluate(mnist):
    with tf.Graph().as_default():
        # 定义输入输出的格式
        x = tf.placeholder(tf.float32, 
                           [None,
                           mnist_inference.IMAGE_SIZE,
                           mnist_inference.IMAGE_SIZE,
                           mnist_inference.NUM_CHANNELS],
                           name='x-input')
        y_ = tf.placeholder(tf.float32, [None,mnist_inference.OUTPUT_NODE],
                            name='y-input')
        reshaped_xs = np.reshape(mnist.validation.images,
                                 (np.shape(mnist.validation.images)[0],
                                  mnist_inference.IMAGE_SIZE,
                                  mnist_inference.IMAGE_SIZE,
                                  mnist_inference.NUM_CHANNELS))
        validate_feed = {x: reshaped_xs,
                         y_: mnist.validation.labels}
        # 直接通过调用封装好的函数来计算前向传播结果。因为测试的时候不关注正则化损失的值
        # 函数被设置成为None
        y = mnist_inference.inference(x, None,None )
        
        # 使用前向传播的结果计算正确率。
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        # 通过变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用求
        # 滑动平均的函数来获取平均值。
        variable_averages = tf.train.ExponentialMovingAverage(
                mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        # 每隔EVAL_INTERVAL_SECS秒调用一次计算正确率的过程以检测训练过程中
        # 正确率的变化
        with tf.Session() as sess:
            # tf.train.get_checkpoint_state 函数会通过checkpoint文件
            # 自动找到目录中的最新模型的文件名
            print(sess.run(variables_to_restore))
            ckpt = tf.train.get_checkpoint_state(
                    mnist_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                # 加载模型
                saver.restore(sess,ckpt.model_checkpoint_path)
                # 通过文件名得到模型保存时迭代的轮数
                global_step = os.path.split(ckpt.model_checkpoint_path)[-1]\
                .split('-')[-1]
                accuracy_score = sess.run(accuracy,feed_dict=validate_feed)
                print('After %s training step(s), validation '
                      "accuracy = %g" % (global_step, accuracy_score))
            else:
                print('No checkpoint file found')
                return 
            #time.sleep(EVAL_INTERVAL_SECS)
               
def main(argv=None):
    mnist = input_data.read_data_sets(
            r'C:\Users\Ean2\GitHub\MyTensorFlow\OriginData', one_hot=True)
    evaluate(mnist)
    
if __name__ == '__main__':
    tf.app.run()
    quit()