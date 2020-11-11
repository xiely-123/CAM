# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 15:44:09 2020

@author: ALW
"""

import cifar10, cifar10_input
import tensorflow as tf
import numpy as np
import time
import math
import model 
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
max_steps = 300000
batch_size = 256
data_dir = './cifar10_data/cifar-10-batches-bin'


def variable_with_weight_loss(shape, stddev, w1):
    '''定义初始化weight函数,使用tf.truncated_normal截断的正态分布，但加上L2的loss，相当于做了一个L2的正则化处理'''
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    '''w1:控制L2 loss的大小，tf.nn.l2_loss函数计算weight的L2 loss'''
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
        '''tf.add_to_collection:把weight losses统一存到一个collection，名为losses'''
        tf.add_to_collection('losses', weight_loss)

    return var


# 使用cifar10类下载数据集并解压展开到默认位置
cifar10.maybe_download_and_extract()

'''distored_inputs函数产生训练需要使用的数据，包括特征和其对应的label,
返回已经封装好的tensor，每次执行都会生成一个batch_size的数量的样本'''
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir,batch_size=batch_size)

images_test, labels_test = cifar10_input.inputs(eval_data=True,
                                                data_dir=data_dir,
                                                batch_size=200)





image_holder = tf.placeholder(tf.float32, [None, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [None])


layer1=tf.layers.conv2d(image_holder,64,5,strides=1, padding='same')
layer1=tf.nn.relu(layer1)

layer1=model.Resblock(layer1,3,1,256)
layer1=model.SK_block(layer1,3,5,128)
layer1=model.Resblock(layer1,5,1,256)
#layer1=model.Dual_attention(layer1,24,24,128)
#layer1=tf.nn.max_pool(layer1, 5, 2, padding="SAME")
layer1=model.Resblock(layer1,3,1,128)
layer1=model.Resblock(layer1,3,1,128)
#layer1=tf.nn.max_pool(layer1, 5, 2, padding="SAME")
layer1=model.Resblock(layer1,3,1,64)
layer1=model.Resblock(layer1,3,1,32)

###########分类

layer1_transpose=tf.layers.conv2d_transpose(layer1,10,3,strides=1, padding='same')
layer1_transpose=tf.nn.relu(layer1_transpose)
print(layer1_transpose)

GAP=tf.keras.layers.GlobalAvgPool2D()(layer1_transpose)
print(GAP)

# 全连接层，隐含层节点数下降了一半


'''正态分布标准差设为上一个隐含层节点数的倒数，且不计入L2的正则'''
weight5 = variable_with_weight_loss(shape=[10, 10], stddev=1 / 10, w1=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(GAP, weight5), bias5)
print(logits)

def loss(logits, labels):
    '''计算CNN的loss
    tf.nn.sparse_softmax_cross_entropy_with_logits作用：
    把softmax计算和cross_entropy_loss计算合在一起'''
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    # tf.reduce_mean对cross entropy计算均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                        name='cross_entropy')
    # tf.add_to_collection:把cross entropy的loss添加到整体losses的collection中
    tf.add_to_collection('losses', cross_entropy_mean)
    # tf.add_n将整体losses的collection中的全部loss求和得到最终的loss
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


# 将logits节点和label_holder传入loss计算得到最终loss
loss = loss(logits, label_holder)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
# 求输出结果中top k的准确率，默认使用top 1(输出分类最高的那一类的准确率)
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()
all_time=0
saver = tf.train.Saver()
basic_acc=0
for step in range(max_steps):
    '''training:'''
    start_time = time.time()
    # 获得一个batch的训练数据
    image_batch, label_batch = sess.run([images_train, labels_train])
    # 将batch的数据传入train_op和loss的计算
    _, loss_value = sess.run([train_op, loss],
                             feed_dict={image_holder: image_batch, label_holder: label_batch})
    train_time = time.time() - start_time
    all_time = all_time + train_time
    #print(step,all_time)
    #duration = time.time() - start_time
    if step % 10 == 0:
        start_time_test = time.time()
        #print(all_time)
        # 每秒能训练的数量
        #examples_per_sec = batch_size / duration
        # 一个batch数据所花费的时间
        #sec_per_batch = float(duration)


        # 获取images-test labels_test的batch
        image_batch, label_batch = sess.run([images_test, labels_test])
        # 计算这个batch的top 1上预测正确的样本数
        preditcions = sess.run([top_k_op], feed_dict={image_holder: image_batch,
                                                     label_holder: label_batch
                                                     })
        #precision=np.sum(preditcions)/1000
        #print('precision = %.3f' % precision)
# 样本数
        num_examples = 10000
        num_iter = int(math.ceil(num_examples / 200))
        true_count = 0
        total_sample_count = num_iter * 200
        step_test = 0
        while step_test < num_iter:     
            # 获取images-test labels_test的batch
            image_batch, label_batch = sess.run([images_test, labels_test])
            # 计算这个batch的top 1上预测正确的样本数
            preditcions = sess.run([top_k_op], feed_dict={image_holder: image_batch,
                                                          label_holder: label_batch
                                                         })
            # 全部测试样本中预测正确的数量
            true_count += np.sum(preditcions)
            step_test += 1
       # 准确率
        precision = true_count / total_sample_count
        
        if basic_acc <= precision:
           saver.save(sess, './Mobel/best.ckpt')
           basic_acc = precision
        
        test_time = time.time() - start_time_test
        
        all_time = all_time + test_time
        format_str = ('step %d, loss=%.2f (train_batch_time = %.2f sec, test_time = %.2f sec, all_time = %.2f sec, precision @ 1 = %.3f)')
        print(format_str % (step, loss_value, train_time, test_time, all_time, precision)) 
        all_time = 0