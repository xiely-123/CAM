# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 13:44:26 2020

@author: ALW
"""

import tensorflow as tf


def BN(x):
    return tf.layers.batch_normalization(x,axis=-1,momentum=0.99,epsilon=0.001, center=True, scale=True,)
 
def Resblock(x,kernel_size,stride,channel):
    layer1=tf.nn.relu(BN(tf.layers.conv2d(x,channel,kernel_size,strides=stride, padding='same')))
    layer1=tf.nn.relu(BN(tf.layers.conv2d(layer1,channel,kernel_size,strides=1, padding='same')))
    skip=tf.layers.conv2d(x,channel,1,strides=stride, padding='same')
    res_value=tf.nn.relu(BN(skip+layer1))
    return res_value

def Dual_attention(x,H,W,C):
    ######position
    x=tf.layers.conv2d(x,C*3,1,strides=1, padding='same')
    Q,K,V_poition=tf.split(x, 3, axis=3)
    Q=tf.reshape(Q,[-1,H*W,C])
    K=tf.reshape(K,[-1,H*W,C])
    V_poition=tf.reshape(V_poition,[-1,H*W,C])
    K=tf.transpose(K,[0,2,1])
    result=tf.matmul(Q,K)
    result=tf.nn.softmax(result)
    V_poition=tf.matmul(result,V_poition)
    V_poition=tf.reshape(V_poition,[-1,H,W,C])
    ######channel
    x=tf.layers.conv2d(x,C*3,1,strides=1, padding='same')
    Q,K,V_channel=tf.split(x, 3, axis=3)
    Q=tf.reshape(Q,[-1,H*W,C])
    K=tf.reshape(K,[-1,H*W,C])
    V_channel=tf.reshape(V_channel,[-1,H*W,C])
    K=tf.transpose(K,[0,2,1])
    result=tf.matmul(K,Q)
    result=tf.nn.softmax(result)
    V_channel=tf.matmul(V_channel,result)
    V_channel=tf.reshape(V_channel,[-1,H,W,C])
    V=V_channel+V_poition
    return V
 

def SK_block(x,kernel1,kernel2,channel):
    ############Spilt
    U1=tf.layers.conv2d(x,channel,kernel1,strides=1, padding='same')
    U2=tf.layers.conv2d(x,channel,kernel2,strides=1, padding='same')
    ############Fuse    
    U=U1+U2
    S=tf.keras.layers.GlobalAvgPool2D()(U)
    print(S)
    S=tf.reshape(S,[-1,1,1,channel])
    print(S)
    Z=tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(S,32,1,strides=1, padding='same'),axis=-1,momentum=0.99,epsilon=0.001, center=True, scale=True,))
    print(Z)
    a=tf.layers.conv2d(Z,channel,1,strides=1, padding='same')
    b=tf.layers.conv2d(Z,channel,1,strides=1, padding='same')
    print(a,b)
    combine=tf.concat([a,b],1)
    print(combine)
    combine=tf.nn.softmax(combine,axis=1)
    print(combine)
    a,b=tf.split(combine,num_or_size_splits=2, axis=1)
    print(a,b)
    V=a*U1+b*U2
    print(V)
    return V   
    
    