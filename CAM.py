# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 14:39:51 2020

@author: ALW
"""

import tensorflow as tf
import cifar10, cifar10_input
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

data_dir = './cifar10_data/cifar-10-batches-bin'
images_test, labels_test = cifar10_input.inputs(eval_data=True,
                                                data_dir=data_dir,
                                                batch_size=1)
print(images_test,labels_test)


saver = tf.train.import_meta_graph( './Mobel_service/best.ckpt.meta')# 加载图结构
gragh = tf.get_default_graph()# 获取当前图，为了后续训练时恢复变量
tensor_name_list = [tensor.name for tensor in gragh.as_graph_def().node]# 得到当前图中所有变量的名称
#print(tensor_name_list)
x = gragh.get_tensor_by_name('Placeholder:0')
y = gragh.get_tensor_by_name('Placeholder_1:0')
feature = gragh.get_tensor_by_name('Relu_20:0')
weights = gragh.get_tensor_by_name('global_average_pooling2d_1/Mean:0')
#test_result =  gragh.get_tensor_by_name('Add_7:0')
CAM = tf.reduce_sum(feature*weights, -1)
print(CAM)



sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()

image_batch, label_batch = sess.run([images_test, labels_test])
CAM_result=sess.run([CAM], feed_dict={x: image_batch, y: label_batch})
CAM_result = np.reshape(CAM_result, [24,24])
#test_result = sess.run([test_result], feed_dict={x: image_batch, y: label_batch})



print(CAM_result.shape)
###显示
image_batch = np.reshape(image_batch, [24,24,3])
plt.figure()
plt.subplot(1,2,1)
plt.imshow(image_batch, interpolation = 'bilinear')
plt.subplot(1,2,2)
plt.imshow(CAM_result, interpolation = 'bilinear')
plt.show()
#####
print(image_batch.shape)
print(label_batch)
#print(test_result)