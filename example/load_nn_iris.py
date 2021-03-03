# -*- coding:utf-8 -*-

from tinyframework.trainer import saver
import tinyframework as tf
import numpy as np

saver = saver.Saver('./mode_save')
saver.load(model_file_name='nn_iris.json', weights_file_name='nn_iris.npz')

x = tf.get_node_from_graph('Variable:0')
pred = tf.get_node_from_graph('SoftMax:16')

# 6.2	3.4	5.4	2.3
x.set_value(np.mat([7, 3.2, 4.7, 1.4]).T)
pred.forward()
print(pred.value)
print("predict class: {}".format(np.argmax(pred.value)+1))