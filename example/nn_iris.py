# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import tinyframework as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tinyframework.utils import draw
from tinyframework.trainer import saver
from tinyframework import get_node_from_graph


data = pd.read_csv('../data/Iris.csv').drop('Id', axis=1)
# shuffle
data = data.sample(len(data), replace=False)

le = LabelEncoder()
num_label = le.fit_transform(data['Species'])

oh = OneHotEncoder(sparse=False)
one_hot_label = oh.fit_transform(num_label.reshape(-1, 1))

# build feature
features = data[['SepalLengthCm',
                 'SepalWidthCm',
                 'PetalLengthCm',
                 'PetalWidthCm']].values

x = tf.core.Variable(dim=(4, 1), init=False, trainable=False)
one_hot = tf.core.Variable(dim=(3, 1), init=False, trainable=False)

hidden1 = tf.layer.fc(x, 4, 10, "ReLU")
hidden2 = tf.layer.fc(hidden1, 10, 10, "ReLU")
out = tf.layer.fc(hidden2, 10, 3, None)

predict = tf.ops.SoftMax(out)

loss = tf.ops.loss.CrossEntropy(out, one_hot)
learning_rate = 0.005

optimizer = tf.optimizer.Adam(tf.default_graph, loss, learning_rate)
batch_size = 32

for epoch in range(200):
    batch_cnt = 0
    for i in range(len(features)):
        feat = np.mat(features[i, :]).T
        label = np.mat(one_hot_label[i, :]).T

        x.set_value(feat)
        one_hot.set_value(label)

        optimizer.one_step()
        batch_cnt += 1

        if batch_cnt >= batch_size:
            optimizer.update()
            batch_cnt = 0

    pred = []
    for i in range(len(features)):
        feat = np.mat(features[i, :]).T
        x.set_value(feat)
        predict.forward()
        pred.append(predict.value.A.ravel())

    pred = np.array(pred).argmax(axis=1)

    accuracy = (num_label == pred).astype(np.int32).sum() / len(features)

    print("epoch:{:d}, accuracy:{:.3f}".format(epoch + 1, accuracy))


draw.draw_graph(filename='nn_iris')
saver = saver.Saver('./mode_save')
saver.save(model_file_name='nn_iris.json', weights_file_name='nn_iris.npz')

