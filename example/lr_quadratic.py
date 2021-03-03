import numpy as np
import tinyframework as tf
from sklearn.datasets import make_circles

# Constructing concentric circle data
X, y = make_circles(200, noise=0.1, factor=0.2)
y = y*2-1  # *2-1
use_quadratic = True

x1 = tf.core.Variable(dim=(2, 1), init=False, trainable=False)
label = tf.core.Variable(dim=(1, 1), init=False, trainable=False)
b = tf.core.Variable(dim=(1, 1), init=True, trainable=True)

if use_quadratic:
    # trans self and matmul then reshape
    x2 = tf.ops.Reshape(tf.ops.MatMul(x1, tf.ops.Reshape(x1, shape=(1, 2))), shape=(4, 1))
    # concat 1 order and 2 order
    x = tf.ops.Concat(x1, x2)  # this ops first flatten 2 params by line, then concat
    w = tf.core.Variable(dim=(1, 6), init=True, trainable=True)
else:
    x = x1
    w = tf.core.Variable(dim=(1, 2), init=True, trainable=True)

out = tf.ops.Add(tf.ops.MatMul(w, x), b)
predict = tf.ops.Logistic(out)

loss = tf.ops.loss.LogLoss(tf.ops.MatMul(label, out))

learning_rate = 0.01

optimizer = tf.optimizer.Adam(tf.default_graph, loss, learning_rate)

batch_size = 50

for epoch in range(200):
    batch_no = 0
    for i in range(len(X)):
        x1.set_value(np.mat(X[i]).T)
        label.set_value(np.mat(y[i]))

        optimizer.one_step()
        batch_no += 1

        if batch_no >= batch_size:
            optimizer.update()
            batch_no = 0

    pred = []
    for i in range(len(X)):
        x1.set_value(np.mat(X[i]).T)
        label.set_value(np.mat(y[i]))

        predict.forward()

        pred.append(predict.value[0, 0])

    pred = (np.array(pred) > 0.5).astype(np.int32)*2-1

    accuracy = (y == pred).astype(np.int16).sum() / len(X)

    print("epoch:{:d}, accuracy:{:3f}".format(epoch, accuracy))


