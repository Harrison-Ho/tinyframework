import numpy as np
import pandas as pd
import tinyframework as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# read data
data = pd.read_csv("../data/titanic.csv").drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

le = LabelEncoder()
ohe = OneHotEncoder(sparse=False)

Pclass = ohe.fit_transform(le.fit_transform(data["Pclass"].fillna(0)).reshape(-1, 1))  # 1列
Sex = ohe.fit_transform(le.fit_transform(data["Sex"].fillna("")).reshape(-1, 1))
Embarked = ohe.fit_transform(le.fit_transform(data["Embarked"].fillna("")).reshape(-1, 1))

# concat columns
features = np.concatenate([Pclass,
                           Sex,
                           data[["Age"]].fillna(0),
                           data[["SibSp"]].fillna(0),
                           data[["Parch"]].fillna(0),
                           data[["Fare"]].fillna(0),
                           Embarked], axis=1)

labels = data["Survived"].values*2-1
feat_dim = features.shape[1]

emb_size = 2
# 1 order
x = tf.core.Variable(dim=(feat_dim, 1), init=False, trainable=False)
w = tf.core.Variable(dim=(1, feat_dim), init=True, trainable=True)

x_Pclass = tf.core.Variable(dim=(Pclass.shape[1], 1), init=False, trainable=False)
x_Sex = tf.core.Variable(dim=(Sex.shape[1], 1), init=False, trainable=False)
x_Embarked = tf.core.Variable(dim=(Embarked.shape[1], 1), init=False, trainable=False)

emb_weight_Pclass = tf.core.Variable(dim=(emb_size, Pclass.shape[1]), init=True, trainable=True)
emb_weight_Sex = tf.core.Variable(dim=(emb_size, Sex.shape[1]), init=True, trainable=True)
emb_weight_Embark = tf.core.Variable(dim=(emb_size, Embarked.shape[1]), init=True, trainable=True)

emb_Pclass = tf.ops.MatMul(emb_weight_Pclass, x_Pclass)
emb_Sex = tf.ops.MatMul(emb_weight_Sex, x_Sex)
emb_Embarked = tf.ops.MatMul(emb_weight_Embark, x_Embarked)

bias = tf.core.Variable(dim=(1, 1), init=True, trainable=True)

emb_feat = tf.ops.Concat(emb_Pclass, emb_Sex, emb_Embarked)

# FM part
fm = tf.ops.Add(tf.ops.MatMul(w, x),   # 一次部分
                # 二次部分
                tf.ops.MatMul(tf.ops.Reshape(emb_feat, shape=(1, 3 * emb_size)), emb_feat)
                )

# deep part
hidden_1 = tf.layer.fc(emb_feat, 3*emb_size, 8, "Relu")
hidden_2 = tf.layer.fc(hidden_1, 8, 4, "Relu")
deep = tf.layer.fc(hidden_2, 4, 1, None)

output = tf.ops.Add(fm, deep, bias)
predict = tf.ops.Logistic(output)
label = tf.core.Variable(dim=(1, 1), init=False, trainable=False)

loss = tf.ops.loss.LogLoss(tf.ops.Multiply(label, output))
learning_rate = 0.05
optimizer = tf.optimizer.Adam(tf.default_graph, loss, learning_rate)

batch_size = 64
for epoch in range(200):
    batch_cnt = 0
    for i in range(len(features)):
        x.set_value(np.mat(features[i]).T)
        x_Pclass.set_value(np.mat(features[i, :3]).T)
        x_Sex.set_value(np.mat(features[i, 3:5]).T)
        x_Embarked.set_value(np.mat(features[i, 9:]).T)
        label.set_value(np.mat(labels[i]))

        optimizer.one_step()
        batch_cnt += 1

        if batch_cnt > batch_size:
            optimizer.update()
            batch_cnt = 0

    pred = []
    for i in range(len(features)):
        x.set_value(np.mat(features[i]).T)
        x_Pclass.set_value(np.mat(features[i, :3]).T)
        x_Sex.set_value(np.mat(features[i, 3:5]).T)
        x_Embarked.set_value(np.mat(features[i, 9:]).T)

        predict.forward()
        pred.append(predict.value[0, 0])

    pred = (np.array(pred) > 0.5).astype(np.int16)*2-1
    accuracy = (labels == pred).astype(np.int16).sum() / len(features)

    print("epoch:{:d}, acc:{:.3f}".format(epoch+1, accuracy))
