# -*- coding:utf-8 -*-

from ..core import *
from ..ops import *


def fc(input_data, input_shape, output_shape, activation):
    """
    full connection
    :param input_data: input array
    :param input_shape: array shape, n*1
    :param output_shape: mean output dim m*1
    :param activation:
    :return:
    """
    weights = Variable((output_shape, input_shape), init=True, trainable=True)  # m*n
    bias = Variable((output_shape, 1), init=True, trainable=True)
    affine = Add(MatMul(weights, input_data), bias)  # 线性变换，泛函分析中称之为仿射变换（听着高大上）

    if activation == "ReLU":
        return ReLU(affine)
    elif activation == "Logistic":
        return Logistic(affine)
    else:
        return affine


def pooling(feature_maps, kernel_shape, stride):
    """
    pooling layer
    :param feature_maps: array, include many feature_maps which has same shape
    :param kernel_shape: tuple type
    :param stride: tuple, include H direction and V direction
    :return:
    """
    outputs = []
    for feat_map in feature_maps:
        outputs.append(MaxPooling(feat_map, size=kernel_shape, stride=stride))
    return outputs


def conv(feature_maps, input_shape, kernels_n, kernel_shape, activation):
    """
    :param feature_maps: arrays, has same shape
    :param input_shape: tuple
    :param kernels_n: nums of conv kernels
    :param kernel_shape:
    :param activation:
    :return:
    """
    # construct a matrix has same  shape with input
    ones = Variable(input_shape, init=False, trainable=False)
    ones.set_value(np.mat(np.ones(input_shape)))

    outputs = []
    # for every kernel in kernels
    for i in range(kernels_n):
        channels = []
        for feat_map in feature_maps:
            kernel = Variable(kernel_shape, init=True, trainable=True)
            convresult = Conv(feat_map, kernel)
            channels.append(convresult)

        channels = Add(*channels)
        bias = ScalarMultiply(Variable((1, 1), init=True, trainable=True), ones)
        affine = Add(channels, bias)

        if activation == "ReLU":
            outputs.append(ReLU(affine))
        elif activation == "Logistic":
            outputs.append(Logistic(affine))
        else:
            outputs = affine
        # result layers equals to kernels num
        assert len(outputs) == kernels_n
        return outputs