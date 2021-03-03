# -*- coding: utf-8 -*-

from ..core import Node
import numpy as np


class Operator(Node):
    """
    Node abstract operator
    """
    pass


class Add(Operator):
    """
    matrix add
    """
    def calculate(self):
        self.value = np.mat(np.zeros(self.parents[0].shape()))
        for parent in self.parents:
            self.value += parent.value

    def get_jacobi(self, parent):
        # ＋ 节点对任意父节点的jacobi矩阵是单位矩阵
        return np.mat(np.eye(self.dimention()))


def fill_diagonal(dst, filler):
    """
    fill filler into dst's diagonal, filler may be a matrix
     eg: use [[1,2],[3,4]] fill a mat[4*4]:0
      [[1, 2, 0, 0],
       [3, 4, 0, 0],
       [0, 0, 1, 2],
       [0, 0, 3, 4]])
    """
    assert (dst.shape[0] / filler.shape[0]) == (dst.shape[1] / filler.shape[1])
    n = int(dst.shape[0]/filler.shape[0])

    row, col = filler.shape
    for i in range(n):
        dst[i*row:(i+1)*row, i*col:(i+1)*col] = filler

    return dst


class MatMul(Operator):
    """
    matrix multiplicity
    """
    def calculate(self):
        # A:m*k  B:k*n  only and only 2 ops
        assert len(self.parents) == 2 and self.parents[0].shape()[1] == self.parents[1].shape()[0]
        self.value = self.parents[0].value * self.parents[1].value

    def get_jacobi(self, parent):
        """
        multi_ops partial
        C = A * B  A:m*n B:n*k, therefore, C:m*k
        数学上可以推出，C对A的Jacobi：m*k × m*n维度的广义对角阵
                     C对B的Jacobi：m*k × n*K维度的广义对角阵
        推导比较繁琐，基本思路是C、A、B按照行展开写成向量形式
        """
        zeros = np.mat(np.zeros((self.dimention(), parent.dimention())))
        if parent is self.parents[0]:
            return fill_diagonal(zeros, self.parents[1].value.T)
        else:
            jacobi = fill_diagonal(zeros, self.parents[0].value)
            row_sort = np.arange(self.dimention()).reshape(self.shape()[::-1]).T.ravel()  # flatten返回拉平的副本
            col_sort = np.arange(parent.dimention()).reshape(parent.shape()[::-1]).T.ravel()
            return jacobi[row_sort, :][:, col_sort]


class Logistic(Operator):
    """
    1.0 / (1.0 + exp(-x))
    """
    def calculate(self):
        x = self.parents[0].value
        self.value = np.mat(1.0 / (1.0 + np.power(np.e, np.where(-x > 1e2, 1e2, -x))))   # avoid overflow

    def get_jacobi(self, parent):
        return np.diag(np.mat(np.multiply(self.value, 1-self.value)).A1)


class ReLU(Operator):
    """
    Return max(eta*X,X)  leakRelu
    """
    slop = 0.2

    def calculate(self):
        self.value = np.mat(np.where(
                self.parents[0].value > 0.0,
                self.parents[0].value,
                self.slop * self.parents[0].value)
        )

    def get_jacobi(self, parent):
        return np.diag(np.where(self.parents[0].value.A1 > 0.0,  1.0,  self.slop))


class SoftMax(Operator):
    """

    """
    @staticmethod
    def softmax(x):
        # avoid overflow
        x[x > 1e2] = 1e2
        ex = np.power(np.e, x)
        return ex / np.sum(ex)

    def calculate(self):
        self.value = SoftMax.softmax(self.parents[0].value)

    def get_jacobi(self, parent):
        """
        CrossEntropy
        """
        raise NotImplementedError("we not use softmax's node jacobi")


class Reshape(Operator):
    """
    reshape parent
    """
    def __init__(self, *parent, **kargs):
        Operator.__init__(self, *parent, **kargs)
        self.to_shape = kargs.get('shape')
        assert isinstance(self.to_shape, tuple) and len(self.to_shape) == 2

    def calculate(self):
        self.value = self.parents[0].value.reshape(self.to_shape)

    def get_jacobi(self, parent):
        assert parent is self.parents[0]
        # 数学可以证明，reshape 的Jacobi是一个对角阵，reshape操作也是乘以一个对角阵
        return np.mat(np.eye(self.dimention()))


class Multiply(Operator):
    """
    matrix element wise multiplicity
    """
    def calculate(self):
        self.value = np.multiply(self.parents[0].value, self.parents[1].value)

    def get_jacobi(self, parent):
        if parent is self.parents[0]:
            return np.diag(self.parents[1].value.A1)
        else:
            return np.diag(self.parents[0].value.A1)


class Conv(Operator):
    """
    feature_map: parents[0]   filter: parents[1]
    """
    def __init__(self, *parents, **kargs):
        assert len(parents) == 2
        Operator.__init__(self, *parents, **kargs)
        self.padded = None

    def calculate(self):
        data = self.parents[0].value   # feature
        kernel = self.parents[1].value

        w, h = data.shape   # feature shape
        kw, kh = kernel.shape  # kernel shape
        _kw, _kh = int(kw/2), int(kh/2)  # half of kernel shape

        # padding
        pw, ph = tuple(np.add(data.shape, np.multiply((_kw, _kh), 2)))  # shape after padding
        self.padded = np.mat(np.zeros((pw, ph)))
        self.padded[_kw:_kw+w, _kh:_kh+h] = data
        self.value = np.mat(np.zeros(w, h))

        # 2d convolution
        for i in np.arange(_kw, _kw+w):
            for j in np.arange(_kh, _kh+h):
                self.value[i-_kw, j-_kh] = np.sum(
                    np.multiply(self.padded[i - _kw:i - _kw + kw, j - _kh:j - _kh + kh], kernel))

    def get_jacobi(self, parent):
        data = self.parents[0].value  # feature
        kernel = self.parents[1].value

        w, h = data.shape
        kw, kh = kernel.shape
        _kw, _kh = int(kw/2), int(kh/2)

        # padding
        pw, ph = tuple(np.add(data.shape, np.multiply((_kw, _kh), 2)))  # shape after padding

        jacobi = []
        if parent is self.parents[0]:
            for i in np.arange(_kw, _kw+w):
                for j in np.arange(_kh, _kh+h):
                    mask = np.mat(np.zeros((pw, ph)))
                    mask[i-_kw: i-_kw+kw, j-_kh: j-_kh+kh] = kernel
                    jacobi.append(mask[_kw:_kw+w, _kh:_kh+h].A1)
        elif parent is self.parents[1]:
            for i in np.arange(_kw, _kw+w):
                for j in np.arange(_kh, _kh+h):
                    jacobi.append(
                        self.padded[i-_kw:i-_kw+kw, j-_kh:j-_kh+kh])
        else:
            raise Exception("not current node's parent")

        return np.mat(jacobi)


class MaxPooling(Operator):
    """
    maxpooling
    """
    def __init__(self, *parent, **kargs):
        Operator.__init__(self, *parent, **kargs)
        self.stride = kargs.get('stride')
        assert self.stride is not None
        self.stride = tuple(self.stride)
        assert isinstance(self.stride, tuple) and len(self.stride) == 2

        self.size = kargs.get('size')
        assert self.size is not None
        self.size = tuple(self.size)
        assert isinstance(self.size, tuple) and len(self.size) == 2

        self.flag = None

    def calculate(self):
        data = self.parents[0].value  # feature map
        w, h = data.shape
        dim = w * h
        sw, sh = self.stride
        kw, kh = self.size  # pooling kernel size
        _kw, _kh = int(kw/2), int(kh/2)  # 1/2 of kernel size

        result = []
        flag = []

        for i in np.arange(0, w, sw):
            row = []
            for j in np.arange(0, h, sh):
                # get max_vale in pooling window
                top, bottom = max(0, i-_kw), min(w, i+_kw+1)
                left, right = max(0, j-_kh), min(h, j+_kh+1)
                window = data[top:bottom, left:right]
                row.append(np.max(window))

                # record max value position for bp
                pos = np.argmax(window)
                w_width = right - left
                offset_w, offset_h = top+pos//w_width, left+pos % w_width
                offset = offset_w * w + offset_h
                temp = np.zeros(dim)
                temp[offset] = 1
                flag.append(temp)
            result.append(row)

        self.flag = np.mat(flag)
        self.value = np.mat(result)

    def get_jacobi(self, parent):
        assert parent is self.parents[0] and self.jacobi is not None
        return self.flag


class Concat(Operator):
    """
    concatenate parents node
    """
    def calculate(self):
        assert len(self.parents) > 0
        self.value = np.concatenate([p.value.flatten() for p in self.parents], axis=1).T

    def get_jacobi(self, parent):
        assert parent in self.parents

        dims = [p.dimention() for p in self.parents]
        pos = self.parents.index(parent)  # index of parents
        dim = parent.dimention()  # cur parent node elements number

        assert dim == dims[pos]

        jacobi = np.mat(np.zeros((self.dimention(), dim)))
        start_row = int(np.sum(dims[:pos]))
        jacobi[start_row:start_row+dim, 0:dim] = np.eye(dim)

        return jacobi


class ScalarMultiply(Operator):
    """
    Scalar * matrix
    """
    def calculate(self):
        assert self.parents[0].shape() == (1, 1)  # parent[0] : scalar
        self.value = np.multiply(self.parents[0].value, self.parents[1].value)

    def get_jacobi(self, parent):
        assert parent in self.parents
        if parent is self.parents[0]:
            return self.parents[1].value.platten().T
        else:
            return np.mat(np.eye(self.parents[1].dimention())) * self.parents[0].value[0, 0]


class Step(Operator):
    """
    step function
    """
    def calculate(self):
        self.value = np.mat(np.where(self.parents[0].value >= 0.0, 1.0, 0.0))

    def get_jacobi(self, parent):
        np.mat(np.eye(self.dimention()))
        return np.zeros(np.where(self.parents[0].value.A1 > 0.0, 0.0, -1.0))