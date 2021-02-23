# -*- coding: utf-8 -*-

from ..core import Node
from ..ops import SoftMax
import numpy as np


class LossFunction(Node):
    """
    loss function abstract class
    """
    pass


class LogLoss(LossFunction):
    def calculate(self):
        assert len(self.parents) == 1
        x = self.parents[0].value
        self.value = np.log(1 + np.power(np.e, np.where(-x > 1e2, 1e2, -x)))

    def get_jacobi(self, parent):
        x = parent.value
        diag = -1 / (1 + np.power(np.e, np.where(x > 1e2, 1e2, x)))
        return np.diag(diag.ravel())


class CrossEntropy(LossFunction):
    """
    softmax(parents[0]) , parents[1] is label (one_hot)
    """
    def calculate(self):
        prob = SoftMax.softmax(self.parents[0].value)
        self.value = np.mat(
            -np.sum(np.multiply(self.parents[1].value, np.log(prob + 1e-10))))

    def get_jacobi(self, parent):
        prob = SoftMax.softmax(self.parents[0].value)
        if parent is self.parents[0]:
            return (prob - self.parents[1].value).T
        else:
            return (-np.log(prob)).T
