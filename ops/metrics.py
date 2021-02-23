# -*- coding: utf-8 -*-

import numpy as np
import abc
from ..core import Node


class Metrics(Node):
    """
    Metrics abstract class
    """
    def __init__(self, *parents, **kargs):
        kargs['need_save'] = kargs.get('need_save', False)
        Node.__init__(self, *parents, **kargs)
        self.init()

    def reset(self):
        self.reset_value()
        self.init()

    @abc.abstractmethod
    def init(self):
        """
        depend on metrics ,case by case
        """
        pass

    @staticmethod
    def prob2label(prob, thresholds=0.5):
        if prob.shape[0] > 1:
            # multi-classifier
            labels = np.zeros((prob.shape[0], 1))
            labels[np.argmax(prob, axis=0)] = 1
        else:
            # 2-classifier
            labels = np.where(prob < thresholds, -1, 1)

        return labels

    def get_jacobi(self, parent):
        """
        calculate jacobi have no significance for metrics
        """
        raise NotImplementedError()

    def value2str(self):
        return "{}:{:.4f}".format(self.__class__.__name__, self.value)


class Accuracy(Metrics):
    """
    for accuracy
    Acc = (TP+TN)/All
    """
    def __init__(self, *parents, **kargs):
        Metrics.__init__(self, *parents, **kargs)

    def init(self):
        self.correct_num = 0
        self.total_num = 0

    def calculate(self):
        """
        assume parents[0] as prob, parents[1] as label
        """
        pred = Metrics.prob2label(self.parents[0].value)
        groundtruth = self.parents[1].value
        assert len(pred) == len(groundtruth)
        if pred.shape[0] > 1:  # add annotations
            self.correct_num += np.sum(np.multiply(pred, groundtruth))
            self.total_num += pred.shape[1]
        else:
            self.correct_num += np.sum(pred == groundtruth)
            self.total_num += len(pred)
        self.value = 0
        if self.total_num > 0:
            self.value = float(self.correct_num) / self.total_num


class Precision(Metrics):
    """
    for precision
    precision = TP/(TP+FP)
    also assume parents[0] as prob, parents[1] as label
    """
    def __init__(self, *parents, **kargs):
        Metrics.__init__(self, *parents, **kargs)

    def init(self):
        self.pred_positive = 0   # pread as P
        self.true_positive = 0   # pread as P and real as P

    def calculate(self):
        assert self.parents[0].value.shape[1] == 1
        pred_label = Metrics.prob2label(self.parents[0].value)
        groundtruth = self.parents[1].value
        self.pred_positive += np.sum(pred_label == 1)
        self.true_positive += np.sum(pred_label == groundtruth and pred_label == 1)
        self.value = 0
        if self.pred_positive > 0:
            self.value = float(self.true_positive) / self.pred_positive


class Recall(Metrics):
    """
    for recall
    recall = TP/(TP+FN)
    also assume parents[0] as prob, parents[1] as label
    """
    def __init__(self, *parents, **kargs):
        Metrics.__init__(self, *parents, *kargs)

    def init(self):
        self.real_positive = 0  # groudtrue_positive
        self.true_positive = 0  # pred as P and real as P

    def calculate(self):
        assert self.parents[0].value.shape[0] == 1
        pred_label = Metrics.prob2label(self.parents[0].value)
        groundtruth = self.parents[1].value

        self.real_positive += np.sum(groundtruth == 1)
        self.true_positive += np.sum(pred_label == groundtruth and pred_label == 1)
        self.value = 0
        if self.true_positive > 0:
            self.value = float(self.true_positive) / self.real_positive


class ROC(Metrics):
    """
    ROC curve
    """
    def __init__(self, *parents, **kargs):
        Metrics.__init__(self, *parents, **kargs)

    def init(self):
        self.count = 100  # set 100 threshold points
        self.real_positive = 0
        self.real_negative = 0
        self.true_positive = np.array([0] * self.count)  # pred 1 and real 1
        self.false_positive = np.array([0] * self.count) # pred 1 but real 0
        self.tpr = np.array([0] * self.count)
        self.fpr = np.array([0] * self.count)

    def calculate(self):
        prob = self.parents[0].value
        groundtruth = self.parents[1].value
        self.real_positive += np.sum(groundtruth == 1)
        self.real_negative += np.sum(groundtruth == 0)

        thresholds = list(np.arange(0.01, 1.00, 0.01))  #  99
        # using thresholds to generate TP and FP
        for idx in range(0, len(thresholds)):
            pred = Metrics.prob2label(prob, thresholds=thresholds[idx])
            self.true_positive[idx] += np.sum(pred == groundtruth and pred == 1)
            self.false_positive[idx] += np.sum(pred != groundtruth and pred == 1)

        # calculate tpr and fpr
        if self.true_positive > 0 and self.false_positive > 0:
            self.tpr = self.true_positive / self.real_positive
            self.fpr = self.false_positive / self.real_negative
            """
            draw curve
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.plot(self.fpr, self.tpr)
            plt.show()
            """


class AUC(Metrics):
    """
    calculate Area under curve
    """
    def __init__(self, *parents, **kargs):
        Metrics.__init__(self, *parents, *kargs)

    def init(self):
        self.real_positive = []
        self.real_negative = []

    def calculate(self):
        prob = self.parents[0].value
        groundtruth = self.parents[1].value

        if groundtruth[0, 0] == 1:
            self.real_positive.append(prob)
        else:
            self.real_negative.append(prob)
        self.total_area = len(self.real_positive) * len(self.real_negative)

    def auc_area(self):
        count = 0
        # Visit m x n sample pairs, calculate the number of positive probability greater than negative probability
        for real_p in self.real_positive:
            for real_n in self.real_negative:
                if real_p > real_n:
                    count += 1
        self.value = float(count) / self.total_area


class F1Score(Metrics):
    """
    F1score = 2 * (precision * recall) / (precision + recall)
    """
    def __init__(self, *parents, **kargs):
        Metrics.__init__(self, *parents, **kargs)
        self.true_positive = 0
        self.pred_positive = 0
        self.real_positive = 0

    def calculate(self):
        assert self.parents[0].value.shape[1] == 1
        pred = Metrics.prob2label(self.parents[0].value)
        groundtruth = self.parents[1].value

        self.pred_positive += np.sum(pred)
        self.real_positive += np.sum(groundtruth)
        self.true_positive += np.multiply(pred, groundtruth).sum()
        self.value = 0
        precision = 0
        recall = 0

        if self.pred_positive > 0:
            precision = float(self.true_positive) / self.pred_positive

        if self.real_positive > 0:
            precision = float(self.true_positive) / self.real_positive

        if precision + recall > 0:
            self.value = 2 * (np.multiply(precision, recall)) / (precision + recall)