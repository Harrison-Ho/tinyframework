# -*- coding:utf-8 -*-

import abc
from ..core import Variable, default_graph
import numpy as np


class Trainer(object):
    """

    """
    def __init__(self, input_x, input_y, loss_op, optimizer, epoches, batch_size=16,
                 is_eval_on_train=False, metrics_ops=None, *args, **kargs):
        """
        :param input_x:  allow graph has multi_input, for correctly match node and sample
                        here, data input as dict, key:node_name, value:sample_value
        :param input_y: the same as input_x
        :param loss_op:
        :param optimizer:
        :param epoches:
        :param batch_size:
        :param is_eval:
        :param metrics_ops: type:list, include acc, precision, recall ... one or more
        :param args:
        :param kargs:
        """
        self.inputs = input_x
        self.input_y = input_y

        self.loss_op = loss_op
        self.optimizer = optimizer

        self.epochs = epoches
        self.epoch = 0
        self.batch_size = batch_size

        self.is_eval = is_eval_on_train
        self.metrics_ops = metrics_ops

        self.print_iteration_interval = kargs.get("print_iteration_interval", 100)

    def train_and_eval(self, train_x, train_y, test_x=None, test_y=None):
        """
        start train(evaluate)
        """
        assert len(train_x) == len(self.inputs)

        if test_x is not None and test_y is not None:
            assert len(test_x) == len(self.inputs)

        # init weights
        self._variable_weights_init()
        print('INIT Variable weight init finished')

        self.train_loop(train_x, train_y, test_x, test_y)

    def train_loop(self, train_x, train_y, test_x, test_y):
        """
        for every epoch, start training
        """
        for self.epoch in range(self.epochs):
            self.train(train_x, train_y)

            if self.is_eval and test_x is not None and test_y is not None:
                self.eval(test_x, test_y)

    def train(self, train_x, train_y):
        for i in range(len(list(train_x.values())[0])):
            # for every sample, execute fp and bp
            self.one_step(self._get_input_values(train_x, i), train_y[i])
            if (i+1) % self.batch_size == 0:
                self._optimizer_update()

    def eval(self, test_x, test_y):
        """
        :param test_x:
        :param test_y:
        :return:
        """
        # for every metrics way
        for metrics_op in self.metrics_ops:
            metrics_op.reset_value()
        # for every simple
        for i in range(len(list(test_x.values())[0])):
            self.one_step(self._get_input_values(test_x, i), test_y[i], is_eval=True)
            for metrics_op in self.metrics_ops:
                metrics_op.forward()

        metrics_str = 'Epoch[{}] evaluation metrics'.format(self.epoch+1)
        for metrics_op in self.metrics_ops:
            metrics_str += metrics_op.value2str()
        print(metrics_str)



    def one_step(self, data_x, data_y, is_eval=False):
        """
        execute one step on fp and bp(only on train)
        is_eval_step=True, parameters will not be updated
        """
        # for inputs every node
        for i in range(len(self.inputs)):
            #  find value from input dict by input node name
            input_value = data_x.get(self.inputs[i].name)
            self.inputs[i].set_value(np.mat(input_value).T)

        self.input_y.set_value(np.mat(data_y).T)
        # only on train stage, optimizer will execute
        if not is_eval:
            self.optimizer.one_step()

    def _get_input_values(self, x, index):
        """
        :param x: dict class
        :param index:
        :return:
        """
        input_values = dict()
        # for inputs every node
        for input_node_name in x.keys():
            input_values[input_node_name] = x[input_node_name][index]
        return input_values

    @abc.abstractmethod
    def _variable_weights_init(self):
        """
        implement in subclass
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _optimizer_update(self):
        """
        the way to update parameters, implement in sub class
        """
        raise NotImplementedError()
