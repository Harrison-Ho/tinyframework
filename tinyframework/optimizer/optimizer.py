# -*- coding: utf-8 -*-

import abc
import numpy as np
from ..core import Node, Variable, get_node_from_graph
from ..core.graph import Graph


class Optimizer(object):
    """
    Optimizer base class
    steps：
        1. 调用 loss的forward方法
        2. 对所有可训练节点调用backward方法得到loss对该节点的jacobi
        3. 做可训练节点value(参数)更新，更新完清除，返回1
    """
    def __init__(self, graph, target, learning_rate=0.005):
        """
        optimizer init receive a calculation graph object, target node and learning rate
        """
        assert isinstance(target, Node) and isinstance(graph, Graph)
        self.graph = graph
        self.target = target
        self.learning_rate = learning_rate

        self.acc_gradient = dict()
        self.acc_no = 0

    def one_step(self):
        """
        calculate and accumulate(累加) the samples gradient of batch
        include a fp and bp
        """
        self.forward_backward()
        self.acc_no += 1

    def get_gradient(self, node):
        """
        return mean gradient for node
        """
        assert node in self.acc_gradient
        return self.acc_gradient[node] / self.acc_no

    @abc.abstractmethod
    def _update(self):
        """
        implement in the subClass, GD, RMSProp,Momentum,Adam...
        """

    def forward_backward(self):
        """
        for every sample，bp to calculate predict, then fp calculate jacobi(gradient)
        """
        self.graph.clear_jacobi()
        # bp
        self.target.forward()
        # fp
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                node.backward(self.target)
                # target 若为标量，target对节点的jacobi是行向量，将其转置后成为列向量即为梯度向量
                # 将梯度向量reshape成node形状，便于节点值更新
                gradient = node.jacobi.T.reshape(node.shape())
                if node not in self.acc_gradient:
                    self.acc_gradient[node] = gradient
                else:
                    self.acc_gradient[node] += gradient

    def apply_gradients(self, node_gradients_dict, summarize=False, acc_no=None):
        """
        compatible with gradient collection in distributed training
        """
        for node, gradient in node_gradients_dict.items():
            if isinstance(node, Node):
                pass
            else:
                target_node = get_node_from_graph(node)
                assert target_node is not None
                assert self.acc_gradient[target_node].shape == gradient.shape
                if summarize:
                    self.acc_gradient[target_node] += gradient
                else:
                    self.acc_gradient[target_node] = gradient

        if summarize:
            self.acc_no += acc_no  # for distribute training
        else:
            if acc_no is None:
                # 若输入的是平均梯度，则令acc_no=1，避免梯度更新时再次平均
                self.acc_no = 1
            else:
                self.acc_no = acc_no

    def update(self, var_gradients=None):
        if var_gradients is not None:
            self.apply_gradients(var_gradients)  # used for distribute calculate
        self._update()
        # clear accumulate gradient dict and acc_num
        self.acc_gradient.clear()
        self.acc_no = 0


class GradientDescent(Optimizer):
    """
    gradient descent optimizer
    """
    def __init__(self, graph, target, learning_rate=0.005):
        Optimizer.__init__(self, graph, target)
        self.learning_rate = learning_rate

    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                # get every node's mean gradient
                gradient = self.get_gradient(node)
                node.set_value(node.value - self.learning_rate * gradient)


class Momentum(Optimizer):
    """
    momentum optimizer
    动量法包括了速度更新和权重更新两步
    gradient = ▽f(w)
    V = momentum * V - lr * gradient
    w = w + v
    """
    def __init__(self, graph, target, learning_rate=0.01, momentum=0.9):
        Optimizer.__init__(self, graph, target)
        self.learning_rate = learning_rate
        self.momentum = momentum  # 衰减系数默认设置为0.9
        self.V = dict()  # record history velocity

    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                # get the node's mean gradient
                gradient = self.get_gradient(node)
                if node not in self.V:
                    self.V[node] = gradient
                else:
                    self.V[node] = self.momentum * self.V[node] - self.learning_rate * gradient

                node.set_value(node.value + self.V[node])


class AdaGrad(Optimizer):
    """
    AdaGrad Optimizer
    变学习率的优化方法,历史梯度大，调小学习率，反之亦然
    gradient = ▽f(w)
    s = s + gradient * gradient
    w = w - learning_rate/sqrt(s+小值）* gradient  小值防止一开始梯度为0导致分子为0
    """
    def __init__(self, graph, target, learning_rate=0.01):
        Optimizer.__init__(self, graph, target)
        self.learning_rate = learning_rate
        self.s = dict()

    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                # get the node's mean gradient
                gradient = self.get_gradient(node)
                # accumulate the square of gradient
                if node not in self.s:
                    self.s[node] = np.power(gradient, 2)
                else:
                    self.s[node] += np.power(gradient, 2)

                # update weight
                node.set_value(node.value - self.learning_rate * gradient / (np.sqrt(self.s[node] + 1e-10)))


class RMSProp(Optimizer):
    """
    Root Mean Square Propagation
    AdaGrad对历史梯度的平方做了累计，此法不妥，应尽可能考虑近期梯度，更远一些的给与衰减
    gradient = ▽f(w)
    s = β * s + （1 - β）*gradient * gradient
    w = w - learning_rate* gradient / sqrt(s+小值）  小值防止一开始梯度为0导致分子为0
    """
    def __init__(self, graph, target, learning_rate=0.01, beta=0.9):
        Optimizer.__init__(self, graph, target)
        self.learning_rate = learning_rate
        assert 0.0 < beta < 1.0
        self.beta = beta
        self.s = dict()

    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                # get the node's mean gradient
                gradient = self.get_gradient(node)

                if node not in self.s:
                    self.s[node] = np.power(gradient, 2)
                else:
                    self.s[node] = self.beta * self.s[node] + (1 - self.beta) * np.power(gradient, 2)

                node.set_value(node.value - self.learning_rate * gradient / np.sqrt(self.s[node] + 1e-10))


class Adam(Optimizer):
    """
    Adaptive Momentum Estimation Optimizer
    Momentum 累计了历史梯度，AdaGrad是变学习率，RMSProp累计了历史梯度的平方变化学习率
    Adam结合了Momentum和RMSProp，同时利用嘞V和S，两个超参β_v 和 β_s

    gradient = ▽f(w)
    V = β_v * V - （1 - β_v） * gradient
    s = β_s * s + （1 - β_s） * gradient * gradient
    w = w - leaning_rate * V / sqrt(s+小值) 小值防止一开始梯度为0导致分子为0
    """
    def __init__(self, graph, target, learning_rate=0.01, beta_v=0.9, beta_s=0.95):
        Optimizer.__init__(self, graph, target)
        self.learning_rate = learning_rate
        assert 0.0 < beta_v < 1.0
        self.beta_v = beta_v
        assert 0.0 < beta_s < 1.0
        self.beta_s = beta_s
        self.v = dict()
        self.s = dict()

    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                # get the node's mean gradient
                gradient = self.get_gradient(node)

                if node not in self.v:
                    self.v[node] = gradient
                    self.s[node] = np.power(gradient, 2)
                else:
                    self.v[node] = self.beta_v * self.v[node] + (1 - self.beta_v) * gradient
                    self.s[node] = self.beta_s * self.s[node] + (1 - self.beta_s) * np.power(gradient, 2)

                node.set_value(node.value - self.learning_rate * self.v[node] / np.sqrt(self.s[node] + 1e-10))