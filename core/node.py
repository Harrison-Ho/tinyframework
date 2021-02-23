# -*- coding： utf-8 -*-

import abc
import numpy as np
from .graph import Graph, default_graph


class Node(object):
    """
    TinyFramework 计算图 Node Basic class
    """
    def __init__(self, *parents, **kargs):
        self.kargs = kargs
        self.graph = kargs.get('graph', default_graph)
        self.need_save = kargs.get('need_save', True)
        #self.gen_node_name(**kargs)
        self.parents = list(parents) #接受其它Node类的一个or多个parents对象
        self.children = []
        self.value = None
        self.jacobi = None  # 特指从loss node到本节点的jacobi matrix
        """
        assign node name
        default name if user not assgin, like 'logist:2'
        if name_scope assigned, node_name is updateed like 'outlayer/ logist:2'
        """
        self.name = kargs.get('name', '{}:{}'.format(self.__class__.__name__, self.graph.node_cnt()))
        if self.graph.name_scope:
            self.name = kargs.get('name', '{}/{}'.format(self.graph.name_scope, self.name))

        for parent in self.parents:
            parent.children.append(self)  # add self to parents.children list

        # add new node to graph
        self.graph.add_node(self)

    def dimention(self):
        """
        dimention: rows * columns
        """
        return self.value.shape[0] * self.value.shape[1]

    def get_parents(self):
        return self.parents

    def get_children(self):
        return self.children

    def gen_node_name(self):
        return self.name

    def shape(self):
        return self.value.shape

    def reset_value(self, recursive=True):
        self.value = None
        if recursive:
            for child in self.children:
                child.reset_value()

    def forward(self):
        """
        Recursively calculate the value of the node
        """
        for node in self.parents:
            if node.value is None: # for complex graph, a node may have many parents
                node.forward()
        self.calculate()

    @abc.abstractmethod
    def calculate(self):
        """
        calculate self by parent node
        """
    @abc.abstractmethod
    def get_jacobi(self, parent):
        """
        calculate self to every parent node jacobi
        need to implement in subClass
        f(w+δ) = f(w) + ▽f * δ ，
        若 f()是标量，则▽是梯度，
        若 f()是向量，则以▽为行组成的矩阵为jacobi矩阵
        """

    def backward(self, result):  # actually, bp return a jacobi of loss->any node
        if self.jacobi is None: # for complex graph，a node may be visited many times
            if self is result:
                self.jacobi = np.mat(np.eye(self.dimention())) # 节点对自身的Jacobi是单位矩阵
            else:
                self.jacobi = np.mat(np.zeros((result.dimention(), self.dimention())))  # 构造0矩阵作为累加器
            """
            记某个节点为 f, f的子节点为s(可能为多个)，结果节点对f的Jacobi记为Jrf，结果节点对s的Jacobi记为Jrs
            每个子节点s对父节点f的Jacobi记为Jsf，数学上可以证明 Jrf = ∑（Jrs * Jsf） 
            """
            for child in self.get_children():
                if child.value is not None:
                    self.jacobi += child.backward(result) * child.get_jacobi(self)

        return self.jacobi

    def clear_jacobi(self):
        """
        一次前向传播后，计算出pred，与label运算后得到loss，求loss对每个可训练节点的jacobi视为bp
        bp过程中可以更新节点的value，下一次fp后再bp时，因value得到了更新，所以要清除jacobi
        """
        self.jacobi = None

"""
将节点抽象为两类，一类是op节点，用于向量/矩阵的加、减、乘、除、reshape ..etc.. 在ops包中继承Node节点实现
               一类是Variable节点，用于weight、input、pred、loss..etc..
"""

class Variable(Node):
    """
    diffirent to op node, Variable has no parents, therefore, it must be assigned dimention
    """
    def __init__(self, dim, init=False, trainable=True, **kargs):
        Node.__init__(self, **kargs)
        self.dim = dim
        if init:
            self.value = np.mat(np.random.normal(0, 0.01, self.dim))
        self.trainable = trainable

    def set_value(self, value):
        assert isinstance(value, np.matrix) and value.shape == self.dim  # 类型和维度判断
        self.reset_value()
        self.value = value