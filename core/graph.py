# -*- coding:utf-8 -*-

class Graph(object):
    """
    TinyFramework 计算图
    """
    def __init__(self):
        self.nodes = []
        self.name_scope = None

    def add_node(self, node):
        self.nodes.append(node)

    def clear_jacobi(self):
        """
        when 。。。 need to clear jacobi matrix
        """
        for node in self.nodes:
            node.clear_jacobi()   # node 类需要实现clear jacobi方法

    def reset_value(self):
        """
        节点值重置
        """
        for node in self.nodes:
            node.reset_value()    # as above

    def node_cnt(self):
        return len(self.nodes)

    def draw(self):
        pass  #


default_graph = Graph()