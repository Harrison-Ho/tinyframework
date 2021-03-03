# -*- coding:utf-8 -*-

import datetime
import json
import os

from ..core.core import get_node_from_graph
from ..core import Variable
from ..core.core import default_graph
from ..ops.metrics import *
from ..utils import ClassMining


class Saver(object):
    """
    save and load tools for model, graph
    include struct of graph and params
    """
    def __init__(self, root_dir=''):
        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

    def save(self, graph=None, meta=None, service_signature=None,
             model_file_name='model.json', weights_file_name='weights.npz'):
        if graph is None:
            graph = default_graph

        # meta records model save time and node_value filename
        meta = {} if meta is None else meta
        meta['save_time'] = str(datetime.datetime.now())
        meta['weights_file_name'] = weights_file_name

        # description for interface service
        service = {} if service_signature is None else service_signature

        self._save_model_and_weights(graph, meta, service, model_file_name, weights_file_name)

    def _save_model_and_weights(self, graph, meta, service, model_file_name, weights_file_name):
        model_json = {
            'meta': meta,
            'service': service
        }
        graph_json = []   # record node_json
        weights_dict = dict()

        # save node meta_info as dict/json
        for node in graph.nodes:
            if not node.need_save:
                continue
            node.kargs.pop('name', None)
            node_json = {
                'node_type': node.__class__.__name__,
                'name': node.name,
                'parents': [parent.name for parent in node.parents],
                'children': [child.name for child in node.children],
                'kargs': node.kargs
            }

            if node.value is not None:
                if isinstance(node.value, np.matrix):
                    node_json['dim'] = node.value.shape

            graph_json.append(node_json)

            if isinstance(node, Variable):
                weights_dict[node.name] = node.value

        model_json['graph'] = graph_json

        model_file_path = os.path.join(self.root_dir, model_file_name)
        with open(model_file_path, 'w') as model_file:
            json.dump(model_json, model_file, indent=4)
            print("Save model into file: {}".format(model_file.name))

        weights_file_path = os.path.join(self.root_dir, weights_file_name)
        with open(weights_file_path, 'wb') as weights_file:
            np.savez(weights_file, **weights_dict)
            print("Save weights to file: {}".format(weights_file.name))

    @staticmethod
    def create_node(graph, from_model_json, node_json):
        """
         Recursively create nodes who do not exist
        """
        node_type = node_json['node_type']
        node_name = node_json['name']
        parents_name = node_json['parents']
        dim = node_json.get('dim', None)
        kargs = node_json.get('kargs', None)

        parents = []
        for parent_name in parents_name:
            parent_node = get_node_from_graph(parent_name, graph=graph)
            if parent_node is None:
                parent_node_json = None
                for node in from_model_json:
                    if node['name'] == parent_name:
                        parent_node_json = node
                assert parent_node_json is not None
                # 如果父节点不存在，递归调用
                parent_node = Saver.create_node(
                    graph, from_model_json, parent_node_json)

            parents.append(parent_node)

        # Create node instances by reflex
        if node_type == 'Variable':
            assert dim is not None

            dim = tuple(dim)
            return ClassMining.get_instance_by_subclass_name(Node, node_type)(*parents, dim=dim, name=node_name, **kargs)
        else:
            return ClassMining.get_instance_by_subclass_name(Node, node_type)(*parents, name=node_name, **kargs)

    def _restore_nodes(self, graph, from_model_json, from_weights_dict):
        for i in range(len(from_model_json)):
            node_json = from_model_json[i]
            node_name = node_json['name']

            weights = None
            if node_name in from_weights_dict:
                weights = from_weights_dict[node_name]

            # whether cur node has been created, if exist update weight else create is
            target_node = get_node_from_graph(node_name, graph=graph)
            if target_node is None:
                print("Target node {} of Type {} not exist, try to create the instance".format(node_json['name'], node_json['node_type']))
                target_node = Saver.create_node(graph, from_model_json, node_json)

            target_node.value = weights

    def load(self, to_graph=None, model_file_name='model.json', weights_file_name='weights.npz'):
        """
        Read and restore the structure of the calculation graph and the corresponding values from the file
        """
        if to_graph is None:
            to_graph = default_graph

        model_json = {}
        graph_json = []
        weights_dict = dict()

        model_file_path = os.path.join(self.root_dir, model_file_name)
        with open(model_file_path, 'r') as model_file:
            model_json = json.load(model_file)

        weights_file_path = os.path.join(self.root_dir, weights_file_name)
        with open(weights_file_path, 'rb') as weights_file:
            weights_npz_files = np.load(weights_file)
            for file_name in weights_npz_files.files:
                weights_dict[file_name] = weights_npz_files[file_name]
            weights_npz_files.close()

        graph_json = model_json['graph']
        self._restore_nodes(to_graph, graph_json, weights_dict)
        print('Load and restore model from {} and {}'.format(model_file_path, weights_file_path))

        self.meta = model_json.get('meta', None)
        self.service = model_json.get('service', None)
        return self.meta, self.service
