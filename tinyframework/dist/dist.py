# -*- coding=utf-8 -*-

import numpy as np
from ..core import Node
from .proto import comm_pb2


class DistComm(object):
    @staticmethod
    def _serialize_proto_node_gradients(node_gradients_dict):
        """
        serialize node_gradient dict to protobuf object
        """
        proto_node_gradients = comm_pb2.NodeGradients()
        for name, g in node_gradients_dict.items():
            proto_node = proto_node_gradients.nodes.add()
            if isinstance(name, Node):
                name = name.name
            proto_node.name = name
            proto_gradient = proto_node_gradients.gradients.add()
            proto_gradient.value.extend(np.array(g).flatten())
            proto_gradient.dim.extend(list(g.shape))

        return proto_node_gradients

    @staticmethod
    def _deserialize_proto_node_gradients(node_gradients):
        """
        deserialize proto obj to node_gradient_dict
        """
        proto_nodes = node_gradients.nodes
        proto_gradients = node_gradients.gradients
        assert len(proto_nodes) == len(proto_gradients)
        node_with_gradients = dict()

        for idx in range(len(proto_nodes)):
            node_name = proto_nodes[idx].name
            gradients_value = proto_gradients[idx].value
            gradients_dim = tuple(proto_gradients[idx].dim)
            gradients_mat = np.mat(gradients_value, dtype=np.float32)
            gradients_mat = np.reshape(gradients_mat, gradients_dim)
            node_with_gradients[node_name] = gradients_mat

        return node_with_gradients

    @staticmethod
    def _serialize_proto_variable_weights(variable_weights_dict):
        """
        serialize var_weight to proto obj
        """
        var_weights_req_resp = comm_pb2.VariableWeightsReqResp()
        for name, mat in variable_weights_dict.items():
            var = var_weights_req_resp.variables.add()
            if isinstance(name, Node):
                name = name.name
            var.name = name
            weight = var_weights_req_resp.weights.add()
            weight.value.extend(np.array(mat).flatten())
            weight.dim.extent(list(mat.shape))

        return var_weights_req_resp

    @staticmethod
    def _deserialize_proto_variable_weights(variable_weights_req_resp):
        """
        deserialize proto obj to var-weights dict
        """
        proto_variables = variable_weights_req_resp.variables
        proto_weights = variable_weights_req_resp.weights
        assert len(proto_weights) == len(proto_variables)

        var_weights_dict = dict()
        for idx in range(len(proto_weights)):
            var_name = proto_variables[idx].name
            weights_value = proto_weights[idx].value
            weights_dim = tuple(proto_weights[idx].dim)
            weights_mat = np.mat(weights_value, dtype=np.float32)
            weights_mat = np.reshape(weights_mat, weights_dim)
            var_weights_dict[var_name] = weights_mat

        return var_weights_dict

