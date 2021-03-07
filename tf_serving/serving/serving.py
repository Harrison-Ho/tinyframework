# -*- coding:utf-8 -*-

import threading
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import grpc
import tinyframework as tf
from .proto import serving_pb2, serving_pb2_grpc


class TinyFrameworkServingService(serving_pb2_grpc.TinyFrameworkServingServicer):
    """
    inference service, steps as below:
    1. get input/output nodes in graph according to the interface signature defined in the model file
    2. receive request an deserialize model input
    3. call calculate graph to inference
    4. get output node value and return to interface caller
    """

    def __init__(self, root_dir, model_file_name, weights_file_name):
        self.root_dir = root_dir
        self.model_file_name = model_file_name
        self.weights_file_name = weights_file_name

        saver = tf.trainer.Saver(self.root_dir)

        # load from file and deserialize graph struct and weights, get serving interface signature
        _, service = saver.load(model_file_name=self.model_file_name, weights_file_name=self.weights_file_name)
        assert service is not None

        inputs = service.get('inputs', None)  # get input nodes name
        assert inputs is not None

        outputs = service.get('outputs', None)
        assert outputs is not None

        # get inputs/outputs node from graph according service signature
        self.input_node = tf.get_node_from_graph(inputs['name'])
        assert self.input_node is not None
        assert isinstance(self.input_node, tf.Variable)

        self.input_dim = self.input_node.dim

        self.output_node = tf.get_node_from_graph(outputs['name'])
        assert self.output_node is not None

    @staticmethod
    def deserialize(predict_request):
        infer_req_mat_list = []
        for proto_mat in predict_request.date:
            dim = tuple(proto_mat.dim)
            mat = np.mat(proto_mat.value, dtype=np.float32)
            mat = np.reshape(mat, dim)
            infer_req_mat_list.append(mat)

        return infer_req_mat_list

    def _inference(self, inference_request):
        inference_resq_mat_list = []
        for mat in inference_request:
            self.input_node.set_value(mat.T)
            self.output_node.forward()
            inference_resq_mat_list.append(self.output_node.value)

        return inference_resq_mat_list

    @staticmethod
    def serialize(inference_response):
        response = serving_pb2.PredictResponse()
        for mat in inference_response:
            proto_mat = response.data.add()
            proto_mat.value.extend(np.array(mat).flatten())
            proto_mat.dim.extent(list(mat.shape))

        return response

    def Predict(self, predict_request, context):
        # deserialize protobuf data into np.Mat
        inference_request = TinyFrameworkServingService.deserialize(predict_request)
        # call graph to execute output nodes forward()
        inference_response = self._inference(inference_request)
        # serialize the inference result to protobuf format
        predict_response = TinyFrameworkServingService.serialize(inference_response)

        return predict_response


class TinyFrameworkServer(object):
    """

    """
    def __init__(self, host, root_dir, model_file_name, weights_file_name, max_workers=10):
        self.host = host
        self.max_workers = max_workers
        self.server = grpc.server(ThreadPoolExecutor(max_workers=self.max_workers))

        serving_pb2_grpc.add_TinyFrameworkServingServicer_to_server(
            TinyFrameworkServingService(root_dir, model_file_name, weights_file_name), self.server)

        self.server.add_insecure_port(self.host)

    def serve(self):
        # start rpc serving
        self.server.start()
        print("TinyFramework server running on {}".format(self.host))

        try:
            while True:
                time.sleep(3600 * 24)
        except KeyboardInterrupt:
            self.server.stop(0)