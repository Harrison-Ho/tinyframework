# -*- coding=utf-8 -*-

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import grpc
from ...core import Node
from ..dist import DistComm
from ..proto import parameter_server_pb2 as pspb
from ..proto import parameter_server_pb2_grpc as psrpc


class ParameterService(psrpc.ParameterServiceServicer):
    """
    Parameter service support sync-mode and async-mode
    sync_mode:
        1. all workers push gradients to ps
        2. all workers pull gradients to local
    async_node:
        workers visit ps randomly, update self gradient to ps or pull average gradient to local
    """

    def __init__(self, worker_num, sync=True):
        self.node_gradients_cache = dict()
        self.variable_weights_cache = dict()  # for init

        self.worker_num = worker_num
        self.sync = sync
        self.cur_push_num = 0
        self.cur_pull_num = self.worker_num

        self.cond = threading.Condition()
        self.push_lock = threading.Lock()
        self.init_lock = threading.Lock()
        self.is_init = False
        self.acc_no = 0

    def Push(self, push_req, context):
        """
        push gradients to ps
        """
        node_with_gradients, acc_no = self._deserialize_push_req(push_req)

        # store gradients to local cache
        if self.sync:
            self._push_sync(node_with_gradients, acc_no)
        else:
            self._push_async(node_with_gradients, acc_no)
        return pspb.ParameterPushResp()

    def _push_sync(self, node_with_gradients, acc_no):
        """ push sync mode """
        # add lock
        if self.cond.acquire():
            # waiting until all workers complete last iteration pull ops
            while self.cur_push_num != self.worker_num:
                self.cond.wait()

            # record push_nums
            self.cur_push_num += 1
            # update gradients to cache
            self._update_gradients_cache(node_with_gradients)
            # accelerate gradients nums
            self.acc_no += acc_no
            # if all workers complete gradients push, notify all to pull
            if self.cur_push_num >= self.worker_num:
                self.cur_pull_num = 0
                self.cond.notify_all()
            self.cond.release()
        else:
            self.cond.wait()

    def _push_async(self, node_with_gradients, acc_no):
        self.push_lock.acquire()
        self._update_gradients_cache(node_with_gradients)
        self.acc_no += acc_no
        self.push_lock.release()

    def Pull(self, pull_req, context):
        """
        pull gradient from ps
        """
        if self.sync:
            resp = self._pull_sync()
        else:
            resp = self._pull_async()
        return resp

    def _pull_sync(self):
        """
        sync mode
        """
        # add lock
        if self.cond.acquire():
            # waiting until all workers complete last pull ops
            while self.cur_push_num != self.worker_num:
                self.cond.wait()
            # record pull counts
            self.cur_pull_num += 1
            # calculate average gradient
            self._gradients_cache_mean()
            resp = self._serialize_pull_resp()

            # notify all workers push when complete pull
            if self.cur_pull_num >= self.worker_num:
                self.cur_pull_num = 0
                self._reset_gradients_cache()
                self.cond.notify_all()
            self.cond.release()
        else:
            self.cond.wait()
        return resp

    def _pull_async(self):
        """
        async mode
        """
        self.push_lock.acquire()
        self._gradients_cache_mean()
        resp = self._serialize_pull_resp()
        self._reset_gradients_cache()
        self.push_lock.release()
        return resp

    def _update_gradients_cache(self, node_with_gradients):
        # unsing node name to update gradients cache
        for node_name, gradients in node_with_gradients.items():
            if node_name in self.node_gradients_cache:
                exist_gradient = self.node_gradients_cache[node_name]
                assert exist_gradient.shape == gradients.shape
                self.node_gradients_cache[node_name] = exist_gradient + gradients
            else:
                self.node_gradients_cache[node_name] = gradients

    def _gradients_cache_mean(self):
        """
        get mean gradient in cache
        """
        if self.acc_no != 0:
            for name, gradient in self.node_gradients_cache.items():
                self.node_gradients_cache[name] = self.node_gradients_cache[name] / self.acc_no
            self.acc_no = 0

    def _deserialize_push_req(self, push_req):
        """
        deserialize push_req
        """
        acc_no = push_req.node_gradients.acc_no
        node_with_gradients = DistComm._deserialize_proto_node_gradients(push_req.node_gradients)
        return node_with_gradients, acc_no

    def _serialize_pull_resp(self):
        """
        serialize pull response
        """
        proto_node_gradients = DistComm._serialize_proto_node_gradients(self.node_gradients_cache)
        resp = pspb.ParameterPullResp(node_gradients=proto_node_gradients)
        return resp

    def _reset_gradients_cache(self):
        self.node_gradients_cache.clear()

    def VariableWeightsInit(self, variable_weights_req, context):
        """
        weight initialize, all worker push their init_value to ps, ps use the first value received as init_weight
        and notify all workers
        """
        self.init_lock.acquire()
        # if has not been initialized yet, use the first weight ps received
        if not self.is_init:
            self.variable_weights_cache = DistComm._deserialize_proto_variable_weights(variable_weights_req)
            print('[INIT] Parameter service variable weights initialized!')

        # other workers using exist init_weight
        resp = DistComm._serialize_proto_variable_weights(self.varibale_weights_cache)
        self.is_init = True
        self.init_lock.release()
        return resp


class ParameterServiceClient(object):
    """

    """

    def __init__(self, ps_host):
        # create grpc_stub
        self.stub = psrpc.ParameterServiceStub(grpc.insecure_channel(ps_host))
        assert self.stub is not None
        print('[GRPC] COnnected to parameter service:{}'.format(ps_host))

    def variable_weights_init(self, var_weights_dict):
        init_req = DistComm._serialize_proto_variable_weights(var_weights_dict)
        init_resp = self.stub.VariableWeightsInit(init_req)
        duplicate_var_weights_dict = DistComm._deserialize_proto_variable_weights(init_resp)
        return duplicate_var_weights_dict

    def push_gradients(self, acc_gradients, acc_no):
        # serialize gradients to proto obj
        proto_node_gradients = DistComm._serialize_proto_node_gradients(acc_gradients)
        proto_node_gradients.acc_no = acc_no
        # create req and push
        push_req = pspb.ParameterPushReq(node_gradients=proto_node_gradients)
        resp = self.stub.push(push_req)
        return resp

    def pull_gradients(self, node_name=None):
        # create pull req and pull
        pull_req = pspb.ParameterPullReq()
        pull_resp = self.stub.Pull(pull_req)
        # deserialize receive proto obj
        node_gradients_dict = DistComm._deserialize_proto_node_gradients(pull_resp.node_gradients)
        return node_gradients_dict


class ParameterServiceServer(object):
    """

    """

    def __init__(self, cluster_config, sync=True, max_workers=10):
        self.worker_num = len(cluster_config['workers'])
        self.host = cluster_config['ps'][0]
        self.sync = sync
        self.max_workers = max_workers

        self.server = grpc.server(ThreadPoolExecutor(max_workers=self.max_workers))
        psrpc.add_ParameterServiceServicer_to_server(ParameterService(self.worker_num, self.sync), self.server)
        self.server.add_insecure_port(self.host)

    def server(self):
        # start grpc service
        print('[PS] Parameter server (mode:{}) running on {} and worker num{}'.format('Sync' if self.sync else 'Async'),
              self.host, self.worker_num)
        try:
            while True:
                time.sleep(3600*24)
        except KeyboardInterrupt:
            self.server.stop(0)
