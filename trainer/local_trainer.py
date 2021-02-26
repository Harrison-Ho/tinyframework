# -*- coding:utf-8 -*-

from .trainer import Trainer


class LocalTrainer(Trainer):
    """

    """
    def __init__(self, *args, **kargs):
        Trainer.__init__(self, *args, **kargs)

    def _variable_weights_init(self):
        pass

    def _optimizer_update(self):
        self.optimzer.update()
