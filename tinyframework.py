# -*- coding:utf-8 -*-

from .import core
from .import layer
from .import ops
from .import optimizer
from .import utils
from .import trainer

default_graph = core.default_graph
get_node_from_graph = core.get_node_from_graph
name_scope = core.NameScope
Variable = core.Variable
