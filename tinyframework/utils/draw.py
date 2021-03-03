# -*- coding:utf-8 -*-

from pyecharts import options as ops
from pyecharts.charts import Graph

from ..core.graph import default_graph


def draw_graph(filename=''):
    nodes_for_draw = []
    edges_for_draw = []
    for node in default_graph.nodes:
        nodes_for_draw.append({'name': node.name, "symbolSize": 50})
    for node in default_graph.nodes:
        for child in node.children:
            edges_for_draw.append({'source':node.name, 'target': child.name})
    graph = Graph(init_opts=ops.InitOpts(width='1800px', height='1000px'))
    graph.set_global_opts(title_opts=ops.TitleOpts(title="TinyFramework"))
    graph.add("", nodes_for_draw, edges_for_draw, layout='force', repulsion=8000, edge_symbol=['circle', 'arrow'])
    if filename == '':
        file_name = filename
    else:
        file_name = filename + '_'
    graph.render('./'+file_name+'graph.html')