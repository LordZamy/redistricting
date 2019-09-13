import networkx as nx
import json
import numpy as np

class Dumper():
    def __init__(self):
        self.color_list = []

    def add(self, graph):
        colors = nx.get_node_attributes(graph, 'color')
        self.color_list.append(colors.copy())

    def convert(o):
        if isinstance(o, np.int64): return int(o)
        raise TypeError

    def dump(self, name, num_iterations, beta_type, beta, use_weight_scaling):
        with open(name, 'w') as f:
            output = {'num_iterations': num_iterations, 'beta_type': beta_type, 'beta': beta, 'use_weight_scaling': use_weight_scaling, 'color_list': self.color_list}
            json.dump(output, f, default=Dumper.convert)
