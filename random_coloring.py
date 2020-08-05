import networkx as nx

from random import randrange

# randomizes the colors of a given graph
def randomize(graph, num_colors):
    random_colors = {node: randrange(0, num_colors) for node in graph.nodes()}
    nx.set_node_attributes(graph, random_colors, 'color')
