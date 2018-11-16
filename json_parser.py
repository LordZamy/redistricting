import json
import networkx as nx
import numpy as np

def parse_graph(location):
    with open(location) as f:
        data = json.load(f)

    nodes = data['nodes']
    graph = nx.Graph()
    bad_nodes = []
    for node in nodes:
        precint_id = node['ID']
        color = node['color']
        neighbors = node['neighbors']
        # num_precint_id = int(precint_id)

        # skip nodes belonging to district 0 --- these are bad nodes
        if color == 0:
            bad_nodes.append(precint_id)
            continue

        graph.add_node(precint_id, color=color)
        graph.add_edges_from([(precint_id, neighbor) for neighbor in neighbors])

    # remove bad nodes added implicitly from being a neighbor of a good node
    graph.remove_nodes_from(bad_nodes)

    # remove nodes without neighbors since they cause graph to be non-contiguous
    isolated_nodes = []
    for node in graph.nodes():
        if len(graph[node]) == 0:
            isolated_nodes.append(node)
    graph.remove_nodes_from(isolated_nodes)

    # get number of districts in final graph
    colors = nx.get_node_attributes(graph, 'color')
    num_districts = len(np.unique(list(colors.values())))

    return (graph, num_districts)
