import json
import networkx as nx
import numpy as np

def parse_graph(location):
    with open(location) as f:
        data = json.load(f)

    nodes = data['nodes']
    graph = nx.Graph()
    bad_nodes = []
    for precint_id, value in nodes.items():
        num_precint_id = int(precint_id)

        # skip nodes belonging to district 0 --- these are bad nodes
        if value['district'] == 0:
            bad_nodes.append(num_precint_id)
            continue

        graph.add_node(num_precint_id, color=value['district'])
        graph.add_edges_from([(num_precint_id, neighbor) for neighbor in value['neighbors']])

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
