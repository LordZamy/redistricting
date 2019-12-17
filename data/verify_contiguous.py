import json
import sys
import networkx as nx
import matplotlib.pyplot as plt

def create_graph(location):
    with open(location) as f:
        data = json.load(f)

    graph = nx.Graph()
    nodes = data['nodes']

    for node in nodes:
        node_id = node["ID"]
        color = node['color']
        neighbors = node['neighbors']
        x = node['xCoord']
        y = node['yCoord']

        graph.add_node(node_id, color=color, pos=(x, y));
        graph.add_edges_from([(node_id, neighbor) for neighbor in neighbors])

    return graph

def verify_contiguous(graph, num_components=14):
    # first remove edges across partitions
    colors = nx.get_node_attributes(graph, 'color')
    edges_to_remove = []
    for u, v in graph.edges():
        if colors[u] != colors[v]:
            edges_to_remove.append((u, v))
    graph.remove_edges_from(edges_to_remove)

    n = nx.algorithms.components.number_connected_components(graph)
    print(n)
    if n != num_components:
        print('Graph is flawed!')

    return graph


def draw_graph(graph, nodelist=None):
    if nodelist == None:
        nodelist = graph.nodes()

    plt.rcParams["figure.figsize"] = (8, 7)
    colors = nx.get_node_attributes(graph, 'color')
    node_color = [colors[node] for node in nodelist]
    pos = nx.get_node_attributes(graph, 'pos')
    nx.draw_networkx(graph, nodelist=nodelist, pos=pos, with_labels=False, node_size=15, node_color=node_color, cmap='tab20')
    plt.show()

def draw_components(graph):
    connected_components = nx.algorithms.components.connected_components(graph)
    for component in connected_components:
        draw_graph(graph, nodelist=list(component))


if len(sys.argv) > 1:
    location = sys.argv[1]
    graph = create_graph(location)
    graph = verify_contiguous(graph)
    draw_graph(graph)
    draw_components(graph)
else:
    print("Need json as input!")
