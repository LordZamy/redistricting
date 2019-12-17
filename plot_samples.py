from json_parser import parse_graph
import matplotlib.pyplot as plt
import networkx as nx
import json
import sys

plt.rcParams["figure.figsize"] = (8, 7)

def draw_graph(graph, color_map, i, partition_edges=True, nodelist=None):
    if nodelist == None:
        nodelist = graph.nodes()

    if partition_edges:
        edgelist = graph.edges()
    else:
        edgelist = []
        for u, v in graph.edges():
            if color_map[str(u)] == color_map[str(v)]:
                edgelist.append((u, v))

    node_color = [color_map[str(node)] for node in nodelist]
    pos = nx.get_node_attributes(graph, 'pos')
    nx.draw_networkx(graph, nodelist=nodelist, edgelist=edgelist, pos=pos, with_labels=False, node_size=15, node_color=node_color, cmap='tab20')
    plt.axis('off')
    plt.savefig('results/graphs/' + str(i) + '.png', dpi=200, bbox_inches='tight')
    plt.clf()

def plotter(graph, data):
    color_list = data['color_list']
    i = 0
    for color_map in color_list: 
        if i == 100:
            break
        draw_graph(graph, color_map, i, partition_edges=False)
        i += 1

if len(sys.argv) > 1:
    location = sys.argv[1]
    graph, num_districts = parse_graph('data/GeorgiaGraph.json')
    with open(location) as f:
        data = json.load(f)
    plotter(graph, data)
else:
    print("Need json as input!")
