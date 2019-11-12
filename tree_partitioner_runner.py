import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from contiguous_partition import TreePartitioner, uncolored_lattice

def draw_graph(graph, pos=None, node_size=25, cmap='tab10'):
    colors = nx.get_node_attributes(graph, 'color')
    nx.draw_networkx(graph, pos=pos, with_labels=False, node_size=node_size, node_color=list(colors.values()), cmap=cmap)
    plt.show()

def lattice_layout(graph, N):
    nodes = graph.nodes()
    return {(u, v): (u / float(N), v / float(N)) for u, v in nodes}

num_colors = 8

lattice = uncolored_lattice(N=50, colors=num_colors)
partitioner = TreePartitioner(lattice, num_colors)


# print("Ran for {} timesteps".format(T))
# print("Effective steps: {}".format(partitioner.num_effective_steps))
# print("Number of swaps: {}".format(partitioner.num_swaps))
# if partitioner.minima_at_step == None:
#     print("Minima not found :(")
# else:
#     print("Minima found at step {}".format(partitioner.minima_at_step))

partitioner.run()

colors = nx.get_node_attributes(partitioner.graph, 'color')
print({color for color in colors.values()})

draw_graph(partitioner.graph, pos=lattice_layout(partitioner.graph, 10), cmap='tab10')

