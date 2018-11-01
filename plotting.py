import time
import networkx as nx
import matplotlib.pyplot as plt

OUTPUT_DIR = 'output'

def save_plot(plt):
    plt.savefig('{}/{}.png'.format(OUTPUT_DIR, time.time()), bbox_inches='tight', dpi=300)

def plot_graph(graph, pos=None, num_iterations=10):
    plt.title('T: {}'.format(num_iterations))
    colors = nx.get_node_attributes(graph, 'color')
    nx.draw_networkx(graph, pos=pos, with_labels=False, node_size=8, node_color=list(colors.values()), cmap='tab10')
    save_plot(plt)
