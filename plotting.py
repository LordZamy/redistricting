import time
import networkx as nx
import matplotlib.pyplot as plt

OUTPUT_DIR = 'output'

def save_plot(plt):
    plt.savefig('{}/{}.png'.format(OUTPUT_DIR, time.time()), bbox_inches='tight', dpi=300)

def plot_graph(graph, pos=None, num_iterations=10, node_size=8, cmap='tab10', q=None, num_colors=None, R=None, beta=None, within_county_weight=None):
    title = ''
    if num_iterations != None:
        title += 'T: {}'.format(num_iterations) + ', '
    if q != None:
        title += 'q: {}'.format(q) + ', '
    if num_colors != None:
        title += 'colors: {}'.format(num_colors) + ', '
    if R != None:
        title += 'R: {}'.format(R) + ', '
    if beta != None:
        title += 'beta: {}'.format(beta) + ', '
    if within_county_weight != None:
        title += 'within_weight: {}'.format(within_county_weight) + ', '

    plt.title(title)
    colors = nx.get_node_attributes(graph, 'color')
    nx.draw_networkx(graph, pos=pos, with_labels=False, node_size=node_size, node_color=list(colors.values()), cmap=cmap)
    save_plot(plt)
