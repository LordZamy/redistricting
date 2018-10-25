from itertools import product
import random

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class Chain():
    def __init__(self, graph, q, num_colors=4, R=2):
        self.current_graph = graph
        self.prob_keep_edge = q
        self.num_colors = num_colors
        # constant that determines the number of non-adjacent boundary components
        # selected during step 3 of the algorithm
        self.R = R

    """
    Removes edges between color partitions and returns a new adjacency graph.
    """
    def create_adjacency_graph(self, graph):
        adjacency_graph = graph.copy()
        colors = nx.get_node_attributes(graph, 'color')
        edges = adjacency_graph.edges()
        edges_to_remove = []
        for u, v in edges:
            if colors[u] != colors[v]:
                edges_to_remove.append((u, v))
        adjacency_graph.remove_edges_from(edges_to_remove)

        return adjacency_graph

    """
    Mutates a given graph by deleting edges randomly with `1 - prob_keep_edge` chance
    """
    def randomly_remove_edges(self, graph):
        edges = graph.edges()
        edges_to_remove = []
        for u, v in edges:
            if random.random() < 1 - self.prob_keep_edge:
                edges_to_remove.append((u, v))
        graph.remove_edges_from(edges_to_remove)

        return graph

    def connected_components(self, graph):
        return list(nx.algorithms.components.connected_components(graph))

    """
    Assumption: each component in `components` contains vertices of the same color
    """
    def boundary_connected_components(self, components):
        colors = nx.get_node_attributes(self.current_graph, 'color')
        boundary_components = []
        for component in components:
            boundary_nodes = nx.algorithms.boundary.node_boundary(self.current_graph, component)
            # get an arbitrary element from component and use it to find the color of the component
            component_node = next(iter(component))
            for node in boundary_nodes:
                if colors[node] != colors[component_node]:
                    boundary_components.append(component)
                    break

        return boundary_components

    def non_adjacent_boundary_connected_components(self, boundary_components):
        non_adjacent_components = []
        for component in boundary_components:
            if random.random() < 0.5:
                # check if non-adjacent to currently chosen components
                nodes_of_chosen_components = set().union(*non_adjacent_components)
                adjacent_nodes = nx.algorithms.boundary.node_boundary(self.current_graph, component, nodes_of_chosen_components)
                if len(adjacent_nodes) is 0:
                    non_adjacent_components.append(component)

            if len(non_adjacent_components) >= self.R:
                # stop after selecting at most R components
                break

        # TODO: add special case for when no components are chosen
        return non_adjacent_components

    """
    Assigns a new color based on adjacent color partitions of given non adjacent
    boundary connected components.
    """
    def swap_color(self, components):
        swapped_colorings = []
        colors = nx.get_node_attributes(self.current_graph, 'color')
        for component in components:
            adjacent_nodes = nx.algorithms.boundary.node_boundary(self.current_graph, component)
            component_node = next(iter(component))
            adjacent_color_partions = set()
            for node in adjacent_nodes:
                if colors[node] != colors[component_node]:
                    adjacent_color_partions.add(colors[node])

            new_color = np.random.choice(list(adjacent_color_partions))
            swapped_colorings.append((component, new_color))

        return swapped_colorings

    """
    If a swap is valid, returns the post-swap graph. Else returns None.
    """
    def verify_swap(self, swapped_colorings):
        values = {node: color for component, color in swapped_colorings for node in component}
        graph = self.current_graph.copy()
        nx.set_node_attributes(graph, values, 'color')
        adjacency_graph = self.create_adjacency_graph(graph)
        if nx.algorithms.components.number_connected_components(adjacency_graph) == self.num_colors:
            return graph
        return None


    """
    Accepts the swap and updates `current_graph`
    """
    def accept_proposal(self, swapped_colorings):
        values = {node: color for component, color in swapped_colorings for node in component}
        nx.set_node_attributes(self.current_graph, values, 'color')

    def simulate_step(self):
        adjacency_graph = self.create_adjacency_graph(self.current_graph)
        self.randomly_remove_edges(adjacency_graph)
        # draw_graph(adjacency_graph, pos=lattice_layout(adjacency_graph, 50))
        CP = self.connected_components(adjacency_graph)
        # print('CP', len(CP), [len(v) for v in CP])
        while True:
            BCP = self.boundary_connected_components(CP)
            # print('BCP', len(BCP), [len(v) for v in BCP])
            VCP = self.non_adjacent_boundary_connected_components(BCP)
            # print('VCP', len(VCP), [len(v) for v in VCP])
            swapped_colorings = self.swap_color(VCP)
            swapped_graph = self.verify_swap(swapped_colorings)
            if swapped_graph != None:
                break
        # self.accept_proposal(swapped_colorings)
        self.current_graph = swapped_graph

def four_color_lattice(N):
    lattice = nx.generators.lattice.grid_2d_graph(N, N, periodic=True)
    half_N = int(N / 2)
    color_partitions = {
                        1: product(range(half_N), range(half_N)),
                        2: product(range(half_N), range(half_N, N)),
                        3: product(range(half_N, N), range(half_N)),
                        4: product(range(half_N, N), range(half_N, N)),
                        }
    colors = {n: i for i in range(1, 5) for n in list(color_partitions[i])}
    nx.set_node_attributes(lattice, colors, name='color')
    return lattice

def draw_graph(graph, pos=None):
    colors = nx.get_node_attributes(graph, 'color')
    nx.draw_networkx(graph, pos=pos, with_labels=False, node_size=8, node_color=list(colors.values()), cmap='tab10')
    plt.show()

def draw_graph_online(graph, pos=None):
    plt.cla()
    colors = nx.get_node_attributes(graph, 'color')
    nx.draw_networkx(graph, pos=pos, with_labels=False, node_size=8, node_color=list(colors.values()), cmap='tab10')
    plt.pause(0.0001)

def lattice_layout(graph, N):
    nodes = graph.nodes()
    return {(u, v): (u / float(N), v / float(N)) for u, v in nodes}

lattice = four_color_lattice(50)
redistricting_chain = Chain(lattice, 0.7)

# turn on matplotlib interactive mode
plt.ion()

for i in range(100):
    print("Performing iteration {}.".format(i + 1))
    redistricting_chain.simulate_step()
    draw_graph_online(redistricting_chain.current_graph, pos=lattice_layout(lattice, 50))
