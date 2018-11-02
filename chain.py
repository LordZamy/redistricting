from itertools import product
import random

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from plotting import plot_graph

class Chain():
    def __init__(self, graph, q, num_colors=4, R=1):
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
    def boundary_connected_components(self, components, graph=None):
        if graph == None:
            graph = self.current_graph
        colors = nx.get_node_attributes(graph, 'color')
        boundary_components = []
        for component in components:
            boundary_nodes = nx.algorithms.boundary.node_boundary(graph, component)
            # get an arbitrary element from component and use it to find the color of the component
            component_node = next(iter(component))
            for node in boundary_nodes:
                if colors[node] != colors[component_node]:
                    boundary_components.append(component)
                    break

        return boundary_components

    def non_adjacent_boundary_connected_components(self, boundary_components):
        while True:
            non_adjacent_components = np.random.choice(boundary_components, self.R)
            redo = False
            for component in non_adjacent_components:
                other_components = [x for x in non_adjacent_components if x != component]
                # check if non-adjacent to currently chosen components
                nodes_of_chosen_components = set().union(*other_components)
                adjacent_nodes = nx.algorithms.boundary.node_boundary(self.current_graph, component, nodes_of_chosen_components)
                if len(adjacent_nodes) is not 0:
                    redo = True
            if not redo:
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

    def count_swendson_wang_cut(self, adjacency_graph, VCP):
        edges = adjacency_graph.edges()
        edge_count = 0
        for u, v in edges:
            for component in VCP:
                if (u in component and v not in component) or (u not in component and v in component):
                    edge_count += 1
                    break

        return edge_count

    """
    If a swap is valid, returns the post-swap graph. Else returns None.
    """
    def verify_swap(self, swapped_colorings):
        values = {node: color for component, color in swapped_colorings for node in component}
        graph = self.current_graph.copy()
        nx.set_node_attributes(graph, values, 'color')
        adjacency_graph = self.create_adjacency_graph(graph)
        colors = nx.get_node_attributes(adjacency_graph, 'color')
        if nx.algorithms.components.number_connected_components(adjacency_graph) == self.num_colors \
            and len(np.unique(list(colors.values()))) == self.num_colors:
            return graph
        return None

    """
    Returns a copy of the given graph with new colors applied to it.
    """
    def update_colors(self, graph, swapped_colorings):
        values = {node: color for component, color in swapped_colorings for node in component}
        new_graph = graph.copy()
        nx.set_node_attributes(new_graph, values, 'color')
        return new_graph

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
        # draw_graph_online(adjacency_graph, pos=lattice_layout(lattice, 50))
        CP = self.connected_components(adjacency_graph)
        # print('CP', len(CP), [len(v) for v in CP])
        BCP = self.boundary_connected_components(CP)
        while True:
            # print('BCP', len(BCP), [len(v) for v in BCP])
            VCP = self.non_adjacent_boundary_connected_components(BCP)
            # print('VCP', len(VCP), [len(v) for v in VCP])
            swapped_colorings = self.swap_color(VCP)
            swapped_graph = self.verify_swap(swapped_colorings)
            if swapped_graph != None:
                break

        # step 5 computations

        # mutated the original adjacency_graph, so creating a new one
        original_adjacency_graph = self.create_adjacency_graph(self.current_graph)
        old_cut_count = self.count_swendson_wang_cut(original_adjacency_graph, VCP)
        swapped_adjacency_graph = self.create_adjacency_graph(swapped_graph)
        swapped_cut_count = self.count_swendson_wang_cut(swapped_adjacency_graph, VCP)
        swapped_components = self.connected_components(swapped_adjacency_graph)
        swapped_boundary_components = self.boundary_connected_components(CP, swapped_graph)
        # print(len(BCP), len(swapped_boundary_components))

        boundary_total = (float(len(BCP)) / len(swapped_boundary_components)) ** self.R
        prob_remove_edge = 1 - self.prob_keep_edge
        adjacent_total = float(prob_remove_edge ** swapped_cut_count) / (prob_remove_edge ** old_cut_count)
        print(swapped_cut_count, old_cut_count)
        accept_prob = min(1, boundary_total * adjacent_total)
        print(accept_prob)

        # print(boundary_total, adjacent_total)
        if random.random() < accept_prob:
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
redistricting_chain = Chain(lattice, 0.25)

# turn on matplotlib interactive mode
plt.ion()

num_iterations = 1000
for i in range(num_iterations):
    print("Performing iteration {}.".format(i + 1))
    redistricting_chain.simulate_step()
    draw_graph_online(redistricting_chain.current_graph, pos=lattice_layout(lattice, 50))

# save figure to disk at the end
plot_graph(redistricting_chain.current_graph, pos=lattice_layout(lattice, 50), num_iterations=num_iterations)
