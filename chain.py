from itertools import product, combinations, starmap
import random
import math

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from plotting import plot_graph
from json_parser import parse_graph
from shape_parser import parse_shape, parse_pos

# change figure size
plt.rcParams["figure.figsize"] = (8, 7)

lattice_population_parity = 625

class Chain():
    def __init__(self, graph, q, num_colors=4, R=1):
        self.current_graph = graph
        self.prob_keep_edge = q
        self.num_colors = num_colors
        # constant that determines the number of non-adjacent boundary components
        # selected during step 3 of the algorithm
        self.R = R

        # confirm initial graph is correct
        adjacency_graph = self.create_adjacency_graph(graph)
        # the computed compactness score per partition
        # self.compactness = compute_initial_compactness(self.connected_components(adjacency_graph))
        self.compactness = self.compute_compact_constraint(self.connected_components(adjacency_graph))
        print(self.compactness)
        # draw_graph(adjacency_graph, georgia_layout, cmap='tab20')
        colors = nx.get_node_attributes(graph, 'color')
        unique_colors = len(np.unique(list(colors.values())))
        num_components = nx.algorithms.components.number_connected_components(adjacency_graph)
        print([len(c) for c in self.connected_components(adjacency_graph)])
        try:
            assert num_components == num_colors \
                and unique_colors == num_colors, \
                'Partitions of graph are not contiguous! Components: {}, Unique colors: {}, Num colors: {}'.format(num_components, unique_colors, num_colors)
        except AssertionError:
            # this is a shitty solution which works for the georgia graph
            components = self.connected_components(adjacency_graph)
            lengths = [len(c) for c in components]
            min_index = np.argmin(lengths)
            component_to_drop = components[min_index]
            self.current_graph.remove_nodes_from(component_to_drop)

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
            adjacent_color_partitions = set()
            for node in adjacent_nodes:
                if colors[node] != colors[component_node]:
                    adjacent_color_partitions.add(colors[node])

            new_color = np.random.choice(list(adjacent_color_partitions))
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

    def compute_pop_constraint(self, partitions):
        global lattice_population_parity
        return sum([abs(len(p) / float(lattice_population_parity) - 1) for p in partitions])

    """
    Currently letting e_i and e_j equal to 1.
    """
    def compute_compact_constraint(self, partitions):
        partition_pairs = [combinations(partition, 2) for partition in partitions]
        partition_distances = [Chain.euclidean_distance_square(*pair) for partition in partition_pairs for pair in partition]
        return sum(partition_distances)

    def compute_initial_compactness(self, partitions):
        partition_pairs = [combinations(partition, 2) for partition in partitions]
        partition_distances = [starmap(Chain.euclidean_distance_square, partition) for partition in partition_pairs]
        return [sum(partition) for partition in partition_distances]

    def update_compactness(self, old_graph, old_partitions, new_graph, new_partitions, swapped_colorings):
        op = Chain.map_partitions_to_color(old_graph, old_partitions)
        np = Chain.map_partitions_to_color(new_graph, new_partitions)
        osc = Chain.map_swapped_colorings_to_old_colors(old_graph, swapped_colorings)
        nsc = Chain.map_swapped_colorings_to_color(swapped_colorings)

        # placeholder for cached compactness score
        compactness = self.compactness
        flat_swap = [component for component, _ in swapped_colorings]
        # print(flat_swap)
        print(osc)
        print(nsc)
        # removing vertices
        for color, components in osc.items():
            monotone_components = components + [op[color].difference(*flat_swap)]
            print(len(op[color].difference(*flat_swap)))
            print([len(c) for c in components])
            # print(len(monotone_components))
            monotone_component_pairs = combinations(monotone_components, 2)
            c = 0
            for pair in monotone_component_pairs:
                c += 1
                compactness -= Chain.inter_component_distance(*pair)
            print('removal length:', c)

        # adding vertices
        for color, component in nsc.items():
            monotone_components = components + [np[color].difference(*flat_swap)]
            print(len(np[color].difference(*flat_swap)))
            print([len(c) for c in components])
            monotone_component_pairs = combinations(monotone_components, 2)
            c = 0
            for pair in monotone_component_pairs:
                c += 1
                compactness += Chain.inter_component_distance(*pair)
            print('addition length:', c)

        return compactness


    """
    Returns dict of form {color: partition}
    """
    def map_partitions_to_color(graph, partitions):
        colors = nx.get_node_attributes(graph, 'color')
        color_partitions = {}
        for partition in partitions:
            node = next(iter(partition))
            color_partitions[colors[node]] = partition
            print(colors[node], len(partition))
        return color_partitions

    def map_swapped_colorings_to_color(swapped_colorings):
        colormap = {}
        for component, color in swapped_colorings:
            if color not in colormap:
                colormap[color] = [component]
            else:
                colormap[color].append(component)
        return colormap

    def map_swapped_colorings_to_old_colors(old_graph, swapped_colorings):
        old_colors = nx.get_node_attributes(old_graph, 'color')
        colormap = {}
        for component, color in swapped_colorings:
            node = next(iter(component))
            old_color = old_colors[node]
            if old_color not in colormap:
                colormap[old_color] = [component]
            else:
                colormap[old_color].append(component)
        return colormap

    """
    Sums the square distance between every pair of nodes u, v s.t. u belongs
    to first and v belongs to second.
    """
    def inter_component_distance(first, second):
        pairs = list(product(first, second))
        print(len(pairs))
        return sum(starmap(Chain.euclidean_distance_square, pairs))

    def euclidean_distance_square(u, v):
        dx = u[0] - v[0]
        dy = u[1] - v[1]
        return dx * dx + dy * dy

    def compute_compact_constraint_internal(self, partitions):
        total = 0
        for partition in partitions:
            pairs = combinations(partition, 2)
            # partition_graph = nx.Graph()
            partition_graph = self.current_graph.subgraph(partition)
            print(partition_graph.number_of_nodes(), partition_graph.number_of_edges())
            # partition_graph.add_nodes_from(partition)
            distance = dict(nx.algorithms.shortest_paths.unweighted.all_pairs_shortest_path_length(partition_graph))
            print('computed')
            for pair in pairs:
                total += distance[pair[0]][pair[1]]

        return total

    def simulate_step(self):
        adjacency_graph = self.create_adjacency_graph(self.current_graph)
        self.randomly_remove_edges(adjacency_graph)
        # draw_graph(adjacency_graph, pos=lattice_layout(adjacency_graph, 50))
        # draw_graph_online(adjacency_graph, pos=lattice_layout(lattice, 50))
        # global georgia_layout
        # draw_graph_online(adjacency_graph, pos=georgia_layout)
        CP = self.connected_components(adjacency_graph)
        # print('CP', len(CP), [len(v) for v in CP])
        BCP = self.boundary_connected_components(CP)
        while True:
            # print('BCP', len(BCP), [len(v) for v in BCP])
            VCP = self.non_adjacent_boundary_connected_components(BCP)
            # print('VCP', len(VCP), [len(v) for v in VCP])
            swapped_colorings = self.swap_color(VCP)
            swapped_graph = self.verify_swap(swapped_colorings)
            # print(swapped_graph)
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
        print([len(c) for c in self.connected_components(original_adjacency_graph)])

        boundary_total = (float(len(BCP)) / len(swapped_boundary_components)) ** self.R
        prob_remove_edge = 1 - self.prob_keep_edge
        adjacent_total = float(prob_remove_edge ** swapped_cut_count) / (prob_remove_edge ** old_cut_count)
        print(swapped_cut_count, old_cut_count)
        accept_prob = min(1, boundary_total * adjacent_total)
        # print(accept_prob)

        # original_pop_constraint = self.compute_pop_constraint(self.connected_components(original_adjacency_graph))
        # swapped_pop_constraint = self.compute_pop_constraint(swapped_components)
        #
        # pop_accept_prob = min(1, math.exp(-0.4 * (swapped_pop_constraint - original_pop_constraint)) * boundary_total * adjacent_total)

        # original_compact_constraint = self.compute_compact_constraint(self.connected_components(original_adjacency_graph))
        original_compact_constraint = self.compactness
        # swapped_compact_constraint = self.compute_compact_constraint(swapped_components)
        swapped_compact_constraint = self.update_compactness(self.current_graph, self.connected_components(original_adjacency_graph), swapped_graph, swapped_components, swapped_colorings)
        # print(swapped_compact_constraint - original_compact_constraint, swapped_compact_constraint, original_compact_constraint)
        actual_swapped_compactness = self.compute_compact_constraint(swapped_components)
        print(swapped_compact_constraint, actual_swapped_compactness)
        scale = 0.00001
        compact_accept_prob = min(1, math.exp(-0.9 * scale * (swapped_compact_constraint - original_compact_constraint)) * boundary_total * adjacent_total)

        # print(boundary_total, adjacent_total)
        # if random.random() < accept_prob:
        #     self.current_graph = swapped_graph

        # print(pop_accept_prob)
        # if random.random() < pop_accept_prob:
        #     self.current_graph = swapped_graph

        print(compact_accept_prob)
        if random.random() < compact_accept_prob:
            # move chain to new graph
            self.current_graph = swapped_graph
            # update compactness score
            self.compactness = swapped_compact_constraint

def four_color_lattice(N, periodic=True):
    lattice = nx.generators.lattice.grid_2d_graph(N, N, periodic=periodic)
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

def bad_two_color_lattice(N):
    lattice = lattice = nx.generators.lattice.grid_2d_graph(N, N, periodic=False)
    colors = {}
    for i in range(N):
        for j in range(N):
            if j < N * 2 / 10:
                colors[(i, j)] = 1
            elif j > N * 8 / 10:
                colors[(i, j)] = 2
            elif i * 10 / N % 2 == 0:
                colors[(i, j)] = 1
            else:
                colors[(i, j)] = 2

    nx.set_node_attributes(lattice, colors, name='color')
    return lattice

def bad_uniform_two_color_lattice(N):
    assert N % 5 == 0
    lattice = lattice = nx.generators.lattice.grid_2d_graph(N, N, periodic=False)
    colors = {}
    for i in range(N):
        for j in range(N):
            if j % 10 < 5:
                if i == N - 1:
                    colors[(i, j)] = 2
                else:
                    colors[(i, j)] = 1
            else:
                if i == 0:
                    colors[(i, j)] = 1
                else:
                    colors[(i, j)] = 2

    nx.set_node_attributes(lattice, colors, name='color')
    return lattice

def concentric_circle_graph():
    graph = nx.Graph()
    nodes = []
    for r in range(1, 5):
        angles = np.linspace(0, 2 * np.pi, 20)
        nodes += [polar_to_cartesian(angle, r) for angle in angles]
    graph.add_nodes_from(nodes)
    colors = {node: 1 for node in nodes}
    nx.set_node_attributes(graph, colors, name='color')
    return graph

def polar_to_cartesian(theta, r):
    return (math.cos(theta) * r, math.sin(theta) * r)


def draw_graph(graph, pos=None, node_size=8, cmap='tab10'):
    colors = nx.get_node_attributes(graph, 'color')
    nx.draw_networkx(graph, pos=pos, with_labels=False, node_size=node_size, node_color=list(colors.values()), cmap=cmap)
    plt.show()

def draw_graph_online(graph, pos=None, node_size=8, cmap='tab10'):
    plt.cla()
    colors = nx.get_node_attributes(graph, 'color')
    nx.draw_networkx(graph, pos=pos, with_labels=False, node_size=node_size, node_color=list(colors.values()), cmap=cmap)
    plt.pause(0.0001)

def lattice_layout(graph, N):
    nodes = graph.nodes()
    return {(u, v): (u / float(N), v / float(N)) for u, v in nodes}

def concentric_circle_layout(graph):
    nodes = graph.nodes()
    return {(u, v): (u / 5.0, v / 5.0) for u, v in nodes}

# lattice = four_color_lattice(50)
# for testing compactness constraint
# lattice = four_color_lattice(50, periodic=False)
lattice = bad_two_color_lattice(50)
# lattice = bad_uniform_two_color_lattice(30)
circle = concentric_circle_graph()

georgia_graph, num_districts = parse_graph('data/GeorgiaGraph.json')
# georgia_layout = parse_shape('data/ga_2016/ga_2016.shp', georgia_graph)
georgia_layout = parse_pos()
georgia_cmap = cm.get_cmap('hsv', 14)
# draw_graph(georgia_graph, georgia_layout, cmap=georgia_cmap)
# plot_graph(georgia_graph, pos=georgia_layout, num_iterations=0, cmap=georgia_cmap)
# draw_graph(lattice, lattice_layout(lattice, 50), cmap='tab10')
# draw_graph(lattice, lattice_layout(lattice, 30), cmap='winter')
draw_graph(circle, concentric_circle_layout(circle), cmap='winter')

# redistricting_chain = Chain(lattice, 0.07, R=2)
# redistricting_chain = Chain(georgia_graph, 0.07, num_colors=num_districts, R=5)
redistricting_chain = Chain(lattice, 0.07, num_colors=2, R=2)

# turn on matplotlib interactive mode
plt.ion()

num_iterations = 5000
for i in range(num_iterations):
    print("Performing iteration {}.".format(i + 1))
    redistricting_chain.simulate_step()
    # draw_graph_online(redistricting_chain.current_graph, pos=lattice_layout(lattice, 50))
    # draw_graph_online(redistricting_chain.current_graph, pos=georgia_layout, cmap=georgia_cmap)

# save figure to disk at the end
# plot_graph(redistricting_chain.current_graph, pos=lattice_layout(lattice, 50), num_iterations=num_iterations, node_size=20, cmap='winter')
# plot_graph(redistricting_chain.current_graph, pos=georgia_layout, num_iterations=num_iterations, cmap=georgia_cmap)
plot_graph(redistricting_chain.current_graph, pos=lattice_layout(lattice, 50), num_iterations=num_iterations, node_size=20, cmap='winter')
