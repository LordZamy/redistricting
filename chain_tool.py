from graph_tool import Graph, GraphView, VertexPropertyMap
from graph_tool.topology import label_components
from numba import jit
import numpy as np
import random

from typing import Optional, Set
from collections import defaultdict

class MarkovChain():
    def __init__(self, graph: Graph, q: float, num_colors: int, R: int, color_map: Optional[VertexPropertyMap] = None, pos: Optional[VertexPropertyMap] = None):
        self.current_graph = graph.copy()
        self.current_graph.set_fast_edge_removal(True)
        self.eprop_filter = self.current_graph.new_edge_property('bool')

        self.prob_keep_edge = q
        self.num_colors = num_colors
        self.R = R

        if color_map != None:
            self.color_map = color_map
        else:
            self.color_map = self.current_graph.vp['color']

        if pos != None:
            self.pos = pos
        else:
            self.pos = self.current_graph.vp['pos']

    def func(u, v):
        return self.color_map[u] == self.color_map[v]

    """
    Returns a modified copy of the given graph with all edges between colors removed.
    """
    def create_adjacency_graph(self, graph: Graph) -> GraphView:
        # adjacency_graph = graph.copy()
        # assert adjacency_graph.get_fast_edge_removal() == True
        assert graph.get_fast_edge_removal() == True

        # create new edge filter that disables edges
        eprop_filter = graph.new_edge_property('bool', val=True)

        # print(eprop_filter.a)
        func = lambda x: self.color_map[x[0]] == self.color_map[x[1]]
        eprop_filter.a = np.apply_along_axis(func, 1, graph.get_edges())
        # eprop_filter.a = self.remove_dichromatic_edges(graph)

        # print(eprop_filter.a)
        # for e in graph.edges():
        #     if self.color_map[e.source()] != self.color_map[e.target()]:
        #         eprop_filter[e] = False

        return GraphView(graph, efilt=eprop_filter)

    """
    Mutates a given graph by deleting edges randomly with `1 - prob_keep_edge` chance.
    """
    def randomly_remove_edges(self, graph: Graph) -> GraphView:
        rand_arr = np.random.rand(graph.num_edges()) < self.prob_keep_edge
        eprop_filter = graph.new_edge_property('bool', vals=rand_arr)

        # for e in graph.edges():
        #     if random.random() < (1 - self.prob_keep_edge):
        #         eprop_filter[e] = False

        return GraphView(graph, efilt=eprop_filter)

    """
    component_labels should correspond to the adjacency graph with edges removed
    """
    def boundary_connected_components(self, graph: Graph, component_labels: VertexPropertyMap):
        boundary_components = set()
        for v in graph.vertices():
            # optimize by not considering already adding component labels?
            if component_labels[v] in boundary_components:
                continue

            for neighbor in self.current_graph.get_all_neighbors(v):
                if self.color_map[v] != self.color_map[neighbor]:
                    boundary_components.add(component_labels[v])
                    break

        assert len(boundary_components) != 0
        return boundary_components

    def non_adjacent_boundary_connected_components(self, boundary_components: Set, component_labels: VertexPropertyMap):
        while True:
            non_adjacent_component_labels = set(np.random.choice(list(boundary_components), self.R))
            boundary_component_vertices = filter(lambda x: component_labels[x] in non_adjacent_component_labels, self.current_graph.vertices())
            redo = False
            # non_adjacent_component_label -> set of neighbors mapping
            neighbors = defaultdict(set)
            # non_adjacent_component_label -> color
            component_colors = {}
            # non_adjacent_component_label -> color
            neighbor_colors = defaultdict(set)
            for v in boundary_component_vertices:
                n = self.current_graph.get_all_neighbors(v, vprops=[self.color_map])
                label = component_labels[v]
                neighbors[label].update(n[:, 0])
                component_colors[label] = self.color_map[v]
                neighbor_colors[label].update(n[:, 1])

            # check if selecting components are non-adjacent i.e. their node
            # boundaries are disjoint
            neighbor_values = neighbors.values()
            num_neighbors = sum(map(len, neighbor_values))
            all_neighbors = set().union(*neighbor_values)
            if len(all_neighbors) != num_neighbors:
                # node boundaries are not disjoint --- sample again
                redo = True

            if not redo:
                break

        return non_adjacent_component_labels, boundary_component_vertices, neighbors, component_colors, neighbor_colors

    # neighbors should be a dict of non_adjacent_component -> set of neighbors
    # return component_label -> color mapping
    def swap_colors(self, component_colors, neighbor_colors):
        # component_label -> color
        swapped_colorings = {}

        for component_label, colors in neighbor_colors.items():
            # modifying colors here!
            colors.remove(component_colors[component_label])
            new_color = np.random.choice(list(colors))
            swapped_colorings[component_label] = new_color

        return swapped_colorings

    # updates the color vertex property map
    def perform_swap(self, VCP, component_labels, swapped_colorings):
        new_colors = self.color_map.copy()
        for vertex in VCP:
            new_colors[vertex] = swapped_colorings[component_labels[vertex]]

        self.current_graph.vp['color'].swap(new_colors)

    def simulate_step(self, beta):
        # remove edges from graph
        adjacency_graph = self.create_adjacency_graph(self.current_graph)
        adjacency_graph = self.randomly_remove_edges(adjacency_graph)

        # find connected components
        component_labels, _ = label_components(adjacency_graph)
        BCP = self.boundary_connected_components(adjacency_graph, component_labels)
        while True:
            non_adjacent_component_labels, VCP, neighbors, component_colors, neighbor_colors = self.non_adjacent_boundary_connected_components(BCP, component_labels)
            swapped_colorings = self.swap_colors(component_colors, neighbor_colors)
            self.perform_swap(VCP, component_labels, swapped_colorings)
            break
