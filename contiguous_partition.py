import networkx as nx
from networkx.generators.lattice import grid_2d_graph
from networkx.classes.graphviews import subgraph_view
from networkx.classes.function import number_of_nodes
from networkx.algorithms.components import number_connected_components
from networkx.algorithms.boundary import node_boundary
import numpy as np


class Partitioner():
    """
    Assumption: the graph passed in must be connected.
    """
    
    def __init__(self, graph, num_colors=2):
        self.graph = graph
        self.num_colors = num_colors

        # statistics
        self.num_swaps = 0
        self.num_steps = 0
        self.num_effective_steps = 0
        self.minima_at_step = None
    
    @staticmethod
    def get_adjacency_graph(graph, colors):
        filter_edge = lambda u, v: colors[u] == colors[v]
        return subgraph_view(graph, filter_edge=filter_edge)
    
    @staticmethod
    def get_adjacency_graph_with_color_fn(graph, color_fn):
        filter_edge = lambda u, v: color_fn(u) == color_fn(v)
        return subgraph_view(graph, filter_edge=filter_edge)

    def step(self):
        """
        Each step of the algorithm, we first pick a vertex uniformly at random.
        Then we propose a change to the color of this vertex. We pick this new
        color from one of its neighbors. We accept this change only if the new
        graph has <= number of connected components than the old graph. 
        """
        self.num_steps += 1

        colors = nx.get_node_attributes(self.graph, 'color')
        adjacency_graph = Partitioner.get_adjacency_graph(self.graph, colors)

        # select random vertex
        rand_index = np.random.choice(number_of_nodes(adjacency_graph))
        rand_vertex = list(adjacency_graph.nodes())[rand_index]

        # select random color from neighbors of vertex
        neighbors = self.graph.neighbors(rand_vertex)
        neighbor_colors = {colors[node] for node in neighbors}
        rand_color = np.random.choice(list(neighbor_colors))

        # if we pick the same vertex, then we're done
        if rand_color == colors[rand_vertex]:
            return

        # accessor function that returns the randomly chosen color for the
        # randomly chosen vertex and original colors for every other vertex
        color_fn = lambda v: rand_color if v == rand_vertex else colors[v]

        swapped_adjacency_graph = Partitioner.get_adjacency_graph_with_color_fn(self.graph, color_fn)

        # get number of connected components in old and new graphs
        old_num_components = number_connected_components(adjacency_graph)
        swapped_num_components = number_connected_components(swapped_adjacency_graph)

        print(old_num_components, swapped_num_components)

        # this method should not do anything if we have reached our goal
        if old_num_components == self.num_colors:
            if self.minima_at_step == None:
                # record the point at which we found this minima
                self.minima_at_step = self.num_steps

            print("Won't swap. Only {} components left in graph.".format(old_num_components))
            return

        # we wish to update the color of the vertex only if we reduce the
        # number of components
        if swapped_num_components <= old_num_components:
            colors[rand_vertex] = rand_color
            nx.set_node_attributes(self.graph, colors, name='color')
            self.num_swaps += 1

        self.num_effective_steps += 1

class TreePartitioner():
    def __init__(self, graph, num_colors=2):
        """
        Assumes a connected undirected graph.
        """
        self.graph = graph
        self.num_colors = num_colors
        self.init_components()

        # statistics
        self.num_swaps = 0
        self.num_steps = 0
        self.num_effective_steps = 0
        self.minima_at_step = None
    
    def init_components(self):
        # Each component is a set of vertices. We initialize the each set with
        # a randomly chosen vertex.
        rand_indices = np.random.choice(number_of_nodes(self.graph), size=self.num_colors, replace=False)
        list_of_nodes = list(self.graph.nodes())
        rand_vertices = [list_of_nodes[i] for i in rand_indices]
        self.components = {color: {rand_vertices[color]} for color in range(self.num_colors)}
        # TODO: don't forget to color initial vertices

    def step(self):
        """
        Returns true if some update was made. False if could not update.
        """
        colors = nx.get_node_attributes(self.graph, 'color')

        updated_color_map = {}
        for color, component in self.components.items():
            # get vertices on the boundary of the current component
            boundary = node_boundary(self.graph, component)

            # filter out vertices that have been colored in the last iteration
            # or in this iteration (stored in the updated_color_map)
            uncolored_boundary_vertices = filter(lambda v: colors[v] == -1 and v not in updated_color_map, boundary)

            # if there are no uncolored vertices left, move on to next component
            list_uncolored_vertices = list(uncolored_boundary_vertices)
            print(list_uncolored_vertices)
            if len(list_uncolored_vertices ) == 0:
                continue

            rand_boundary_vertex_index = np.random.choice(len(list_uncolored_vertices));
            rand_boundary_vertex = list_uncolored_vertices[rand_boundary_vertex_index]
            component.add(rand_boundary_vertex)
            # put new color in the update map
            updated_color_map[rand_boundary_vertex] = color
            
        # print(updated_color_map)
        # return false if no updates were made
        if len(updated_color_map) == 0:
            return False
        
        # update colors of vertices in graph attributes
        for vertex, color in updated_color_map.items():
            colors[vertex] = color
        
        nx.set_node_attributes(self.graph, colors, 'color')
        return True

    def run(self):
        num_iterations = 0
        while self.step():
            num_iterations += 1
            print("On iteration {}".format(num_iterations))
        
def random_lattice(N=10, colors=2, periodic=False):
    lattice = grid_2d_graph(N, N, periodic=periodic)
    # assign random color to each vertex of lattice
    colors = {node: np.random.choice(colors) for node in lattice.nodes()}
    nx.set_node_attributes(lattice, colors, name='color')
    return lattice

def uncolored_lattice(N=10, colors=2, periodic=False):
    lattice = grid_2d_graph(N, N, periodic=periodic)
    colors = {node: -1 for node in lattice.nodes()}
    nx.set_node_attributes(lattice, colors, name='color')
    return lattice