import json
import numpy as np
from graph_tool import Graph, VertexPropertyMap
from graph_tool.draw import graph_draw

def parse_graph(location):
    with open(location) as f:
        data = json.load(f)

    nodes = data['nodes']
    graph = Graph()
    bad_nodes = []
    colors = set()
    # map from vertex -> precint_id
    vertex_map = {}
    # define vertex property maps
    vprop_precinct_id = graph.new_vertex_property('int32_t')
    vprop_color = graph.new_vertex_property('int16_t')
    vprop_pos = graph.new_vertex_property('vector<float>')
    # vprop_county_id = graph.new_vertex_property('int32_t')
    for node in nodes:
        precinct_id = node['ID']
        color = node['color']
        vertex = graph.add_vertex()
        vertex_map[precinct_id] = vertex

        # skip nodes belonging to district 0 --- these are bad nodes
        if color == 0:
            bad_nodes.append(precinct_id)

    for node in nodes:
        precinct_id = node['ID']
        color = node['color']
        x = node['xCoord']
        y = node['yCoord']
        neighbors = {int(neighbor): weight for neighbor, weight in node['neighbors'].items()}
        county_id = node.get('countyID')
        if county_id == None:
            print('No county id for precinct {}'.format(precinct_id))

        # get vertex from map
        vertex = vertex_map[precinct_id]

        # set vertex prpoerties
        vprop_precinct_id[vertex] = precinct_id
        vprop_color[vertex] = color
        vprop_pos[vertex] = [x, y]
        # vprop_county_id[vertex] = county_id

        # add edges
        graph.add_edge_list([(vertex, vertex_map[neighbor]) for neighbor, weight in neighbors.items()])

    # set vertex properties internal to graph
    graph.vp['precint_id'] = vprop_precinct_id
    graph.vp['color'] = vprop_color
    graph.vp['pos'] = vprop_pos
    # graph.vp['county_id'] = vprop_county_id

    # remove the bad nodes
    for node in bad_nodes:
        # remove the vertex properties
        for vprop in graph.vp:
            del vprop[node]

        # remove the vertex
        graph.remove_vertex(vertex_map[node])

    # remove nodes without neighbors since they cause graph to be non-contiguous
    all_degrees = graph.get_total_degrees(graph.get_vertices())
    isolated_vertices = [graph.vertex(i) for i, item in enumerate(all_degrees) if item == 0]
    for vertex in isolated_vertices:
        for vprop in graph.vp:
            del vprop[node]

        graph.remove_vertex(vertex)

    num_districts = len(np.unique(vprop_color.get_array()))
    return graph, num_districts

# g = parse_graph('./data/GeorgiaGraph.json')
# graph_draw(g, g.vp.pos)
