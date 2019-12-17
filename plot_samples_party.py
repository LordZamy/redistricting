from json_parser import parse_graph
import matplotlib.pyplot as plt
import networkx as nx
import json
import sys
import pickle
from vote_counter import count_votes_per_party, count_precinct_level_votes_per_party
import geopandas as gpd
from color_utils import color_list_to_matched_precincts
from match_precincts import PrecinctMatcher
import matching
import dill

plt.rcParams["figure.figsize"] = (8, 7)

def get_precinct_matcher():
    with open('data/precinct_matcher.json', 'r') as f:
        return json.load(f)

# get match in election result
def clean_cnty_name(name):
    return name.lower() + ' county'

def clean_precinct_name(name):
    return name.lower()

# load ga metadata
with open('data/GeorgiaMetadata.json', 'r') as read_file:
    ga_metadata = json.load(read_file)

GA_precincts = gpd.read_file('data/GA_precincts16/GA_precincts16.shp')

# load precinct_matcher
precinct_matcher = get_precinct_matcher()

# bridges between our metadata and MGGG shapefile to perform match into election data
GA_precincts_id = GA_precincts.set_index(keys='ID')
def matching_fn(synthetic_id):
    if ga_metadata[synthetic_id]['PRECINCT_N'] == None or ga_metadata[synthetic_id]['PRECINCT_I'] == None:
        return None
    
    precinct_id = int(ga_metadata[synthetic_id]['ID'])
    shape_precinct_name = GA_precincts_id.loc[precinct_id]['PRECINCT_N']
    shape_county_name = GA_precincts_id.loc[precinct_id]['CTYNAME']
    if shape_precinct_name != None and shape_county_name != None:
        cl_precinct_name = clean_precinct_name(shape_precinct_name)
        cl_county_name = clean_cnty_name(shape_county_name)
        # matched_precinct_name = precinct_matcher.get_match(cl_precinct_name, cl_county_name)
        matched_precinct_name = precinct_matcher[cl_county_name][cl_precinct_name]
        assert isinstance(matched_precinct_name, str)
        return (matched_precinct_name, cl_county_name)
    else:
        print(shape_precinct_name, shape_county_name)
        return None

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

    # node_color = [color_map[str(node)] for node in nodelist]
    pos = nx.get_node_attributes(graph, 'pos')

    color_list = color_list_to_matched_precincts(color_map, matching_fn)
    color_votes = count_votes_per_party(vote_map, color_list, candidate_party_map)
    color_votes_r_intensity = {}
    for color, party_and_votes in color_votes.items():
        r_votes = party_and_votes['republican']
        d_votes = party_and_votes['democratic']
        intensity = float(r_votes) / (r_votes + d_votes)
        print(intensity)
        color_votes_r_intensity[color] = intensity
    # now color_votes_r_intensity is {color: [0, 1]}

    node_color = []
    for v in graph.nodes():
        # want to map each v to a color gradient (blue/red) based on party votes
        color = color_map[str(v)]
        intensity = color_votes_r_intensity[color]
        node_color.append(intensity)

    nx.draw_networkx(graph, nodelist=nodelist, edgelist=edgelist, pos=pos, with_labels=False, node_size=15, node_color=node_color, cmap='seismic', vmin=0.0, vmax=1.0)
    plt.axis('off')
    plt.savefig('results/graphs-party/' + str(i) + '_party.png', dpi=200, bbox_inches='tight')
    plt.clf()

def draw_graph_precinct_level(graph, color_map, i, partition_edges=True, nodelist=None):
    if nodelist == None:
        nodelist = graph.nodes()

    if partition_edges:
        edgelist = graph.edges()
    else:
        edgelist = []
        for u, v in graph.edges():
            if color_map[str(u)] == color_map[str(v)]:
                edgelist.append((u, v))

    # node_color = [color_map[str(node)] for node in nodelist]
    pos = nx.get_node_attributes(graph, 'pos')

    # color_list = color_list_to_matched_precincts(color_map, matching_fn)
    # color_votes = count_votes_per_party(vote_map, color_list, candidate_party_map)
    precinct_level_votes = count_precinct_level_votes_per_party(vote_map, candidate_party_map)
    # print(candidate_party_map, vote_map)
    # print(precinct_level_votes)
    # return
    # color_votes_r_intensity = {}
    # for color, party_and_votes in color_votes.items():
    #     r_votes = party_and_votes['republican']
    #     d_votes = party_and_votes['democratic']
    #     intensity = float(r_votes) / (r_votes + d_votes)
    #     color_votes_r_intensity[color] = intensity
    # now color_votes_r_intensity is {color: [0, 1]}

    node_color = []
    j = 0
    for v in graph.nodes():
        # want to map each v to a color gradient (blue/red) based on party votes
        # color = color_map[str(v)]
        # intensity = color_votes_r_intensity[color]
        # matched_precinct_name, cl_county_name = matching_fn(str(v))
        matching_tup = matching_fn(str(v))
        if matching_tup == None:
            # set the unknown precinct to middle value so they appear as white
            node_color.append(0.5)
            continue

        matched_precinct_name, cl_county_name = matching_tup
            
        # print(matched_precinct_name, cl_county_name)
        party_and_votes = precinct_level_votes[cl_county_name][matched_precinct_name]
        # print(party_and_votes)
        r_votes = party_and_votes['republican']
        d_votes = party_and_votes['democratic']
        # print(r_votes, d_votes)
        try:
            intensity = float(r_votes) / (r_votes + d_votes)
        except ZeroDivisionError:
            # set to white
            intensity = 0.5
        node_color.append(intensity)

    nx.draw_networkx(graph, nodelist=nodelist, edgelist=edgelist, pos=pos, with_labels=False, node_size=15, node_color=node_color, cmap='seismic', vmin=0.0, vmax=1.0)
    plt.axis('off')
    plt.savefig('results/graphs-party/' + str(i) + '.png', dpi=200, bbox_inches='tight')
    plt.clf()

def plotter(graph, data):
    color_list = data['color_list']
    i = 0
    for color_map in color_list: 
        if i == 100:
            break
        draw_graph(graph, color_map, i, partition_edges=False)
        # draw_graph_precinct_level(graph, color_map, i, partition_edges=True)
        i += 1

if len(sys.argv) > 3:
    # location of samples
    location = sys.argv[1]
    graph, num_districts = parse_graph('data/GeorgiaGraph.json')
    with open(location) as f:
        data = json.load(f)

    vote_map_location = sys.argv[2]
    with open(vote_map_location) as f:
        vote_map = json.load(f)

    candidate_party_map_location = sys.argv[3] 
    with open(candidate_party_map_location) as f:
        candidate_party_map = json.load(f)

    plotter(graph, data)
else:
    print("Need jsons as input!")
