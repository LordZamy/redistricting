from chain_tool import MarkovChain
from json_parser_tool import parse_graph

from graph_tool.draw import graph_draw

input_graph, num_colors = parse_graph('./data/GeorgiaGraph.json')
q = 0.5
R = 2
redistricting_chain = MarkovChain(input_graph, q, num_colors, R)

for i in range(100):
    redistricting_chain.simulate_step(0.5)
    print(i)
