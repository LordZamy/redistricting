import networkx as nx
import fiona
from shapely.geometry import *
import pickle

def parse_shape(location, graph=None):
    fprecincts = fiona.open(location)
    nameprop = "ID"

    precincts = []
    for p in fprecincts:
        coords = p['geometry']['coordinates'][0]
        gtype = p['geometry']['type']
        if (gtype == 'Polygon'):
            pPoly = Polygon(coords)
        elif (gtype == 'MultiPolygon'):
            pPoly = MultiPolygon([Polygon(c) for c in coords])
        p['shape'] = pPoly #.simplify(0.2)
        precincts.append(p)

    pos = {}
    for p in precincts:
        pPoly = p['shape']
        name = p["properties"][nameprop]
        # skip adding node to pos if does not exist in given graph
        if graph != None and not graph.has_node(name):
            continue
        x, y = pPoly.centroid.xy
        pos[name] = (x[0], y[0])

    return pos

def parse_pos(location=None):
    graph = pickle.load(open("data/graph.pkl", "rb"))
    return graph["pos"]
