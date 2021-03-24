# -*- coding: utf-8 -*-
import node2vec as n2v
import networkx as nx


#####load network
def read_graph(args):
    if args.weighted:
        g = nx.read_edgelist(args.input, nodetype=str, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        g = nx.read_edgelist(args.input, nodetype=str, create_using=nx.DiGraph())
        for edge in g.edges():
            g[edge[0]][edge[1]]['weight'] = 1
    if not args.directed:
        g = g.to_undirected()
    return g



####random walk
def node_walk(args):
    nx_g = read_graph(args)
    g = n2v.Graph(nx_g, args.directed, args.p, args.q)
    g.preprocess_transition_probs()
    walks = g.simulate_walks(args.num_walks, args.walk_length)
    return walks