import numpy as np
import networkx as nx
from gekko import GEKKO
import matplotlib.pyplot as plt
import random
from makespan_energy.visualization_util import make_graph_visual




def constructGraph(num_tasks, edges, plot=True):
    G = nx.DiGraph()
    G.add_nodes_from(range(num_tasks))
    G.add_edges_from(edges)

    if plot:
        make_graph_visual(G, num_tasks)

    return G

def random_dag(nodes, edges, seed=5):
        """ Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges.
        """

        random.seed(seed+1)

        G = nx.DiGraph()
        for i in range(nodes):
            G.add_node(i)
        while edges > 0:
            a = random.randint(0,nodes-1)
            b=a
            while b==a:
                b = random.randint(0,nodes-1)

            G.add_weighted_edges_from([(a, b, 1)])
            if nx.is_directed_acyclic_graph(G):
                edges -= 1
            else:
                # we closed a loop!
                G.remove_edge(a,b)
        return G
