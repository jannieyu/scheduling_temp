import networkx as nx
import random
import numpy as np
from makespan_energy.visualization_util import make_graph_visual


def er_dag(n, p, seed, plot=False):
    """
    Adds edges by erdos renyi.
    Takes node, gets nodes higher than it in ordinality, number = k, and then bin(k, p) to determine no of children,
    then randomly samples children from higher order node. Repeats till end.
    :param n: number of nodes
    :param p: probability of connection
    :param seed: optional seed
    :param plot: optional, plots graph if true
    :return: random networkx dag G
    """
    random.seed(seed)
    np.random.seed(seed)
    all_nodes = range(n)
    G = nx.DiGraph()
    G.add_nodes_from(all_nodes)
    for node in all_nodes[:n-1]:
        # Get higher order nodes
        k = n - node - 1
        num_children = np.random.binomial(k, p)
        # length1 = len(all_nodes[node+1:])
        # sample for children in higher order nodes
        # if length1 >= num_children:
        # print(length1, num_children)
        # print("num_children is ", num_children, "k is ", k, "node is ", node, "n is ", n)
        children = random.sample(all_nodes[node + 1:], num_children)
        # else:
        # print("in else")
        # print(length1, num_children)
        # if length1 != 0:
        #    children = random.sample(all_nodes[node + 1:], length1)
        # else:
        #        children = []
        # print("children ", children)
        if len(children) > 0:
            # print("length bigger than 0")
            # add edges to graph if not last node
            G.add_edges_from([(node, child) for child in children])
    if plot:
        make_graph_visual(G, n)
    # Verify that graph is a dag
    # print(nx.is_directed_acyclic_graph(G))
    return G
