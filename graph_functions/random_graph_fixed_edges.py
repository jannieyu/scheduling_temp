import networkx as nx
from random import seed, choices, choice


def random_dag_fixed_edges(n, e, seed):
    seed(seed)
    nodes = range(n)
    # Generate random graph
    G = nx.generators.random_graphs.gnm_random_graph(n, e, seed)
    # make it a dag by pruning wrong edges
    DAG = nx.DiGraph([(u, v) for (u, v) in G.edges() if u < v])
    # Add necessary number of extra edges
    num_edges = len(list(DAG.edges))
    if e > num_edges:
        # Id sources
        sources = choices(nodes[:n], k=e - num_edges)
        DAG.add_edges_from([(source, choice(nodes[source:])) for source in sources])
    return DAG