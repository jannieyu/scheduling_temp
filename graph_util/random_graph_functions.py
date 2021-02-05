import networkx as nx
import random



def random_all_fork(num_tasks, seed=None):
    """
    makes an all fork tree using nx functions. Randomly makes graph and then randomly picks root.
    :param num_tasks: number of tasks
    :param seed: seed, optional defaults to None
    :return: a random tree
    """

    if not seed:
        seed = random.randint(0, 10000000)
      
    tree = nx.generators.trees.random_tree(num_tasks, seed)
    directed_graph = tree.to_directed()
    random.seed(seed+1)
    root = random.choice(list(directed_graph.nodes))

    all_fork_tree = make_rooted_tree(directed_graph, root)


    return all_fork_tree, seed


def make_rooted_tree(G, root):
    """
    carefully prunes edges to make it a directed tree as a recursive subroutine
    :param G: Graph
    :param root: current root
    :return: Pruned Graph
    """
    for child in G.successors(root):
        G.remove_edge(child, root)
        if G.out_degree(child) != 0:
            G = make_rooted_tree(G, child)

    return G


def random_all_join(num_tasks, seed=None):
    """
    makes an all join tree using nx functions
    :param num_tasks: number of tasks
    :param seed: seed, optional defaults to None
    :return: a random tree
    """

    if not seed:
        seed = random.randint(0, 10000000)
        


    all_fork_tree, _ = random_all_fork(num_tasks, seed)
    all_join_tree = all_fork_tree.reverse(copy=True)
    return all_join_tree, seed


def subgraphs_via_dfs(G):
    """
    creates a list of subgraphs that are only all-fork/all-join via modified DFS
    :param G: Graph
    :return: list of subgraphs
    """

    # list to mark what nodes we have explored in DFS
    explored_node = [False for i in range(len(list(G.nodes())))]


    temp_G = G.copy()
    # list of subgraphs created
    subgraph = []

    while False in explored_node:

        root_lst = [x for x in temp_G.nodes() if temp_G.in_degree(x)==0]

        for node in root_lst:

            G_prime = nx.DiGraph()

            if explored_node[node] == False:
                G_prime.add_node(node)
                G_prime, explored_node = explore(G, G_prime, node, explored_node)
                subgraph.append(G_prime)

            temp_G.remove_node(node)



    return subgraph


def explore(G, G_prime, node, explored_node, direction=None):
    """
    Explores node's children if we are constructing an all-fork tree and
    explores node's parents if we are constructing an all-join tree.
    :param G: Graph
    :param G_prime: all-fork/all-join subgraph that we are constructing
    :param node: current node that we are exploring from
    :param explored_node: list that tracks if a node has been explored using DFS
    :param direction: marked 'fork' if we are creating an all-fork tree
                      marked 'join' if we are creating an all-join tree
                      marked 'None' if we haven't denoted the type of tree to
                      make
    :return: G_prime subgraph, updated explored_node list

    """
    explored_node[node] = True

    # if G_prime subgraph that we are constructing is an all-fork tree
    if direction == 'fork':
        for child in G.successors(node):
            if explored_node[child] == False:
                G_prime, explored_node = traverse_G_prime(G, G_prime, node, child, explored_node, direction)

    # if G_prime subgraph that we are constructing is an all-join tree
    elif direction == 'join':
        for parent in G.predecessors(node):
            if explored_node[parent] == False:
                G_prime, explored_node = traverse_G_prime(G, G_prime, node, parent, explored_node, direction)

    # if G_prime subgraph is empty and we haven't denoted a direction to explore
    else:
        if len(list(G.successors(node))) != 0:
            direction = 'fork'
            for child in G.successors(node):
                if explored_node[child] == False:
                    G_prime, explored_node = traverse_G_prime(G, G_prime, node, child, explored_node, direction)

        elif len(list(G.predecessors(node))) != 0:
            direction = 'join'
            for parent in G.predecessors(node):
                if explored_node[parent] == False:
                    G_prime, explored_node = traverse_G_prime(G, G_prime, node, parent, explored_node, direction)
        else:
            G_prime.add_node(node)

    return G_prime, explored_node


def traverse_G_prime(G, G_prime, node, next_node, explored_node, direction):
    """
    Recursive subroutine that adds the next_node to node (with appropriate edge direction)
    to G_prime and calls explore() on the next_node.
    :param G: Graph
    :param G_prime: all-fork/all-join subgraph that we are constructing
    :param node: current node that we are exploring from
    :param next_node: next node that we are exploring
    :param explored_node: list that tracks if a node has been explored using DFS
    :param direction: marked 'fork' if we are creating an all-fork tree
                      marked 'join' if we are creating an all-join tree
                      marked 'None' if we haven't denoted the type of tree to
                      make
    :return: G_prime subgraph, updated explored_node list

    """
    G_prime.add_node(next_node)
    G_prime.add_edge(node, next_node)
    G_prime, explored_node = explore(G, G_prime, next_node, explored_node, direction)

    return G_prime, explored_node



def num_leaf_nodes(G):
    """
    Finds the list of leaf nodes in G
    :param G: Graph
    :return: list of leaf nodes in G
    """
    leaf_lst = [x for x in G.nodes() if G.out_degree(x)==0]
    return leaf_lst



def num_root_nodes(G):
    """
    Finds the list of root nodes in G
    :param G: Graph
    :return: list of leaf nodes in G
    """
    root_lst = [x for x in G.nodes() if G.in_degree(x)==0]
    return root_lst
