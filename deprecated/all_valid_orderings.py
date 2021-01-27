from conjecture.permutations import get_permutations
import networkx as nx


def get_orderings(graph, machine_number):
    """
    Gets all the precedent constrained orderings for tasks across various machines.
    Suitable for both speed scaled tasks and speed scaled machines.
    :param graph: networkx DAG to schedule
    :param machine_number: number of machines to schedule over
    :return: all valid orderings of tasks on machines. This is as a 2d list in the following format
    [ [[ordering for machine 1], [ordering for machine 2], [ordering for machine 3], [[], [], []] ....]
    """
    task_list = list(graph.nodes)

    # Partition tasks across machines
    all_permutations = get_permutations(task_list, machine_number)

    # Get only topologically valid orders (remove invalid orders)
    orderings = prune_permutations(all_permutations, graph)
    return orderings


def prune_permutations(unpruned_orderings, graph):
    # TODO remove for loop and make it a nested list comprehension for efficiency
    """
    Takes in all the possible permutations and splits across machines for tasks and returns only the valid
    orderings
    :param unpruned_orderings: 2d list of machine orderings
    :param graph: graph to schedule
    :return: 2d list of valid orderings
    """
    pruned_orderings = []

    for total_ordering in unpruned_orderings:

        add = True

        for task_lsting in total_ordering:
            if not check_valid_topological_ordering(task_lsting, graph):
                add = False
                break

        if add:
            pruned_orderings.append(total_ordering)
    return pruned_orderings


def check_valid_topological_ordering(order, graph):
    """
    Takes in a valid order for any machine and then checks if descendants are already visited
    If visited, returns False, else True
    :param order: 1d list of nodes in the dag
    :param graph: dag to schedule
    :return: boolean, True, if it is valid ordering, False otherwise
    """

    visited = []
    for node in order:
        visited.append(node)
        reachable_nodes = list(nx.algorithms.dag.descendants(graph, node))
        for descendant in reachable_nodes:
            if descendant in visited:
                return False
    return True


if __name__ ==  "__main__":
    dag = nx.DiGraph()
    dag.add_nodes_from(range(5))
    dag.add_edges_from([(0,1), (1,2), (0,2), (2, 3), (3, 4)])
    print(get_orderings(dag, 3))