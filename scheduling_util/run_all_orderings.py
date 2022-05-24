from itertools import permutations, product
import networkx as nx

def get_permutations_helper(machine_order):
    perms = list(permutations(machine_order))
    [list(perm) for perm in perms]
    return [list(perm) for perm in perms]


def get_permutations(task_list, num_machines):
    # Get partitions
    set_partitions = list(set_partition(task_list, num_machines))

    # Permute across each partition
    task_permutations = []
    for partition in set_partitions:
        partition_permutations = [[list(machine)] if (len(list(machine)) == 0 or len(list(machine)) == 1) else get_permutations_helper(machine) for machine in partition]
        task_permutations.extend(list(product(*partition_permutations)))

    task_permutations = [list(perm) for perm in task_permutations]
    return task_permutations


def set_partition(ns, m):
    def visit(n, a):
        ps = [[] for i in range(m)]
        for j in range(n):
            ps[a[j + 1]].append(ns[j])
        return ps

    def f(mu, nu, sigma, n, a):
        if mu == 2:
            yield visit(n, a)
        else:
            for v in f(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v
        if nu == mu + 1:
            a[mu] = mu - 1
            yield visit(n, a)
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                yield visit(n, a)
        elif nu > mu + 1:
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = mu - 1
            else:
                a[mu] = mu - 1
            if (a[nu] + sigma) % 2 == 1:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v

    def b(mu, nu, sigma, n, a):
        if nu == mu + 1:
            while a[nu] < mu - 1:
                yield visit(n, a)
                a[nu] = a[nu] + 1
            yield visit(n, a)
            a[mu] = 0
        elif nu > mu + 1:
            if (a[nu] + sigma) % 2 == 1:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] < mu - 1:
                a[nu] = a[nu] + 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = 0
            else:
                a[mu] = 0
        if mu == 2:
            yield visit(n, a)
        else:
            for v in b(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v

    n = len(ns)
    a = [0] * (n + 1)
    for j in range(1, m + 1):
        a[n - m + j] = j - 1
    return f(m, n, 0, n, a)


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
    """
    Takes in all the possible permutations and splits across machines for tasks and returns only the valid
    orderings
    :param unpruned_orderings: 2d list of machine orderings
    :param graph: graph to schedule
    :return: 2d list of valid orderings
    """
    def check_ordering(ordering):
        valid = [check_valid_topological_ordering(t, graph) for t in ordering]
        return all(valid)

    return [ordering for ordering in unpruned_orderings if check_ordering(ordering)]
    


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


