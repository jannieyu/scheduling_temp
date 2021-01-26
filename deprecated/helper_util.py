import numpy as np
# import plotly.figure_factory as ff
from itertools import permutations, groupby
import networkx as nx
import random


def get_task_permutation_dict(order):
    """
    returns a dict such that every task t's value is the previous job running on that machine
    :param order: 2d list of machine task ordering
    :return: task_permutation_dict
    """
    task_permutation_dict = {}
    for task_lst in order:
        for i in range(len(task_lst)):
            if i == 0:
                task_permutation_dict[task_lst[i]] = None
            else:
                task_permutation_dict[task_lst[i]] = task_lst[i-1]

    return task_permutation_dict

    

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
    all_permutations = partition(machine_number, task_list)

    # Get only topologically valid orders (remove invalid orders)
    orderings = prune_permutations(all_permutations, graph)
    return orderings


def partition(machine_number, tasks):
    """
    partition splits the tasks up for number of machines. NOTE: These may include topologically invalid solutions,
    need to prune after this.
    :param machine_number: Number of machines to split up by
    :param tasks: list of tasks to split up
    :return: List of all orderings. Each ordering is a 2d list where each sublist is for a single machine.
    Can be used to indicate which machine. For instance, index 0 can mean machine 0, index 1 machine 1 and so on.
    """
    # append | to indicate end of each machine
    tasks_duplicate = list(tasks)
    for split in range(machine_number - 1):
        tasks_duplicate.append('|')

    # Split based on |
    barred_permutations = list(permutations(tasks_duplicate))
    unbarred_permutations = []
    for ordering in barred_permutations:
        ordering = [list(group) for k, group in groupby(ordering, lambda x: x == "|") if not k]
        unbarred_permutations.append(ordering)
        # Pad each ordering with empty lists if no task is scheduled on one machine
        while len(ordering) < machine_number:
            ordering.append([])

    return unbarred_permutations


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


def get_t(order, graph, task_number, task_process_time):
    """
    returns t which is basically time that each task starts at before speed scaling
    :param order: task-machine assignment 2d list
    :param graph: dag used to schedule
    :param task_number: number of tasks in dag
    :return: t (list)
    """
    # initialize t with starting and ending times for each task.
    t = [[0, 0] for i in range(task_number)]

    # initialize scheduled dict for each task to free
    scheduled = {}
    for task in range(task_number):
        scheduled[task] = "free"

    task_machine_map = {}
    for machine in range(len(order)):
        for task in order[machine]:
            task_machine_map[task] = machine

    # Initialize Array of starting times for each machine
    starting_times = [0.0] * len(order)

    # Initialize indices of task currently being scheduled
    curr_task_on_machine = [0] * len(order)

    machine_cycle_count = 0

    while not all(v=="blocked" for v in list(scheduled.values())):
  
        machine_cycle_count += 1
        machine = machine_cycle_count % len(order)
        # order for that particular machine
        task_lst = order[machine]
        # current task index to look at
        task_index = curr_task_on_machine[machine]


        while task_index >= len(task_lst):
                machine_cycle_count += 1
                machine = machine_cycle_count % len(order)

                # order for that particular machine
                task_lst = order[machine]
                # current task index to look at
                task_index = curr_task_on_machine[machine]

        # current task
        task = task_lst[task_index]
        parents = list(graph.predecessors(task))

 
        # machine the task is scheduled on
        # machine = task_machine_map[task]
        # print("task before loop is ", task)
        child_ready_to_schedule = parents_scheduled(parents, scheduled)

 
        if (len(parents) == 0 or child_ready_to_schedule) and (scheduled[task] == 'free'):
            # print("Task being scheduled on is ", task)
            if len(parents) != 0:
                # print("Task in second loop is ", task)
                max_finish_time = max(t[i][1] for i in parents)
                t[task][0] = max(starting_times[machine], max_finish_time)
                t[task][1] = t[task][0] + task_process_time[task]
            else:
                t[task][0] = starting_times[machine]
                t[task][1] = starting_times[machine] + task_process_time[task]
            scheduled[task] = "blocked"
            curr_task_on_machine[machine] += 1
    
        starting_times[machine] = t[task][1]

    return t


def get_time_chunks(t, order):
    """
    returns the order padded by idle time
    :param t: time list for each task before speed scaling
    :param order: task machine ordering
    :return: order padded by idle time
    """
    # multi-dimensional data
    machine_data = [[] for _ in range(len(order))]
    machine_labels = [[] for _ in range(len(order))]
    for m in range(len(order)):
        machine_etfd = 0
        task_list = order[m]
        for task in task_list:
            if machine_etfd < t[task][0]:
                idle_time = t[task][0] - machine_etfd

                for i in range(int(idle_time)):
                    machine_data[m].append(idle_time)
                    machine_labels[m].append('idle')

            process_time = 1.0
            # print("process time is ", process_time, "for task ", t)
            machine_etfd = t[task][1]
            machine_data[m].append(process_time)
            machine_labels[m].append(task)

    segments = max([len(task_list) for task_list in machine_data])
    for i in range(len(machine_data)):
        for j in range(len(machine_data[i]), segments):
            machine_data[i].append(0)
            machine_labels[i].append('idle')
    return machine_labels


def parents_scheduled(parents, scheduled):
    """
    helper function to indicate whether the parents of a node are blocked or not
    :param parents: list of nodes
    :param scheduled: dictionary indicating blocked or not
    :return: boolean, True or False, True if all parents are blocked
    """
    for node in parents:
        if scheduled[node] == 'free':
            return False
    return True

