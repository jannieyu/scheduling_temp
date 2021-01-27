import numpy as np
import plotly.figure_factory as ff
from itertools import permutations, groupby
import networkx as nx
import random
from opt_solver_util import (make_assignment_visual,
                             make_graph_visual,
                             create_start_end_times,
                              solver_results,
                              init_solver,
                              get_makespan,
                            v_helper,
                            define_constraints,
                            add_constraints)
from gekko import GEKKO


def general_power_function(alpha):
    """
    A function to return the appropriate power function for the equation
    :param alpha: scaling exponent for speed
    :return: A new power function with input argument of speed
    """

    def power_function(speed):
        return speed ** alpha

    return power_function


def general_speed_task_speed_scaling(alpha, coefficient):
    """
    returns the speed for the speed scaling case for each task given the alpha for the power function and
    the coefficient
    :param alpha: exponent for the power function
    :param coefficient: coefficient for the time term in the objective
    :return: optimal speed for the speed scaling case
    """
    return (coefficient / (alpha - 1)) ** (1 / alpha)


def general_speed_machine_speed_scaling(alpha, num_tasks, coefficient):
    """
    returns speed for machine given scaling exponent, number of tasks on that machine and coefficient for time
    :param alpha: exponent for power function
    :param num_tasks: number of tasks running on machine
    :param coefficient: coefficient for time term in objective
    :return: optimal speed for each machine
    """
    return (coefficient / (num_tasks * (alpha - 1))) ** (1 / alpha)


def objective(coefficients, speeds, power_function):
    """
    Given the coefficients for the reciprocal of the speeds for the individual tasks from time component of objective
    and the speeds of the individual tasks, objective calculates the objective value for the given power_function of
    the form p(s) = s^(alpha)
    :param coefficients:a list of coefficients of reciprocals of speed from the time part of the objective
    :param speeds: a list of speeds for each task
    :param power_function: function obtained from general_power_function
    :return: net objective
    """
    time_component = sum([coefficient / speed for coefficient, speed in zip(coefficients, speeds)])
    energy_component = sum([power_function(speed) / speed for speed in speeds])
    return time_component + energy_component


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
        new_total_ordering = [machine_ordering for machine_ordering in total_ordering if
                              check_valid_topological_ordering(machine_ordering, graph)]
        pruned_orderings.append(new_total_ordering)
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


def get_coefficients(order, dag):
    # deepcopy graph as we are going to change things
    graph = nx.Graph.copy(dag)
    # add attribute coefficients to each node
    coefficients = {}
    scheduled = {}
    nx.set_node_attributes(graph, coefficients, 'coefficients')
    nx.set_node_attributes(graph, scheduled, 'scheduled')
    for n in graph:
        graph.nodes[n]['coefficients'] = [0] * len(graph)
        graph.nodes[n]['scheduled'] = 0

    # PACK step
    # Get basic equation - precedence tasks in dag not obeyed here. Instead, precedence for machines put in here
    # This is corrected later
    for machine_schedule in order:
        for task in machine_schedule:
            parents = list(graph.predecessors(task))
            # inherit coefficients of previous task in machine and then add yours in
            if len(parents) != 0:
                latest_parent = max(parents, key=sum)
                graph.nodes[task]['coefficients'] = list(graph.nodes[latest_parent]['coefficients'])
            graph.nodes[task]['coefficients'][task] = 1


    # STRETCH step
    # make the tasks obey precedence constraints by adding terms to stretch the schedule to obey constraints


def get_task_permutation_dict(order):
    """
    returns a dict such that every task t's value is the previous job running on that machine
    :param order: 2d list of machine task ordering
    :return: task_permutation_dict
    """
    task_permutation_dict = {}
    for machine_order in order:
        for i in range(len(machine_order)):
            if i == 0:
                task_permutation_dict[machine_order[i]] = None
            else:
                task_permutation_dict[machine_order[i]] = machine_order[i-1]

    return task_permutation_dict


def get_t(order, graph, task_number):
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
    # Initialize indices of machines being currently tracked
    machine_indices = [0] * len(order)

    while not all(v=="blocked" for v in list(scheduled.values())):
        for machine_order_index in range(len(order)):
            # order for that particular machine
            machine_order = order[machine_order_index]
            # current task index to look at
            task_index = machine_indices[machine_order_index]
            # Check make sure that we haven't gone bigger than machine_order
            if task_index >= len(machine_order):
                break
            # current task
            task = machine_order[task_index]
            parents = list(graph.predecessors(task))
            # machine the task is scheduled on
            machine = task_machine_map[task]
            # print("task before loop is ", task)
            child_ready_to_schedule = parents_scheduled(parents, scheduled)
            if (len(parents) == 0 or child_ready_to_schedule) and (scheduled[task] == 'free'):
                # print("Task being scheduled on is ", task)
                if len(parents) != 0:
                    # print("Task in second loop is ", task)
                    max_finish_time = max(t[i][1] for i in parents)
                    t[task][0] = max(starting_times[machine], max_finish_time)
                    t[task][1] = t[task][0] + 1.0
                else:
                    t[task][0] = starting_times[machine]
                    t[task][1] = starting_times[machine] + 1.0
                scheduled[task] = "blocked"
                machine_indices[machine] += 1
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


def make_task_metadata(order, dependencies, task_process_time, num_tasks):
    """
    makes data ready to be plotted on a pretty gantt chart
    :param order: machine task ordering
    :param dependencies: dependencies for every task
    :param task_process_time: total time each task spent processing
    :param num_tasks: total number of tasks for the dag
    :return: a dict of metadata with subdict for fields
    """
    start_end_times = create_start_end_times(task_process_time, dependencies)
    task_metadata = {}
    machines = get_machines(order, num_tasks)
    for task_name in range(len(start_end_times)):

        task_metadata[task_name] = {'start': start_end_times[task_name][0], 'end': start_end_times[task_name][1],
                                    'task': task_name, 'machine': machines[task_name]}
    return task_metadata


def get_machines(order, num_tasks):
    """
    returns list of task machine mappings
    :param order: order of tasks across machines
    :param num_tasks: number of total tasks
    :return:
    """
    machines = num_tasks * [-1]
    for machine_index in range(len(order)):
        for task in order[machine_index]:
            machines[task] = machine_index
    return machines


def plot_task_speed_scaling(task_metadata, objective_value):
    """
    plots the task_speed_scaling gantt chart given the metadata
    :param task_metadata: metadata
    :param objective_value: value of objective for current value
    :return:
    """
    df = []
    colors = {}
    print(task_metadata)
    for task_key in task_metadata:
        task = task_metadata[task_key]
        df.append(dict(Task=str("Machine " + str(task['machine'])), Start=task['start'], Finish=task['end'], Machine=task['task']))
        color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        colors[task['task']] = color
    title = "Task Speed Scaling Gantt Chart for Objective: " + str(objective_value)
    fig = ff.create_gantt(df, colors=colors, index_col='Machine', show_colorbar=True, group_tasks=True, showgrid_x=True, showgrid_y=True, title=title)
    fig.update_xaxes(type='linear')
    fig.show("notebook")
    return


def speed_scaling_task(dag, machine_task_list, num_tasks):
    """
    speed scaling task gets the graph for any individual order given a networkx dag, an ordering and the number of
    tasks
    :param dag: networkx DiGraph DAG
    :param machine_task_list: ordering of task list to each machine 2d list
    :param num_tasks: number of tasks in dag
    :return:
    """
    # get dictionary of previous task on that machine
    task_prev = get_task_permutation_dict(machine_task_list)
    # get time chunks before speed scaling
    t = get_t(machine_task_list, dag, num_tasks)
    # fill in with idle time slots
    time_chunks = get_time_chunks(t, machine_task_list)
    # get dependencies andnumber of tasks that depend on a task
    v, dependencies = v_helper(dag, num_tasks, task_prev, machine_task_list, t)
    # initialize gekko solver
    m, s, O = init_solver(v)
    # add equality and relaxing constraints
    equality_constraints, relaxing_constraints = define_constraints(dag, time_chunks, dependencies)
    m, s = add_constraints(m, s, equality_constraints, relaxing_constraints)
    # get results from
    s, task_process_time, obj = solver_results(s, m, O, verbose=True)
    # plot the gantt chart
    task_metadata = make_task_metadata(machine_task_list, dependencies, task_process_time, 9)
    plot_task_speed_scaling(task_metadata, obj)
    return


def get_precedence_constraints(G, time_chunks, dependencies):
    num_machines = len(time_chunks)
    num_max_tasks = len(time_chunks[0])
    equality_constraints = []

    # index i
    for i in range(0, num_max_tasks):
        for j in range(num_machines):
            # key task that we are checking on
            task = time_chunks[j][i]
            # make sure that all parents finish before task
            if task != "idle":
                parents = G.predecessors(task)
                parent_list = []

                prev_task = time_chunks[j][i - 1]

                if prev_task != 'idle':

                    for k in range(len(time_chunks)):
                        p = time_chunks[k][i - 1]
                        if p in parents and (j != k):
                            parent_list.append(p)

                    for parent in parent_list:
                        left = dependencies[parent]
                        right = dependencies[prev_task]
                        equality_constraints.append([left, right])

    return equality_constraints


def init_solver_machine_scaling(v, machine_task_list, task_machine_dict, O_value=5):
    """

    :param v: number of tasks dependent on each task
    :param machine_task_list: order
    :param task_machine_dict: task machine mapping
    :param O_value: optimal objective value
    :return:
    """
    m = GEKKO()

    # Use IPOPT solver (default)
    m.options.SOLVER = 3

    # Change to parallel linear solver
    m.solver_options = ['linear_solver ma97']

    # variable array dimension
    n = len(machine_task_list)  # rows

    # create array
    s = m.Array(m.Var, n)
    for i in range(n):
        s[i].value = 2.0
        s[i].lower = 0

    # Optimal value for ibjective
    O = m.Var(value=O_value, lb=0)

    # The objective basically
    m.Equation(sum([int(v[i]) / s[task_machine_dict[i]] + s[task_machine_dict[i]] for i in range(len(v))]) == O)
    return m, s, O


def get_task_machine_dict(machine_task_list):
    """
    returns a dict of the task machine mappings
    :param machine_task_list: order
    :return:
    """
    task_machine_dict = {}
    for machine_index in range(len(machine_task_list)):
        for task in machine_task_list[machine_index]:
            task_machine_dict[task] = machine_index

    return task_machine_dict


def solver_results_machine_scaling(s, m, O, task_machine_dict, verbose=True):

    m.Obj(O) # Objective
    m.options.IMODE = 3 # Steady state optimization
    m.solve(disp=False) # Solve

    if verbose:
        print('Results')
        for i in range(len(s)):
            print(str(i) + " " + str(s[i].value))
        print('Objective: ' + str(m.options.objfcnval))

    task_process_time = [float(1 / s[task_machine_dict[i]].value[0]) for i in range(len(task_machine_dict))]

    return s, task_process_time, m.options.objfcnval


def define_constraints_machine_scaling(G, time_chunks, dependencies, machine_task_list):
    num_machines = len(time_chunks)
    num_max_tasks = len(time_chunks[0])
    equality_constraints = []
    relaxing_constraints = []
    # index i

    for i in range(1, num_max_tasks - 1):
        for j in range(num_machines):

            # key task that we are checking on
            task = time_chunks[j][i]

            if task != "idle":

                parents = G.predecessors(task)
                children = G.successors(task)

                # find tasks that need to run at the same time as other tasks
                share_parent_tasks = []
                share_children_tasks = []
                for z in range(num_machines):

                    if z != j and time_chunks[z][i] != 'idle':

                        concurr_task = time_chunks[z][i]
                        other_parents = G.predecessors(concurr_task)
                        other_children = G.successors(concurr_task)

                        share_parent = False
                        share_child = False
                        for a in range(num_machines):
                            prev_task = time_chunks[a][i - 1]
                            if prev_task in parents and prev_task in other_parents:
                                share_parent = True

                            next_task = time_chunks[a][i + 1]
                            if next_task in children and next_task in other_children:
                                share_child = True

                        if share_child and share_parent:
                            equality_constraints.append([task, concurr_task])

                # check if task can be relaxed to the right into idle time
                if time_chunks[j][i + 1] == 'idle':
                    min_machine = None
                    min_relax = num_max_tasks
                    children = G.successors(task)
                    for s in range(num_machines):
                        if s != j:
                            relax = 0

                            for v in range(i + 1, num_max_tasks):

                                if time_chunks[s][v] not in children:
                                    relax += 1
                                else:
                                    break

                            if relax < min_relax:
                                min_relax = relax
                                min_machine = s

                    own_relax = 0
                    for x in range(i + 1, num_max_tasks):
                        if time_chunks[j][x] == 'idle':
                            own_relax += 1
                        else:
                            break

                    end_task = time_chunks[min_machine][i + min(min_relax, own_relax)]
                    relaxing_constraints.append([dependencies[task], dependencies[end_task]])

                # make sure that all parents finish before task starts
                parents = G.predecessors(task)
                parent_list = []

                prev_task = time_chunks[j][i - 1]

                if prev_task != 'idle':

                    for k in range(len(time_chunks)):
                        p = time_chunks[k][i - 1]
                        if p in parents and (j != k):
                            parent_list.append(p)

                    for parent in parent_list:
                        left = dependencies[parent]
                        right = dependencies[prev_task]
                        equality_constraints.append([left, right])
    # add machine speed equality constraints
    for machine_index in range(len(machine_task_list)):
        curr_machine = machine_task_list[machine_index]
        for i in range(len(curr_machine)):
            if i != len(curr_machine) - 1:
                equality_constraints.append([curr_machine[i], curr_machine[i+1]])
            else:
                equality_constraints.append([curr_machine[i], curr_machine[0]])

    return equality_constraints, relaxing_constraints


if __name__ == "__main__":
    dag = nx.DiGraph()
    dag.add_nodes_from(range(9))
    dag.add_edges_from([(0, 2), (2, 3), (2, 4), (3, 5), (4, 5), (5, 6), (6, 7), (7, 8)])
    #nx.draw(dag2)

    machine_task_list = [[0, 2, 3, 5, 6, 7, 8], [4, 1]]
    dependencies = [[0], [0, 2, 4, 1], [0, 2], [0, 2, 3], [0, 2, 4], [0, 2, 3, 5], [0, 2, 3, 5, 6], [0, 2, 3, 5, 6, 7],
                    [0, 2, 3, 5, 6, 7, 8]]
    time_chunks = [[0, 2, 3, 5, 6, 7, 8], ['idle', 'idle', 4, 1, 'idle', 'idle', 'idle']]
    constraints = get_precedence_constraints(dag, time_chunks, dependencies)
    v = [9, 1, 8, 5, 2, 4, 3, 2, 1]
    task_machine_dict = get_task_machine_dict(machine_task_list)
    m, s, O = init_solver_machine_scaling(v, machine_task_list, task_machine_dict)
    s, task_process_time, w = solver_results_machine_scaling(s, m, O, task_machine_dict)
    print(task_process_time)
    #dag = nx.DiGraph()
    #dag.add_nodes_from(range(9))
    #dag.add_edges_from([(0, 2), (2, 3), (2, 4), (3, 5), (4, 5), (5, 6), (6, 7), (7, 8)])
    #t = get_t([[0, 2, 3, 5, 6, 7, 8], [4, 1]], dag, 9)
    #print(get_time_chunks(t, [[0, 2, 3, 5, 6, 7, 8], [4, 1]]))
    # print(get_task_permutation_dict([[0, 2, 3, 5, 6, 7, 8], [4, 1]]))
    # print("Sample code for functions.py")
    # dag = nx.DiGraph()
    # dag.add_nodes_from([1, 2, 3, 4])
    # dag.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])
    # sample_order = [1, 2, 3, 4]
    # good_order = [1, 2, 4]
    # bad_order = [2, 1, 4]
    # print(check_valid_topological_ordering(sample_order, dag))
    # print(check_valid_topological_ordering(good_order, dag))
    # print(check_valid_topological_ordering(bad_order, dag))
