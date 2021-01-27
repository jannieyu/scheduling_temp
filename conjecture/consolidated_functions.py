from conjecture.optimization_functions import *
import networkx as nx


def check_conjecture_all_orderings(mrt, dag, num_machines, weights, verbose=False):
    """
    Checks whether conjecture is true for all valid orderings
    :param mrt: Boolean variable that is True if objective is to optimize for
                MRT + E, False if objective is to optimize Makespan + E.
    :param dag: DAG to schedule
    :param num_machines: number of machines to schedule over
    :param weights: List of task weights, or size of each task
    :param verbose: Boolean variable that is True if verbose print statements
                    are desired, False otherwise
    :return:
    """

    if verbose:
        if mrt:
            print("Using objective to minimize MRT + E.")
        else:
            print("Using objective to minimize Makespan + E.")
        print("Finding total number of topological valid orderings...")

    topological_valid_orderings = get_orderings(dag, num_machines)

    if verbose:
        print("Number of topological valid orderings: " + str(len(topological_valid_orderings)))

    orderings_metadata = []
    counter = 0
    print(topological_valid_orderings[58])
    # Solve for all valid topological orderings
    for order in topological_valid_orderings:

        counter += 1
        if verbose:
            print("Checking order " + str(counter))

        orderings_metadata.append([order, get_objective_single_ordering(mrt, dag, weights, order)])


    # Get the orderings with the minimal objective
    max_objective_task_scaling = min([data[1]['objective_task_scaling'] for data in orderings_metadata])
    max_keys_objective_task_scaling = [data for data in orderings_metadata if data[1]['objective_task_scaling'] ==
                                       max_objective_task_scaling]
    max_objective_machine_scaling = min([data[1]['objective_machine_scaling'] for data in orderings_metadata])
    max_keys_objective_machine_scaling = [data for data in orderings_metadata if data[1]['objective_machine_scaling'] ==
                                          max_objective_machine_scaling]

    print(max_keys_objective_task_scaling)
    print(max_keys_objective_machine_scaling)

    # check if orderings are equal
    for order1_data in max_keys_objective_task_scaling:
        for order2_data in max_keys_objective_machine_scaling:
            if order1_data[0] == order2_data[0]:
                print("Conjecture Valid! Orderings match")
                return order1_data, order2_data

    print("Conjecture Invalid! Orderings don't match.")
    return max_keys_objective_task_scaling, max_keys_objective_machine_scaling


def get_objective_single_ordering(mrt, dag, weights, order, plot=False, compare=True):
    """
    gets the objective dict for a single dict
    :param mrt: Boolean variable that is True if objective is to optimize for
                MRT + E, False if objective is to optimize Makespan + E.
    :param dag: DAG to schedule
    :param weights: List of task weights, or size of each task
    :param order: List of orderings of tasks on each machine
    :param plot: Boolean variable that is True if final plot is desired,
                 False otherwise
    :return:
    """
    num_tasks = dag.number_of_nodes()

    # Signal that we do not need binary variable to find ordering
    x = None

    color_palette = [(0, 0, 255 / 256), (0, 255 / 256, 0), (255 / 256, 255 / 256, 0), (255 / 256, 0, 0),
                     (255 / 256, 128 / 256, 0),
                     (255 / 256, 0, 127 / 256), (0, 255 / 256, 255 / 256), (127 / 256, 0, 255 / 256),
                     (128 / 256, 128 / 256, 128 / 256),
                     (255 / 256, 255 / 256, 255 / 256), (0, 0, 0)]
    # get task scaling ordering
    m1, s1, c1 = init_solver(mrt, dag, num_tasks, weights, order, task_scaling=True)

    _, task_process_time1, ending_time1, intervals1, speeds1, obj_value1 = solver_results(x, s1, m1, c1, weights, order=order, verbose=False)
    if plot:
        #
        # print (task_process_time1, ending_time1, intervals1, speeds1, obj_value1 )

        if obj_value1 == 10000000:
            return None


        metadata1 = make_task_metadata(order, num_tasks, intervals1)
        colors = plot_gantt(metadata1, obj_value1, color_palette)

    if compare:
        # get machine scaling ordering
        m2, s2, c2 = init_solver(mrt, dag, num_tasks, weights, order, task_scaling=False)
        _, task_process_time2, ending_time2, intervals2, speeds2, obj_value2 = solver_results(x, s2, m2, c2, weights, order=order, verbose=False)
        if plot:
            metadata2 = make_task_metadata(order, num_tasks, intervals2)
            colors = plot_gantt(metadata2, obj_value2, colors)

        return {'intervals_task_scaling': intervals1, 'speeds_task_scaling': speeds1, 'objective_task_scaling': obj_value1,
                'intervals_machine_scaling': intervals2, 'speeds_machine_scaling': speeds2,
                'objective_machine_scaling': obj_value2, 'order': order}
    else:
        return intervals1, speeds1, obj_value1, {'intervals_task_scaling': intervals1, 'speeds_task_scaling': speeds1, 'objective_task_scaling': obj_value1, 'order': order}


def get_optimal_schedule(mrt, dag, num_machines, weights, plot=False, verbose=False):
    """
    gets the objective dict for a single dict
    :param mrt: Boolean variable that is True if objective is to optimize for
                MRT + E, False if objective is to optimize Makespan + E.
    :param dag: DAG to schedule
    :param weights: List of task weights, or size of each task

    :return:
    """
    num_tasks = dag.number_of_nodes()
    color_palette = [(0, 0, 255 / 256), (0, 255 / 256, 0), (255 / 256, 255 / 256, 0), (255 / 256, 0, 0),
                     (255 / 256, 128 / 256, 0),
                     (255 / 256, 0, 127 / 256), (0, 255 / 256, 255 / 256), (127 / 256, 0, 255 / 256),
                     (128 / 256, 128 / 256, 128 / 256),
                     (255 / 256, 255 / 256, 255 / 256), (0, 0, 0)]
    # get task scaling ordering
    x1, m1, s1, c1 = init_opt_solver(mrt, dag, num_tasks, num_machines, weights)
    order, task_process_time1, ending_time1, intervals1, speeds1, obj_value1 = solver_results(x1, s1, m1, c1, weights, verbose)
    # print("Order is ", order)

    if plot:
        metadata1 = make_task_metadata(order, num_tasks, intervals1)
        colors = plot_gantt(metadata1, obj_value1, color_palette)

    return intervals1, speeds1, obj_value1, order




if __name__ == '__main__':
    dag1 = nx.DiGraph()
    dag1.add_nodes_from(range(5))
    dag1.add_edges_from([(0, 1), (0, 4), (2, 1), (2, 4)])
    weights = [1] * 5
    order1_data, order2_data = check_conjecture_all_orderings(dag1, 2, weights)
