from scheduling_util.optimization_functions import *
import networkx as nx



def opt_schedule_given_ordering(mrt, dag, weights, order, plot=False, compare=True):
    """
    solves optimization problem given ordering
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
    m1, s1, c1 = init_ordering_solver(mrt, dag, num_tasks, weights, order, task_scaling=True)

    _, task_process_time1, ending_time1, intervals1, speeds1, obj_value1 = solver_results(x, s1, m1, c1, weights, order=order, verbose=False)
    if plot:
     
        if obj_value1 == 10000000:
            return None


        metadata1 = make_task_metadata(order, num_tasks, intervals1)
        colors = plot_gantt(metadata1, obj_value1, color_palette)

    if compare:
        # get machine scaling ordering
        m2, s2, c2 = init_ordering_solver(mrt, dag, num_tasks, weights, order, task_scaling=False)
        _, task_process_time2, ending_time2, intervals2, speeds2, obj_value2 = solver_results(x, s2, m2, c2, weights, order=order, verbose=False)
        if plot:
            metadata2 = make_task_metadata(order, num_tasks, intervals2)
            colors = plot_gantt(metadata2, obj_value2, colors)

        return {'intervals_task_scaling': intervals1, 'speeds_task_scaling': speeds1, 'objective_task_scaling': obj_value1,
                'intervals_machine_scaling': intervals2, 'speeds_machine_scaling': speeds2,
                'objective_machine_scaling': obj_value2, 'order': order}
    else:
        return intervals1, speeds1, obj_value1

def relaxed_opt_schedule(mrt, dag, num_machines, weights, plot=False, verbose=False):
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
    x1, m1, s1, c1 = init_relaxed_opt_solver(mrt, dag, num_tasks, num_machines, weights)
    order, task_process_time1, ending_time1, intervals1, speeds1, obj_value1 = solver_results(x1, s1, m1, c1, weights, verbose)
    # print("Order is ", order)

    if plot:
        metadata1 = make_task_metadata(order, num_tasks, intervals1)
        colors = plot_gantt(metadata1, obj_value1, color_palette)

    return intervals1, speeds1, obj_value1, order