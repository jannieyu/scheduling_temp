import networkx as nx
import numpy as np
from graph_util.visualization_util import *
from scheduling_util.modified_etf import *
from scheduling_util.consolidated_functions import get_objective_single_ordering
from scheduling_util.approx_pseudosizes import *


# def iterative_heuristic_algorithm(dag, weights, num_machines, speeds=None, check_optimal=False, plot=True):
#     """
#     runs the heuristic for the algorithm once.
#     :param dag: Dag being scheduled
#     :param weights: Weights for the tasks
#     :param num_machines: number of machines to schedule on
#     :param speeds: Optional argument, default None which reverts to sqrt(descendants + 1),
#     that can preinitialize speeds to whatever you want
#     :param check_optimal: Optional Boolean on whether to check optimal speed scaling for ordering. Default to False
#     :return: Final order returned by Mod_ETF
#     """

#     num_tasks = dag.number_of_nodes()

#     if speeds is None:
#         # Initialize homogeneous speeds for all tasks
#         # for i in range(num_tasks):
#             # print(nx.algorithms.descendants(dag, i))
#         speeds = [np.sqrt(len(nx.algorithms.descendants(dag, i)) + 1) for i in range(num_tasks)]

#     tie_breaking_rule = 2

  
#     # Get ordering using modified ETF
#     print("First Ordering after initialization")
#     test = Mod_ETF(dag, weights, speeds, num_machines, tie_breaking_rule, plot=plot)

#     best_order = test.order

#     heuristic_opt = test.obj_value

#     psize = approx_psize(dag, best_order, test.t, verbose=False)
#     s_prime = psize_to_speed(psize)

#     # Get ordering using modified ETF
#     print("Second Ordering pseudosize estimation")
#     test2 = Mod_ETF(dag, weights, s_prime, num_machines, tie_breaking_rule, plot=plot)

#     # Check if the objective function value has improved
#     if heuristic_opt > test.obj_value:
#         heuristic_opt = test.obj_value
#         best_order = test2.order

#     heuristic_speeds = s_prime
#     print("Final objective value: " + str(heuristic_opt))

#     print("Heuristic Speeds are:")
#     for i in range(len(heuristic_speeds)):
#         print("task " + str(i) + ":" + str(heuristic_speeds[i]))

#     if check_optimal:
#         print("Checking Optimal Below")
#         try:
#             opt_intervals, opt_speeds, opt_obj_value, _ = get_objective_single_ordering(True, dag, weights, best_order, plot=plot, compare=False)
#             print("Optimal Speeds are:")
#             for i in range(len(opt_speeds)):
#                 print("task " + str(i) + ":" + str(opt_speeds[i]))
#             compare_pseudosize(heuristic_speeds, opt_speeds, square_comparison=True, verbose=True)
#         except:
#             print("Was not able to find optimal ordering")

#     return list(test.order)


def compare_pseudosize(heuristic_speeds, optimal_speeds, square_comparison=True, verbose=True):
    """
    Gets the ratio of heuristic vs optimal speeds
    :param heuristic_speeds: Speeds scaled by heuristic
    :param optimal_speeds: Speeds scaled by optimal solver
    :param square_comparison: Whether to compare by square of the speeds or not, defaults to True
    :param verbose: whether to print or not, defaults to True
    :return: list of ratios
    """
    ratios = []
    if square_comparison:
        for i in range(len(heuristic_speeds)):
            ratios.append((heuristic_speeds[i])**2 / (optimal_speeds[i])**2)
            if verbose:
                print(str(i) + ": Ratio is ", ratios[i])
    else:
        for i in range(len(heuristic_speeds)):
            ratios.append(heuristic_speeds[i] / optimal_speeds[i])
            if verbose:
                print(str(i) + ": Ratio is ", ratios[i])
    return ratios


def machine_manipulation(start_machine_number, end_machine_number, machine_step, dag, weights, speeds=None, plot=True):
    """
    manipulates number of machines for iterative heuristic given arguments. Automatically stops running
     if ordering doesn't change adding a machine.
    :param start_machine_number: no. of machines to start testing for
    :param end_machine_number: no. of machines to end testing for
    :param machine_step: step size for machines
    :param dag: DAG of tasks to be scheduled
    :param weights: weights for tasks (list)
    :param speeds: optional argument to preinitialize speeds, defaults to sqrt(descendants +1)
    :return:
    """
    obj_list = []
    m_star = None
    print("Graph structure is ")
    make_graph_visual(dag, dag.number_of_nodes())
    order1 = None
    order2 = None
    for num_machines in range(start_machine_number, end_machine_number + 1, machine_step):
        print("Testing for machine number: ", str(num_machines))
        order1 = iterative_heuristic_algorithm(dag, weights, num_machines, speeds, check_optimal=True, plot=plot)
        order1 = [x for x in order1 if x != []]
        print("order is ", order1)
        if order1 == order2:
            m_star = num_machines - 1
            break
        order2 = list(order1)

    if m_star == None:
        m_star = "no"

    print("m* = " + str(m_star))
    return m_star

