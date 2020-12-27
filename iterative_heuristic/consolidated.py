from conjecture.consolidated_functions import *
from conjecture.all_valid_orderings import *
from conjecture.optimization_functions import *
from iterative_heuristic.modified_etf import Mod_ETF
from iterative_heuristic.approximate_speeds import *
from makespan_energy.construct_graph_util import *
from makespan_energy.visualization_util import *
from graph_functions.erdos_renyi_dag import random_dag as rd
from iterative_heuristic.naive_rescheduler import *
import networkx as nx
import numpy as np
import math
import random
import copy

def iterative_heuristic(num_tasks, num_machines, seed, homogeneous=True, verbose=False):
    did_not_work = False
    while did_not_work==False:
        random.seed(seed)
        G = rd(num_tasks, 0.05,seed)
        # print(G.number_of_nodes())
        w = [random.randint(1, 50) for _ in range(num_tasks)]
        s = [1 for i in range(num_tasks)]
        tie_breaking_rule = 2
        # Get ordering using modified ETF
        test = Mod_ETF(G, w, s, num_machines, tie_breaking_rule, plot=verbose)
    
        #     # Initialize objective function value
        heuristic_opt = test.obj_value

        #     while True:
        # Get pseudosize, convert to speed
        if homogeneous:
            p_size = approx_psize_homogeneous(G, test.order, test.h, test.t)
        else:
            p_size = approx_psize_heterogeneous(G, test.order, test.t)
            
        s_prime = psize_to_speed(p_size)
        test_heuristic = Mod_ETF(G, w, s_prime, num_machines, tie_breaking_rule, plot=verbose)

        temp = get_objective_single_ordering(True, G, w, test_heuristic.order, plot=verbose, compare=False)
        
        opt_intervals, s_opt, obj_opt, _ = temp
        if obj_opt!= 10000000:
            did_not_work = True
        seed +=1
    return obj_opt / test_heuristic.obj_value


# def iterative_heuristic_no_ratio(num_machines, w, G, verbose=False):
#     #G = rd(num_tasks, 0.05,seed)
#     ## print(G.number_of_nodes())
#     #w = [random.randint(1, 50) for _ in range(num_tasks)]
#     s = [1 for i in range(len(w))]
#     tie_breaking_rule = 2
#     # Get ordering using modified ETF
#     test = Mod_ETF(G, w, s, num_machines, tie_breaking_rule, plot=verbose)
    
#     #     # Initialize objective function value
#     heuristic_opt = test.obj_value

#     #     while True:
#     # Get pseudosize, convert to speed
#     s_prime = approx_speeds(G, test.order)

#     # Get ordering using modified ETF
#     test2 = Mod_ETF(G, w, s_prime, num_machines, tie_breaking_rule, plot=verbose)
#     temp = get_objective_single_ordering(True, G, w, test2.order, plot=verbose, compare=False)
#     opt_intervals, s_opt, obj_opt, _ = temp
#     if obj_opt!= 10000000:
#         return test2.obj_value / obj_opt
#     else:
#         return 10000000

#     return


def iterative_heuristic_no_ratio(num_machines, w, G, homogeneous=True, verbose=False):
    
    s = [1 for i in range(len(w))]
    tie_breaking_rule = 2

    # Get ordering using modified ETF
    test = Mod_ETF(G, w, s, num_machines, tie_breaking_rule, plot=verbose)
    
    if homogeneous:
        p_size = approx_psize_homogeneous(G, test.order, test.h, test.t)
    else:
        p_size = approx_psize_heterogeneous(G, test.order, test.t)
        
    s_prime = psize_to_speed(p_size)
    test_heuristic = Mod_ETF(G, w, s_prime, num_machines, tie_breaking_rule, plot=verbose)
        
    return test_heuristic.obj_value

def iterative_and_naive_heuristic_no_ratio(num_machines, w, G, naive_version=2, iterations=1, homogeneous=True, verbose=False):
    '''
    Runs both the iterative heuristic and the naive method for generating a schedule given G.
    :param num_machines: number of machines 
    :param w:
    :param G:
    :param naive_version: If 1, we have the naive version that runs ETF first, 
    if 2, we have the naive version that creates pseudosize first before running 
    ETF. (2 is more naive)
    :param iterations: Set the number of iterations that we run the iterative method for
    :param homogeneous: If True, we solve for the problem in the homogeneous setting. 
    :param verbose: If True, graphs will be plotted out.
    
    '''
    
    s = [1 for _ in range(len(w))]
    tie_breaking_rule = 2

    # Get initial ordering using modified ETF
    test = Mod_ETF(G, w, s, num_machines, tie_breaking_rule, plot=verbose)

    # Run the naive method
    if naive_version == 1:
        naive_cost = get_cost_naive_1(num_machines, w, G, test.order)
    else:
         
        naive_cost, _ = get_cost_naive_2(G, w, num_machines)
 
    # Run the iterative heuristic
    for _ in range(iterations):

        # Update pseudosize
        if homogeneous:
            p_size,_ = approx_psize_homogeneous(G, test.order, test.h, test.t)
        else:
            p_size,_ = approx_psize_heterogeneous(G, test.order, test.t)

        s = psize_to_speed(p_size)

        # Construct new ordering
        test = Mod_ETF(G, w, s, num_machines, tie_breaking_rule, plot=verbose)

    return naive_cost, test.obj_value, test.order



def compare_naive_versions(num_machines, w, G, verbose=False):

    s = [1 for i in range(len(w))]
    tie_breaking_rule = 2

    # Get initial ordering using modified ETF
    test = Mod_ETF(G, w, s, num_machines, tie_breaking_rule, plot=verbose)

    naive_cost1 = get_cost_naive_1(num_machines, w, G, test.order)
    naive_cost2, _ = get_cost_naive_2(num_machines, w, G)

    return naive_cost1, naive_cost2, test.order

# def compute_cost(w, t, s):
#     total_cost = 0
#     for j in range(len(s)):
#         total_cost += (t[j][1] + (w[j] * s[j]))
#     return total_cost

