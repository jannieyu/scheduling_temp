from conjecture.consolidated_functions import *
from conjecture.all_valid_orderings import *
from conjecture.optimization_functions import *
from iterative_heuristic.modified_etf import Mod_ETF
from iterative_heuristic.approximate_speeds import *
from makespan_energy.construct_graph_util import *
from makespan_energy.visualization_util import *
from graph_functions.erdos_renyi_dag import er_dag as rd
from iterative_heuristic.naive_rescheduler import *
import networkx as nx
import numpy as np
import math
import random
import copy


def iterative_and_naive_heuristic_no_ratio_expanded(num_machines, w, G, naive_version=2, iterations=1, homogeneous=True, verbose=False):
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
    
    s = [1 for i in range(len(w))]
    tie_breaking_rule = 2

    # Get initial ordering using modified ETF
    test = Mod_ETF(G, w, s, num_machines, tie_breaking_rule, plot=verbose)

    # Run the naive method
    if naive_version == 1:
        p_size = approx_psize_naive(G, test.order)
        s_prime_naive = psize_to_speed(p_size)
        naive_t = native_rescheduler(G, s_prime_naive, w, copy.deepcopy(test.order))
        naive_cost = compute_cost(w, naive_t, s_prime_naive)
        for i in range(iterations):

            # Update pseudosize
            if homogeneous:
                p_size,_ = approx_psize_homogeneous(G, test.order, test.h, test.t)
            else:
                p_size,_ = approx_psize_heterogeneous(G, test.order, test.t)

            s = psize_to_speed(p_size)

            # Construct new ordering
            test = Mod_ETF(G, w, s, num_machines, tie_breaking_rule, plot=verbose)

        return naive_cost, test.obj_value, test.order
    elif naive_version ==2:
        psize = [len(nx.algorithms.dag.descendants(G, task))+1 for task in range(len(G))] 
        s_prime_naive = psize_to_speed(p_size)
        naive_t = native_rescheduler(G, s_prime_naive, w, copy.deepcopy(test.order))
        naive_cost = compute_cost(w, naive_t, s_prime_naive)
        for i in range(iterations):

            # Update pseudosize
            if homogeneous:
                p_size,_ = approx_psize_homogeneous(G, test.order, test.h, test.t)
            else:
                p_size,_ = approx_psize_heterogeneous(G, test.order, test.t)

            s = psize_to_speed(p_size)

            # Construct new ordering
            test = Mod_ETF(G, w, s, num_machines, tie_breaking_rule, plot=verbose)

        return naive_cost, test.obj_value, test.order
    else:
        psize = approx_psize_naive(G, test.order)
        s_prime_naive = psize_to_speed(psize)
        # print("naive1 speed = \n", s_prime_naive) 
        naive_t = native_rescheduler(G, s_prime_naive, w, copy.deepcopy(test.order))
        naive_cost1 = compute_cost(w, naive_t, s_prime_naive)
        
        psize = [len(nx.algorithms.dag.descendants(G, task))+1 for task in range(len(G))] 
        s_prime_naive = psize_to_speed(psize)
        # print("naive2 speed = \n", s_prime_naive) 
        naive_t = native_rescheduler(G, s_prime_naive, w, copy.deepcopy(test.order))
        naive_cost2 = compute_cost(w, naive_t, s_prime_naive)
        for i in range(iterations):

            # Update pseudosize
            if homogeneous:
                p_size,_ = approx_psize_homogeneous(G, test.order, test.h, test.t)
            else:
                p_size,_ = approx_psize_heterogeneous(G, test.order, test.t)

            s = psize_to_speed(p_size)

            # Construct new ordering
            test = Mod_ETF(G, w, s, num_machines, tie_breaking_rule, plot=verbose)

        return naive_cost1, naive_cost2, test.obj_value, test.order
 

def compute_cost(w, t, s):
    total_cost = 0
    for j in range(len(s)):
        total_cost += (t[j][1] + (w[j] * s[j]))
    return total_cost

