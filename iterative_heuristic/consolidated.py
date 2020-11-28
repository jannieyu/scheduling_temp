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
    #G = rd(num_tasks, 0.05,seed)
    ## print(G.number_of_nodes())
    #w = [random.randint(1, 50) for _ in range(num_tasks)]
    s = [1 for i in range(len(w))]
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
        
    return test_heuristic.obj_value

def iterative_and_naive_heuristic_no_ratio(num_machines, w, G, homogeneous=True, verbose=False):
    s = [1 for i in range(len(w))]
    tie_breaking_rule = 2
    # Get ordering using modified ETF
    test = Mod_ETF(G, w, s, num_machines, tie_breaking_rule, plot=verbose)
    
    # # Initialize objective function value
    # heuristic_opt = test.obj_value

    # # Naive method
    p_size = approx_psize_naive(G, test.order)
    # print(p_size)
    s_prime_naive = psize_to_speed(p_size)
    naive_t = native_rescheduler(G, s_prime_naive, w, copy.deepcopy(test.order))
    naive_cost = compute_cost(w, naive_t, s_prime_naive)
    # #test_naive = Mod_ETF(G, w, s_prime_naive, num_machines, tie_breaking_rule, plot=verbose)
    
    # Heuristic method
    if homogeneous:
        print(test.order)
        p_size,_ = approx_psize_homogeneous(G, test.order, test.h, test.t)
    else:
        p_size,_ = approx_psize_heterogeneous(G, test.order, test.t)
    
    print(p_size)
   
    s_prime_heuristic = psize_to_speed(p_size)
    test_heuristic = Mod_ETF(G, w, s_prime_heuristic, num_machines, tie_breaking_rule, plot=verbose)
    print("-")
    return naive_cost, test_heuristic.obj_value, test_heuristic.order


def compute_cost(w, t, s):
    total_cost = 0
    energy = 0
    mrt = 0
    for j in range(len(s)):
        total_cost += (t[j][1] + (w[j] * s[j]))
    return total_cost

