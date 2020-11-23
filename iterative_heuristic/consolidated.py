from conjecture.consolidated_functions import *
from conjecture.all_valid_orderings import *
from conjecture.optimization_functions import *
from iterative_heuristic.modified_etf import Mod_ETF
from iterative_heuristic.approximate_speeds import *
from makespan_energy.construct_graph_util import *
from makespan_energy.visualization_util import *
from graph_functions.erdos_renyi_dag import random_dag as rd
import networkx as nx
import numpy as np
import math
import random


def iterative_heuristic(num_tasks, num_machines, seed, verbose=False):
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
        s_prime = approx_speeds(G, test.order)

        # Get ordering using modified ETF
        test2 = Mod_ETF(G, w, s_prime, num_machines, tie_breaking_rule, plot=verbose)
        temp = get_objective_single_ordering(True, G, w, test2.order, plot=verbose, compare=False)
        opt_intervals, s_opt, obj_opt, _ = temp
        if obj_opt!= 10000000:
            did_not_work = True
        seed +=1
    return obj_opt / test2.obj_value


def iterative_heuristic_no_ratio(num_machines, w, G, verbose=False):
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
    s_prime = approx_speeds(G, test.order)

    # Get ordering using modified ETF
    test2 = Mod_ETF(G, w, s_prime, num_machines, tie_breaking_rule, plot=verbose)
    temp = get_objective_single_ordering(True, G, w, test2.order, plot=verbose, compare=False)
    opt_intervals, s_opt, obj_opt, _ = temp
    if obj_opt!= 10000000:
        return test2.obj_value / obj_opt
    else:
        return 10000000


def iterative_heuristic_no_ratio(num_machines, w, G, verbose=False):
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
    s_prime = approx_speeds(G, test.order)

    # Get ordering using modified ETF
    test2 = Mod_ETF(G, w, s_prime, num_machines, tie_breaking_rule, plot=verbose)
        
    return test2.obj_value


def iterative_and_naive_heuristic_no_ratio(num_machines, w, G, verbose=False):
    s = [1 for i in range(len(w))]
    tie_breaking_rule = 2
    # Get ordering using modified ETF
    test = Mod_ETF(G, w, s, num_machines, tie_breaking_rule, plot=verbose)
    
    #     # Initialize objective function value
    heuristic_opt = test.obj_value

    #     while True:
    # Get pseudosize, convert to speed
    p_size = approx_psize(G, test.order, test.t)
    s_prime = [np.sqrt(p) for p in p_size]
    naive_objective = compute_cost(w, test.t, s_prime)
    # Get ordering using modified ETF
    test2 = Mod_ETF(G, w, s_prime, num_machines, tie_breaking_rule, plot=verbose)
    return naive_objective, test2.obj_value, test2.order


def compute_cost(w, t, s):
    total_cost = 0
    energy = 0
    mrt = 0
    for j in range(len(s)):
        total_cost += (t[j][1] + (w[j] * s[j]))
    return total_cost

