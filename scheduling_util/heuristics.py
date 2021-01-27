from scheduling_util.consolidated_functions import *
from scheduling_util.optimization_functions import *
from scheduling_util.modified_etf import Mod_ETF
from scheduling_util.approx_pseudosizes import *
from graph_util.visualization_util import *
from graph_util.erdos_renyi_dag import er_dag as rd
import networkx as nx
import numpy as np
import math
import random
import copy


def heuristics(G, num_machines, naive_version=0, iterations=1, verbose=False):
    '''
    Runs both the iterative heuristic and the naive method(s) for generating a 
    schedule given G.
    :param G:
    :param num_machines: number of machines 
    :param w:
    :param naive_version: 
    
    
    If 1, we will return heuristic cost, naive 1 cost
    If 2, we will return heuristic cost, naive 2 cost
    If 3, we will return heuristic cost, naive 1 cost, naive 2 cost
    Otherwise (default), we will return heuristic cost only.

    :param iterations: Set the number of iterations that we run the iterative method
    :param verbose: If True, graphs will be plotted out.
    
    '''

    # extra safety method, should return a None value unless you request for 
    # the method to be run
    naive1_cost = None
    naive2_cost = None
    heuristics = None
    
    w = [1 for _ in range(len(G))]
    s = [1 for _ in range(len(G))]

    tie_breaking_rule = 2

    # Get initial ordering using modified ETF
    etf = Mod_ETF(G, w, s, num_machines, tie_breaking_rule, plot=verbose)

    # run naive 1
    if naive_version == 1 or naive_version == 3:
        psize = approx_psize_naive1(G, etf.order)
        s_prime_naive = psize_to_speed(psize)
       
        naive_t = native_rescheduler(G, s_prime_naive, w, copy.deepcopy(etf.order))
        naive1_cost = compute_cost(w, naive_t, s_prime_naive)
    
    # run naive 2
    if naive_version == 2 or naive_version == 3:   
        psize = [len(nx.algorithms.dag.descendants(G, task))+1 for task in range(len(G))] 
        s_prime_naive = psize_to_speed(psize)
        naive2_etf = Mod_ETF(G, w, s, num_machines, tie_breaking_rule, plot=verbose)
        naive2_cost = naive2_etf.obj_value
        
    # run iterative heuristic
    for i in range(iterations):
        p_size,_ = approx_psize_homogeneous(G, etf.order, etf.h, etf.t)
        s = psize_to_speed(p_size)
        etf = Mod_ETF(G, w, s, num_machines, tie_breaking_rule, plot=verbose)

    return naive1_cost, naive2_cost, etf.obj_value, etf

def compute_cost(w, t, s):
    '''
    Compute the cost of the schedule given weights, time intervals, speed.
    '''
    total_cost = 0
    for j in range(len(s)):
        total_cost += (t[j][1] + (w[j] * s[j]))
    return total_cost

def native_rescheduler(G, s, w, order):
    '''
    Given fixed ordering and speeds, perform greedy algorithm to obtain correct 
    final time intervals for schedule.
    '''

    machine_earliest_start = [0 for i in range(len(order))]
    t = [[0,0] for _ in range(len(s))]
    processed_tasks = set()
    
    machine_to_task_list = {}
    for i, lst in enumerate(order):
        machine_to_task_list[i] = lst

    
    while len(machine_to_task_list) != 0:
        # print(machine_to_task_list, machine_earliest_start)
        machines_to_remove = []
        for machine, task_lst in machine_to_task_list.items():
            
            if len(task_lst) == 0:
                machines_to_remove.append(machine)
                continue

            task = task_lst[0]

            
            prev_task_list = G.predecessors(task)
            last_child_end = 0
            process = True
            for j in prev_task_list:
                if j not in processed_tasks:
                    process = False
                    break
                else:
                    last_child_end = max(last_child_end, t[j][1])


            if process:
                start_time = machine_earliest_start[machine]
                t[task][0] = max(start_time, last_child_end)
                t[task][1] = t[task][0] + w[task]/s[task]
                machine_earliest_start[machine] += w[task]/s[task]
                processed_tasks.add(task)
                task_lst.pop(0)
                if not task_lst:
                    machines_to_remove.append(machine)
        for machine in machines_to_remove:
            machine_to_task_list.pop(machine)

    return t