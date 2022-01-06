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

def naive_2(G, num_machines):
    
    psize = [len(nx.algorithms.dag.descendants(G, task)) + 1 for task in range(len(G))] 
    s = psize_to_speed(psize)
    w = [1 for _ in range(len(G))]
    tie_breaking_rule = 2

    naive2_etf = Mod_ETF(G, w, s, num_machines, tie_breaking_rule)
    naive2_cost = naive2_etf.obj_value

    return naive2_cost

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
    total_cost = None
    
    w = [1 for _ in range(len(G))]
    s = [1 for _ in range(len(G))]

    tie_breaking_rule = 2

    # Get initial ordering using modified ETF
    etf = Mod_ETF(G, w, s, num_machines, tie_breaking_rule, plot=False)

    # run naive 1
    if naive_version == 1 or naive_version == 3:
        psize = approx_psize_naive1(G, etf.order)
        s_prime_naive = psize_to_speed(psize)
       
        naive_t = native_rescheduler(G, s_prime_naive, w, copy.deepcopy(etf.order))
        naive1_cost, power, energy = compute_cost(w, naive_t, s_prime_naive)
    
    # run naive 2
    if naive_version == 2 or naive_version == 3:   
        psize = [len(nx.algorithms.dag.descendants(G, task)) + 1 for task in range(len(G))] 
        s_prime_naive = psize_to_speed(psize)
        naive2_etf = Mod_ETF(G, w, s_prime_naive, num_machines, tie_breaking_rule, plot=verbose)
       
        naive2_cost = naive2_etf.obj_value
        
    # run iterative heuristic once
    p_size,_ = approx_psize_homogeneous(G, etf.order, etf.h, etf.t)
    s = psize_to_speed(p_size)
    # etf = Mod_ETF(G, w, s, num_machines, tie_breaking_rule, plot=verbose)
    t = native_rescheduler(G, s, w, etf.order)
    total_cost, _, _ = compute_cost(w, t, s)
#     for i in range(iterations -1):
#         s = [1 for _ in range(len(G))]
#         t = get_t(etf.order, G, len(s), s)
#         p_size,_ = approx_psize_homogeneous(G, etf.order, etf.h, t)
#         s = psize_to_speed(p_size)
#         # etf = Mod_ETF(G, w, s, num_machines, tie_breaking_rule, plot=verbose)
#         t = native_rescheduler(G, s, w, etf.order)
#         total_cost, _, _ = compute_cost(w, t, s)
# #         etf2 = Mod_ETF(G, w, s, num_machines, tie_breaking_rule, plot=verbose)
        
        

    return naive1_cost, naive2_cost, total_cost, etf

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


def compute_cost(w, t, s):
    
    power = 0
    energy = 0
    for j in range(len(s)):
        power += w[j] * s[j]
        energy += t[j][1]
    total_cost = power + energy
    return total_cost, power, energy

def native_rescheduler(G, s, w, order):
    '''
    Given fixed ordering and speeds, perform greedy algorithm to obtain correct 
    final time intervals for schedule.
    '''

    machine_earliest_start = [0 for _ in range(len(order))]
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
                if j is not None and j not in processed_tasks:
                    process = False
                    break
                else:
                    last_child_end = max(last_child_end, t[j][1])

            # print(task, list(prev_task_list), process)
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

        # print(machine_to_task_list)

    return t


def general_heuristic(G, num_machines, w, iterations, verbose):
    
    convergence = []
    s = [1 for _ in range(len(G))]
    tie_breaking_rule = 2
    old_pseudosize = []

    for i in range(iterations):
        etf = Mod_ETF(G, w, s, num_machines, tie_breaking_rule, plot=verbose)
        new_pseudosize = approx_psize_general(G, etf.order, etf.t, verbose)

        if old_pseudosize:
            new_pseudosize = (old_pseudosize + new_pseudosize) / 2
            old_pseudosize = new_pseudosize


        s = psize_to_speed(new_pseudosize)

        t = native_rescheduler(G, s, w, etf.order)
        obj_val, time, energy = compute_cost(w, t, s)
        convergence.append(obj_val)

    
    
    return obj_val, time, energy, etf.order, convergence