import networkx as nx
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import zip_longest
import heapq
from collections import defaultdict

def approx_psize_naive1(G, order):
    '''
    Naively compute pseudosize to be proportional to the number of tasks on its
    machine only.
    '''
    num_tasks = len(G)
    num_machines = len(order)
    p = [0 for _ in range(num_tasks)]

    for i in range(num_machines):
        for j in range(len(order[i])):

            curr_task = order[i][j]
            dependencies = []

            for k in range(j, len(order[i])):
                dependencies.append(order[i][k])
 
            # Commented this out because in the naive method, we only consider the p-size 
            # as proportional to the number of tasks on its machine only.

            # for d in list(nx.algorithms.dag.descendants(G, curr_task)):
            #     if d not in dependencies:
            #         dependencies.append(d)
            assert(len(dependencies) == len(order[i]) - j)
            p[curr_task] = len(dependencies)

    return p

def approx_psize_homogeneous(G, order, h, interval, verbose=True):

    num_tasks = len(G)
    num_shared_task_lst = [0 for _ in range(num_tasks)]

    psize = [0 for _ in range(num_tasks)]
    last_on_machine_interval_start = max([interval[i][1] for i in range(num_tasks)])
    interval_group = [[] for _ in range(int(last_on_machine_interval_start))]
    for x in range(len(interval)):
        start, _ = interval[x]
        interval_group[int(start)].append(x)
   
    for x1 in range(len(interval_group) - 1, -1, -1):
        curr_task_set = interval_group[x1]
        curr_task_set_copy = curr_task_set.copy()

        while curr_task_set_copy != []:
           
            shared_children = []
            sharing_subset = []
            task = curr_task_set_copy[0]
            curr_task_set_copy.pop(0)
            checking_sharing_subset = [task]

            while checking_sharing_subset != []:
                task = checking_sharing_subset[0]
                
                checking_sharing_subset.pop(0)
                sharing_subset.append(task)
                next_dependencies = dependencies_in_next_task_set(G, order, h, interval_group, x1, task)
               
                for d in next_dependencies:
                    if d not in shared_children:
                        shared_children.append(d)
                newly_added_sharing_subset = []
                if len(next_dependencies) != 0:
                    for other_task in curr_task_set_copy:
                        
                        next_other_dependencies = dependencies_in_next_task_set(G, order, h, interval_group, x1, other_task)
                        
                        for other_d in next_other_dependencies:
                            if other_d in next_dependencies:
                                if other_task not in checking_sharing_subset:
                                    checking_sharing_subset.append(other_task)
                                    newly_added_sharing_subset.append(other_task)
                                if other_d not in shared_children:
                                    shared_children.append(other_d)
                for j in newly_added_sharing_subset:
                    curr_task_set_copy.remove(j)

            if len(sharing_subset) == 1:
                task = sharing_subset[0]
                for child in shared_children:
                    psize[task] += psize[child]

                psize[task] += 1
            else:
                final_psize = 0
                for child in shared_children:
                    final_psize += psize[child]
                final_psize = (final_psize + len(sharing_subset)) / len(sharing_subset)
                for share_task in sharing_subset:
                    psize[share_task] = final_psize
            for j in sharing_subset:
                num_shared_task_lst[j] = len(sharing_subset)

    return psize, num_shared_task_lst



def last_on_machine_checker(order, h, task):
    # define next task on same machine, if possible

    task_machine = h[task]

    last_on_machine = False
    next_task = None
    if order[task_machine].index(task) == len(order[task_machine]) - 1:
        last_on_machine = True
    else:
        next_task = order[task_machine][order[task_machine].index(task) + 1]
    return last_on_machine, next_task



def dependencies_in_next_task_set(G, order, h, interval_group, x1, task):
    # direct children in next interval task set and task that runs after current task on same machine

    if x1 != len(interval_group) - 1:
        next_task_set = interval_group[x1 + 1]
    else:
        next_task_set = []

    task_machine = h[task]
    task_children = list(nx.algorithms.dag.descendants(G, task))

    last_on_machine, next_task = last_on_machine_checker(order, h, task)

    # create a list of task children that run in the next interval x1 + 1
    dependencies = [child for child in task_children if child in next_task_set]

    # if next task on same machine runs during the next time interval
    # and isn't already in list of task children, then put it in
    if not last_on_machine:
        if next_task in next_task_set:
            if next_task not in dependencies:
                dependencies.append(next_task)


    return dependencies

def length_of_longest_chain(G, node):
    max_length = 0
    for s in list(G.successors(node)):
        length = 1 + length_of_longest_chain(G, s)
        if length > max_length:
            max_length = length

    return max_length


def remaining_on_own_machine(num_tasks, order):
    remaining_lst = [[j] for j in range(num_tasks)]
    for lst in order:
        for j in range(len(lst) - 1, -1, -1):
            if j != len(lst) - 1:
                curr_task = lst[j]
                next_task = lst[j+1]
                remaining_lst[curr_task].extend(remaining_lst[next_task])

    return remaining_lst

def ub_lst(G, order):
#     print("In ub_list")
    num_tasks = len(G)
    descendant_lst = [list(nx.algorithms.dag.descendants(G, task)) for task in range(num_tasks)]
    remaining = remaining_on_own_machine(num_tasks, order)
    lst = [None for _ in range(num_tasks)]
    for j in range(num_tasks):
        lst[j] = remaining[j]
        lst[j].extend([d for d in descendant_lst[j] if d not in lst[j]])

    return [len(l) for l in lst]


def ub_lst_graph(G, order):
    num_tasks = len(G)
    remaining = remaining_on_own_machine(num_tasks, order)
    G_copy = deepcopy(G)
    for machine in order:
        for machine_index in range(len(machine)):
            task = machine[machine_index]
            # If not last element of each machine
            if task != machine[-1]:
                next_task = machine[machine_index + 1]
                if next_task not in nx.algorithms.descendants(G_copy, task):
                    G_copy.add_edge(task, next_task)

    descendant_lst = [list(nx.algorithms.dag.descendants(G_copy, task)) for task in range(num_tasks)]
    return [len(l)+1 for l in descendant_lst]


def getDuplicatesWithInfo(listOfElems):
    ''' Get duplicate element in a list along with thier indices in list
     and frequency count'''
    dictOfElems = dict()
    index = 0
    # Iterate over each element in list and keep track of index
    for elem in listOfElems:
        # If element exists in dict then keep its index in lisr & increment its frequency
        if elem in dictOfElems:
            dictOfElems[elem][0] += 1
            dictOfElems[elem][1].append(index)
        else:
            # Add a new entry in dictionary
            dictOfElems[elem] = [1, [index]]
        index += 1

    dictOfElems = { key:value for key, value in dictOfElems.items() if value[0] >= 1}
    return dictOfElems


def ub_lst_max(G,order, intervals):
    intervals = [str(list1) for list1 in intervals]
    interval_dict = getDuplicatesWithInfo(intervals)
    intervals = []
    for key, val in interval_dict.items():
        intervals.append(val[1])
    num_tasks = len(G)
    max_len = [len(machine) for machine in order]
    descendant_lst = [list(nx.algorithms.dag.descendants(G, task)) for task in range(num_tasks)]
    remaining = remaining_on_own_machine(num_tasks, order)
    lst = [0 for _ in range(num_tasks)]
    psizes = [0 for _ in range(num_tasks)]
    for interval in intervals:
        ps_temp = []
        for task in interval:
            task = int(task)
            lst[task] = remaining[task]
            lst[task].extend([d for d in descendant_lst[task] if d not in lst[task]])
            psize = len(lst[task])+1
            ps_temp.append(psize)
        max_psize = max(ps_temp)
        for task in interval:
            task = int(task)
            psizes[task] = max_psize
    return psizes


def num_remaining_tasks_on_machine(order, num_tasks):

    psize=[-1]* num_tasks

    for machine in order:
       
        num_tasks_on_machine = len(machine)
        
        for i in range(num_tasks_on_machine):
            curr_task = machine[i]
            psize[curr_task] = num_tasks_on_machine - i
            
    return psize_to_speed(psize)

def psize_to_speed(psize):
    s = [np.sqrt(psize[i])for i in range(len(psize))]
    return s

def speed_to_psize(speed):
    psize = [speed[i] ** 2 for i in range(len(speed))]
    return psize


def map_task_to_next_task_on_same_machine(order):
    hashmap = {}
    for task_list in order:
        for i in range(len(task_list)):
            task = task_list[i]
            if i != len(task_list) - 1:
                next_task = task_list[i+1]
                hashmap[task] = next_task
            else:
                hashmap[task] = None
        

    return hashmap

def finish_time_groups(interval):
    hashmap = defaultdict(list)
    for i in range(len(interval)):
        start, end = interval[i]
        hashmap[end].append(i)
    return hashmap

def find_immediate_dependents(G, order, interval, next_task_map):
    num_tasks = len(interval)
    hashmap = {}
    for task in range(num_tasks):
        dependents = set()
        next_task_on_same_machine = next_task_map[task]
        if next_task_on_same_machine and interval[next_task_on_same_machine][0] == interval[task][1]:
            dependents.add(next_task_on_same_machine)

        for d in nx.descendants(G, task):
            if interval[d][0] == interval[task][1]:
                dependents.add(d)
        hashmap[task] = dependents
    return hashmap


def approx_psize_general(G, order, interval, verbose=True):

    num_tasks = len(G)
    psize = [None for _ in range(num_tasks)]
    scheduled = [False for _ in range(num_tasks)]
    unscheduled_tasks = []
    next_task_map = map_task_to_next_task_on_same_machine(order)
    time_to_tasks_map = finish_time_groups(interval)
    task_to_dependents_map = find_immediate_dependents(G, order, interval, next_task_map)

    for i in range(num_tasks):
        unscheduled_tasks.append((- interval[i][1], i))
    heapq.heapify(unscheduled_tasks)
    
    while unscheduled_tasks:
        _, task = heapq.heappop(unscheduled_tasks)
    
        if not scheduled[task]:
            
            parents = set([task])
            all_dependents = task_to_dependents_map[task]

            assert(task in time_to_tasks_map[interval[task][1]])

            for concurr_task in time_to_tasks_map[interval[task][1]]:
                if concurr_task != task:
                    valid_parent = False
                    concurr_task_dependents = task_to_dependents_map[concurr_task]
                    for d in concurr_task_dependents:
                        if d in all_dependents:
                            valid_parent = True
                            break
                    if valid_parent:
                        parents.add(concurr_task)
                        for d in concurr_task_dependents:
                            all_dependents.add(d)
            assert(all([interval[c][0] == interval[task][1] for c in all_dependents]) is True)
            assert(None not in [psize[c] for c in all_dependents])

            for p in parents:
                psize[p] = (sum([psize[c] for c in all_dependents]) + len(parents)) / len(parents)
                scheduled[p] = True

    return psize

