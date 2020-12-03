import networkx as nx
import numpy as np
import random
import math
import matplotlib.pyplot as plt

from copy import deepcopy
from itertools import zip_longest

def approx_psize_naive(G, order):
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

            p[curr_task] = len(dependencies)

    return p

def approx_psize_heterogeneous(G, order, interval, verbose=True):

    num_tasks = len(G)
    num_machines = len(order)
    psize = [0 for _ in range(num_tasks)]

    for curr_machine in range(num_machines):
        for j in order[curr_machine]:

            overlap_counter = [0 for _ in range(num_machines)]
            overlap_counter[curr_machine] += 1
            concurr_tasks = []
            dependencies = []
            dependencies_count = 0
            # curr_start = float(interval[j][0].__round__(1))
            # curr_end = float(interval[j][1].__round__(1))

            curr_start = interval[j][0]
            curr_end = interval[j][1]

            psize[j] += 1
            for d in list(nx.algorithms.dag.descendants(G, j)):
                if d not in dependencies:
                    dependencies.append(d)
                    dependencies_count += 1

            for m in range(num_machines):
                if m != curr_machine:

                    for other_j in order[m]:

                        other_start = interval[other_j][0]
                        other_end = interval[other_j][1]
                      
                        if not (other_end <= curr_start):
                            if not (curr_end <= other_start):
                                concurr_tasks.append(other_j)
                                end = min(curr_end, other_end)
                                start = max(curr_start, other_start)
                                # psize[j] += (end - start)/ (curr_end - curr_start)
                                psize[j] += 1

                                for d in list(nx.algorithms.dag.descendants(G, other_j)):
                                    if d not in concurr_tasks:
                                        if d not in dependencies:
                                            dependencies.append(d)
                                            dependencies_count += 1

                                if overlap_counter[m] == 0:
                                    overlap_counter[m] = 1

            psize[j] += dependencies_count
            psize[j] /= sum(overlap_counter)
    return psize

def approx_psize_homogeneous(G, order, h, interval, verbose=True):

    num_tasks = len(G)
    num_shared_task_lst = [0 for _ in range(num_tasks)]

    psize = [0 for _ in range(num_tasks)]
    last_on_machine_interval_start = max([interval[i][1] for i in range(num_tasks)])
    interval_group = [[] for _ in range(int(last_on_machine_interval_start))]
    for x in range(len(interval)):
        start, _ = interval[x]
        interval_group[int(start)].append(x)
    # print("Intervals is ", interval_group)
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
                #print("Task is ", task)
                checking_sharing_subset.pop(0)
                sharing_subset.append(task)
                next_dependencies = dependencies_in_next_task_set(G, order, h, interval_group, x1, task)
                #print("next dependencises is ", next_dependencies)
                for d in next_dependencies:
                    if d not in shared_children:
                        shared_children.append(d)
                newly_added_sharing_subset = []
                if len(next_dependencies) != 0:
                    for other_task in curr_task_set_copy:
                        #print("other task is ", other_task)
                        next_other_dependencies = dependencies_in_next_task_set(G, order, h, interval_group, x1, other_task)
                        #print("next other dependencies is ",next_other_dependencies)
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
