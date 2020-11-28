import networkx as nx
import numpy as np
import random
import math
import matplotlib.pyplot as plt


def native_rescheduler(G, s, w, order):
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

            
                
