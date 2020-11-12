import networkx as nx
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from conjecture.all_valid_orderings import *
from conjecture.optimization_functions import *
from copy import deepcopy

def approx_speeds(G, order):
	num_tasks = len(G)
	num_machines = len(order)
	s = [0 for _ in range(num_tasks)]

	for i in range(num_machines):
		for j in range(len(order[i])):

			curr_task = order[i][j]
			dependencies = []

			for k in range(j, len(order[i])):
				dependencies.append(order[i][k])

			for d in list(nx.algorithms.dag.descendants(G, curr_task)):
				if d not in dependencies:
					dependencies.append(d)


			s[curr_task] = np.sqrt(len(dependencies))


	return s

def approx_psize(G, order, interval, verbose=True):

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
									if d not in concurr_tasks:
										if d not in dependencies:
											dependencies.append(d)
											dependencies_count += 1



			for m in range(num_machines):
				if m != curr_machine:

					for other_j in order[m]:

						other_start = interval[other_j][0]
						other_end = interval[other_j][1]
						# other_start = float(interval[other_j][0].__round__(1))
						# other_end = float(interval[other_j][1].__round__(1))


						if not (other_end <= curr_start):
							if not (curr_end <= other_start):
								concurr_tasks.append(other_j)
								end = min(curr_end, other_end)
								start = max(curr_start, other_start)
								psize[j] += (end - start)/ (curr_end - curr_start)

								for d in list(nx.algorithms.dag.descendants(G, other_j)):
									if d not in concurr_tasks:
										if d not in dependencies:
											dependencies.append(d)
											dependencies_count += 1

								if overlap_counter[m] == 0:
									overlap_counter[m] = 1

			if verbose:
				print("---")
				print("current task " + str(j))
				print("pre-speed " + str(psize[j]))
				print("Dependencies " + str(dependencies))
				print("overlap counter " + str(overlap_counter))
				print("concurr tasks" + str(concurr_tasks))

			psize[j] += dependencies_count
			psize[j] /= sum(overlap_counter)
	return psize




def approx_psize_homogeneous(G, order, h, interval, verbose=True):

	num_tasks = len(G)
	num_machines = len(order)
	num_shared_task_lst = [0 for _ in range(num_tasks)]


	psize = [0 for _ in range(num_tasks)]
	last_on_machine_interval_start = max([interval[i][1] for i in range(num_tasks)])
	interval_group = [[] for _ in range(int(last_on_machine_interval_start))]
	for x in range(len(interval)):
		start, end = interval[x]
		interval_group[int(start)].append(x)

	for x1 in range(len(interval_group) - 1, -1, -1):
		curr_task_set = interval_group[x1]
		curr_task_set_copy = curr_task_set.copy()


		for t in curr_task_set:
			next_dependencies = dependencies_in_next_task_set(G, order, h, interval_group, x1, t)

			if len(next_dependencies) == 1 and h[next_dependencies[0]] == h[t]:
				curr_task_set_copy.remove(t)
				psize[t] = psize[next_dependencies[0]] + 1


		while curr_task_set_copy != []:
			shared_children = []
			sharing_subset = []
			task = curr_task_set_copy[0]
			curr_task_set_copy.pop(0)
			key_task = task
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


			if verbose:
				print("---")
				print("task subset: "  + str(sharing_subset))
				print("shared children: " + str(shared_children))
				print("remaining in task set: "  + str(curr_task_set_copy))




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


	return psize, interval_group



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
	num_tasks = len(G)
	descendant_lst = [list(nx.algorithms.dag.descendants(G, task)) for task in range(num_tasks)]
	remaining = remaining_on_own_machine(num_tasks, order)

	# print("d list: " + str(descendant_lst))
	# print("remaining: " + str(remaining))

	lst = [None for _ in range(num_tasks)]
	for j in range(num_tasks):
		lst[j] = remaining[j]
		lst[j].extend([d for d in descendant_lst[j] if d not in lst[j]])

	return [len(l) for l in lst]




def lb_lst(G, num_machines, order):
	num_tasks = len(G)
	remaining_lst = remaining_on_own_machine(num_tasks, order)
	# return [(length_of_longest_chain(G, i) + num_shared_task_lst[i]) / num_shared_task_lst[i] for i in range(num_tasks)]
	descendant_lst_plus_itself = [max(1, (len(remaining_lst[task])) / num_machines) for task in range(num_tasks)]

	return descendant_lst_plus_itself

def lb_lst_precise(G, num_machines, order, num_concurrent_running_tasks_lst):
	num_tasks = len(G)
	remaining_lst = remaining_on_own_machine(num_tasks, order)
	# return [(length_of_longest_chain(G, i) + num_shared_task_lst[i]) / num_shared_task_lst[i] for i in range(num_tasks)]
	descendant_lst_plus_itself = [max(1, (len(remaining_lst[task])) / num_concurrent_running_tasks_lst[task]) for task in range(num_tasks)]

	return descendant_lst_plus_itself
	
def num_concurrent_running_tasks(num_tasks, interval_group):
	lst = [None for i in range(num_tasks)]

	for group in interval_group:
		for task in group:
			lst[task] = len(group)

	return lst


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

def psize_to_speed(psize):
	s = [np.sqrt(psize[i])for i in range(len(psize))]
	return s

def speed_to_psize(speed):
	psize = [speed[i] ** 2 for i in range(len(speed))]
	return psize
