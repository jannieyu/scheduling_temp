#from gurobipy import *
import networkx as nx
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from conjecture.all_valid_orderings import *
from conjecture.optimization_functions import *


class Test_old:

    def __init__ (self, \
        task_units, \
        task_transfer_units, \
        machine_speeds, \
        machine_comm_bandwidths, \
        tie_breaking_rule, \
        is_etf=True):
        """
        task_units: Length n list of computation units for each task.
        task_transfer_units: Dictionary containing the dependencies and linkages
            between all specified tasks. Edges that are used to construct the DAG.
        machine_speeds: Length m list of speeds for all given machines
        machine_comm_bandwidths: m x m matrix of communication times between each machine.
        tie_breaking_rule: Integer to specify which way to break ties.
            0: random
        is_etf: Flag to toggle from GETF to pure ETF
        """

        self.tie_breaking_rule = tie_breaking_rule
        self.machine_speeds = machine_speeds
        self.machine_comm_bandwidths = machine_comm_bandwidths
        self.task_units = task_units

        # As given in the paper, n: number of tasks (i.e. nodes)
        self.n = len(self.task_units)

        # m: Number of machines
        self.m = len(self.machine_speeds)

        self.remaining_nodes_list = [0 for _ in range(self.n)]

        # Build DAG called G that contains task computation and info transfer
        self.constructGraph(task_transfer_units)

        # Construct group assignment f
        self.group_assignment(is_etf)

        # Run getf on the data
        self.h, self.t, self.task_process_time, self.machine_task_list = self.getf()

        self.permutation = self.get_permutation()

        self.task_prev = self.construct_prev_link()


        self.cost = self.compute_cost()


        self.time_chunks = self.make_time_chunks()

        

    def power(self, speed):
        return speed ** 2

    def compute_cost(self):
       
        total_cost = 0

        for m in range(self.m):

            task_list = self.machine_task_list[m]
            for t in task_list:

                total_cost += self.power(self.machine_speeds[m])
                total_cost += self.t[t][1]

        return total_cost

    def construct_prev_link(self):

        prev_dict = {}
        for task in self.permutation:

            # search for task among machines
            for m in range(self.m):
                task_list = self.machine_task_list[m]
                for i in range(len(task_list)):
                    if task == task_list[i]:

                        # check if it is first one in machine
                        if i == 0:
                            prev_dict[task] = None
                        else:
                            prev_dict[task] = task_list[i-1]

        return prev_dict

    def make_time_chunks(self):
       
       
        # multi-dimensional data 
        machine_data = [[] for _ in range(self.m)]
        machine_labels = [[] for _ in range(self.m)]

        for m in range(self.m):
            machine_etfd = 0
            task_list = self.machine_task_list[m]
            for t in task_list:
                if machine_etfd < self.t[t][0]:
                    idle_time = self.t[t][0] - machine_etfd

                    for i in range(int(idle_time)):
                        machine_data[m].append(idle_time)
                        machine_labels[m].append('idle')


                process_time = self.task_process_time[t]
                machine_etfd = self.t[t][1]
                machine_data[m].append(process_time)
                machine_labels[m].append(t)

        segments = max([len(task_list) for task_list in machine_data])

        for i in range(len(machine_data)):
            for j in range(len(machine_data[i]), segments):
                machine_data[i].append(0)
                machine_labels[i].append('idle')


        return machine_labels
        


    def make_assignment_visual(self):
       
       
        # multi-dimensional data 
        machine_data = [[] for _ in range(self.m)]
        machine_labels = [[] for _ in range(self.m)]

        for m in range(self.m):
            machine_etfd = 0
            task_list = self.machine_task_list[m]
            for t in task_list:
                if machine_etfd < self.t[t][0]:
                    idle_time = self.t[t][0] - machine_etfd
                    machine_data[m].append(idle_time)
                    machine_labels[m].append('idle')
                process_time = self.task_process_time[t]
                machine_etfd = self.t[t][1]
                machine_data[m].append(process_time)
                machine_labels[m].append(str(t))

        segments = max([len(task_list) for task_list in machine_data])

        for i in range(len(machine_data)):
            for j in range(len(machine_data[i]), segments):
                machine_data[i].append(0)
                machine_labels[i].append('idle')

        data = []
        for s in range(segments):
            section = [machine_data[j][s] for j in range(self.m)]
            data.append(section)

        y_pos = np.arange(self.m)

        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)

        colors ='yg'
        patch_handles = []
        # left alignment of data task_starts at zero
        left = np.zeros(self.m) 
        for i, d in enumerate(data):
            patch_handles.append(ax.barh(y_pos, d, 
              color=colors[i%len(colors)], align='center', 
              left=left))
            left += d

        # search all of the bar segments and annotate
        for j in range(len(patch_handles)):
            for i, patch in enumerate(patch_handles[j].get_children()):
                bl = patch.get_xy()
                x = 0.5*patch.get_width() + bl[0]
                y = 0.5*patch.get_height() + bl[1]
                if j == len(patch_handles):
                    if machine_labels[i][j] == 'idle':
                        ax.text(x,y, '', ha='center')
                else:
                    ax.text(x,y, machine_labels[i][j], ha='center')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(np.arange(self.m))
        ax.set_xlabel('Time')
        plt.show()


    def make_graph_visual(self):
        G = self.G
        labels = {}

        for i in range(0, self.n):
            labels[i] = str(i)

        pos=nx.planar_layout(G)
        nx.draw(G, pos, nodecolor='y',edge_color='k')
        nx.draw_networkx_labels(G, pos, labels, font_size=20, font_color='y')
        plt.axis('off')
        
        plt.show()


    def constructGraph(self, task_transfer_units):
        """
        Builds a directed graph in networkx with the given inputs

        task_computation_times: Vector of n nodes (task) and their computation times.
            Think of this as the computation time of each task
            TODO: Change to be computation power
        task_transfer_units: Dictionary of up to n(n-1)/2 edges and their weights
            Denotes the dependencies between various tasks
        """

        self.G = nx.DiGraph()

        # Add all nodes. This accounts for lone nodes with no dependencies or children.
        for i in range(self.n):
            self.G.add_node(i)



        # Iterate through the edge weight dictionary and add weights to digraph
        for nodes, weight in task_transfer_units.items():
            # Ensure that source and desk are tasks numbered from - to n-1
            source, dest = nodes
            self.G.add_weighted_edges_from([(source, dest, weight)])

        for node in list(self.G.nodes):
            eff_dependency = self.task_units[node]
            for d in nx.algorithms.dag.descendants(self.G, node):
                eff_dependency += self.task_units[d]

            self.remaining_nodes_list[node] = eff_dependency

        # Assert that graph is directed and acyclic
        if not nx.algorithms.dag.is_directed_acyclic_graph(self.G):
            print("Graph is not directed and acyclic!!!")


    def group_assignment(self, is_etf):
        """
        is_etf: If true, than all machines assigned to same group

        Returns: f: vector of size n representing the group of machines that
            task i can be assigned to
        """
        self.f = [set() for i in range(self.n)]

        # In the ETF case, all machines assigned to the same group
        # So the set of feasible machines for each task is everything
        if is_etf:
            for j in range(self.n):
                self.f[j] = range(self.m)
            return

        # Group assignment of all machines
        # -1: Machine is discarded
        machine_assignments = np.zeros(self.m)

        # Build the K groups of machines

        K = 1 # In case of 1 machine, there is only one group
        if self.m > 2:
            alpha = math.log(self.m, 2) / math.log(math.log(self.m, 2), 2)
            if alpha > 0:
                K = int(math.ceil(math.log(self.m, alpha)))
                if K < 1:
                    K = 1

            print("For m = " + str(self.m) + ": alpha = " + str(alpha) + ", K = " + str(K))

        # Assign a machine to a group out of K
        machine_groups = [set() for i in range(K)]
        machine_group_speeds = np.zeros(K)

        if K == 1:
            machine_assignments[0] = 0
            machine_group_speeds[0] += self.machine_speeds[0]

            for i in range(self.m):
                machine_groups[0].add(i)

        else:
            for i in range(self.m):
                # Discard the worst machines from being considered
                # TODO: Are we assuming that all machines have fractional speed????
                if self.machine_speeds[i] < 1.0/self.m:
                    machine_assignments[i] = -1
                else:
                    # TODO: Replace with Binary search to find the result
                    # Place machine in group
                    for k in range(K):
                        if self.machine_speeds[i] >= alpha ** k and  \
                            self.machine_speeds[i] < alpha ** (k+1):
                            machine_assignments[i] = k
                            machine_groups[k].add(i)
                            machine_group_speeds[k] += self.machine_speeds[i]
                            break
                        if k == K-1 and self.machine_speeds[i] == alpha ** (k+1):
                            machine_assignments[i] = k
                            machine_groups[k].add(i)
                            machine_group_speeds[k] += self.machine_speeds[i]
                            break

            print(machine_groups)
        # Formulate a LP and solve it
        success = False
        try:
            # Create a new model
            lp = Model("lp1")

            # x[j, :] is probability distribution over all machines
            # of the probability that task j will appear on machine i
            x = lp.addVars(self.n, self.m, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="x")

            # Completion time for each task j
            c = lp.addVars(self.n, lb=0, vtype=GRB.CONTINUOUS, name="c")

            # Objective T
            T = lp.addVar(lb=0, vtype=GRB.CONTINUOUS, name="T")
            lp.setObjective(T, GRB.MINIMIZE)

            # Constraint 1a: For all tasks j = 0...n-1,
            # it must only be assigned to 1 machine (i.e. dist over i in m must sum to 1)
            a1 = lp.addConstrs((quicksum(x[j, i]  \
                for i in range(self.m)) == 1  \
                for j in range(self.n)), "a1")

            # Constraint 1b:
            b1 = lp.addConstrs((self.task_units[j] *  \
                quicksum(x[j, i] / self.machine_speeds[i]  \
                for i in range(self.m)) <= c[j]  \
                for j in range(self.n)), "b1")

            # Constraint 1c:
            c1 = lp.addConstrs((c[p] + self.task_units[j] *  \
                quicksum(x[j, i] / self.machine_speeds[i]  \
                for i in range(self.m)) <= c[j]  \
                for j in range(self.n) \
                for p in set(self.G.predecessors(j))), "c1")

            # Constraint 1d:
            d1 = lp.addConstrs((quicksum(self.task_units[j] * x[j, i]  \
                for j in range(self.n)) / self.machine_speeds[i] <= T  \
                for i in range(self.m)), "d1")

            # Constraint 1e:
            e1 = lp.addConstrs((c[j] <= T for j in range(self.n)), "e1")

            lp.optimize()

            optT = lp.objVal

            # Get the optimal probability distribution x
            x = np.zeros((self.n, self.m))
            for j in range(self.n):
                for i in range(self.m):
                    varName = "x[" + str(j) + "," + str(i) + "]"
                    x[j][i] = lp.getVarByName(varName).X

            #print(x)
            # --------
            # Part 2:
            # --------

            # Compute the probability distribution that
            # a given task j is in a particular machine group k
            xStar = np.zeros((self.n, K))
            for j in range(self.n):
                for k in range(K):
                    probTaskInGroup = 0.0
                    for i in machine_groups[k]:
                        probTaskInGroup += x[j][i]

                    xStar[j][k] = probTaskInGroup
            #print(xStar)

            # For each task get the largest group index
            # For which more than half of the task is assigned
            # to groups l ... K
            l = [0 for j in range(self.n)]
            for j in range(self.n):

                # Iterate backwards until we reach an index k
                # where the sum from k to K is at least 1/2
                sumlj = 0
                for k in range(K-1, -1, -1):
                    sumlj += xStar[j][k]
                    # TODO: Include parameter to change this threshold
                    if sumlj >= 1/2:
                        l[j] = k
                        break

                # Each task j assigned to group f(j)
                # that maximizes total speed of machines from l[j] ... K
                maxSpeed = 0
                maxGroup = 0
                for k in range(l[j], K, 1):
                    if machine_group_speeds[k] > maxSpeed:
                        maxGroup = k
                        maxSpeed = machine_group_speeds[k]

                # Assign the max group to the task
                self.f[j] = machine_groups[maxGroup]

            success = True
            print("Machine Possibilities for Each Task: ", self.f)

        except GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))

        except AttributeError:
            print('Encountered an attribute error')


        if not success:
            print("Something went wrong")

    def break_tie(self, B):
        """
        Breaks the tie based on the desired tie_breaking_rule input
        B: set of tasks to choose to schedule from
            Each element of b is a tuple consisting of task, task_starting time, machine
        tie_breaking_rule: How we choose to break the ties
            0: random
            1: highest machine speed
            2: lowest task length

        Returns: A task b to schedule
        """

        if len(B) == 0:
            print("Error")

        # Handle base case when there is only one item
        if len(B) == 1:
            return B[0]

        if self.tie_breaking_rule == 0:
            i = random.randint(0, len(B)-1)
            return B[i]

        if self.tie_breaking_rule == 1:
            maxI = 0
            maxSpeed = 0
            for i in range(len(B)):
                _, _, mi = B[i]
                if mi > maxSpeed:
                    maxSpeed = mi
                    maxI = i
            return B[maxI]

        if self.tie_breaking_rule == 2:
            minI = -1
            minLength = 1000000
            for i in range(len(B)):
                ti, _, _ = B[i]
                if self.task_units[ti] < minLength:
                    minLength = self.task_units[ti]
                    minI = i
            return B[minI]

        return B[0]


    def getf(self):
        """
        f: Group Assignment Rule. Vector of size n
        tie_breaking_rule: Integer detemining which method to break tie
        """

        # Set of done tasks
        done = []

        # Initialize list of machine assignments for each task
        h = [0 for _ in range(self.n)] 

        # Initialize task_start and end time for each task
        t = [[0, 0] for _ in range(self.n)] 

        # Initialize earliest time each machine is free
        machine_etf = [0 for i in range(self.m)]

        # Initialize total processing/run time for each task
        task_process_time = [0 for _ in range(self.n)]

        # Initialize constrained task_start times for each task
        task_etf = [None for _ in range(self.n)]

        # Initialize task list for each machine
        machine_task_list = [[] for i in range(self.m)]

        # Append all ready tasks and update constrained task_start times for each task
        A = []
        for j in range(self.n):
            free = True
            for parent in self.G.predecessors(j):
                if parent not in done:
                    free = False
                    break
            if free:
                A.append(j)
                task_etf[j] = 0

        # While all tasks are not "done" 
        while len(done) < self.n:

            max_remaining = 0
            P = []
            for node in A:
                if self.remaining_nodes_list[node] > max_remaining:
                    max_remaining = self.remaining_nodes_list[node]
                    P = [node]
                elif self.remaining_nodes_list[node] == max_remaining:
                    P.append(node)

            num_children = {}
            children = []
            for node in P:
                num_children[node] = len(list(self.G.successors(node)))
                children.append(len(list(self.G.successors(node))))
    
            max_children = max(children)

            P_prime = [key for key in num_children if num_children[key] == max_children]
    

        
            # picked_n = random.choice(P_prime)
            picked_m = None


            possible_tasks = []
            global_earliest_time = np.inf
            for j in P_prime:
                
                # Earliest task_starting time for a given task over all possible machines
                earliest_time = np.inf
                picked_m = 0

                # for task j, find machine that gives earliest start time
                for m in range(self.m):

                    if len(machine_task_list[m]) != 0:
                        last_node = machine_task_list[len(machine_task_list) - 1]
                        if last_node in self.G.predecessors(j):
                            picked_m = m
                            break

                    machine_task_start = machine_etf[m]
                    task_start = task_etf[j]

                    if task_start > machine_task_start:
                        machine_task_start = task_start

                    # Check if this task_start time for a task on a machine is
                    # a minimum across the feasible machines
                    if machine_task_start < earliest_time:
                        earliest_time = machine_task_start
                        picked_m = m

                # print("Prospesctive -- Task " + str(j) + " task_starts at " + str(earliest_time) + " on machine " + str(earliest_machine))
                possible_tasks.append([j, earliest_time, picked_m])

                # Check/update earliest start time among tasks j in A
                if earliest_time < global_earliest_time:
                    global_earliest_time = earliest_time


            # Get list of all tasks that start at global_earliest_time
            B = []
            for i in possible_tasks:
                j, tj, mj = i
                if tj == global_earliest_time:
                    B.append([j, tj, mj])

            # Use tie_breaking_rule to determine the task to schedule
            b = self.break_tie(B)
            picked_n, machine_task_start, picked_m = b
            picked_m = int(picked_m)


            # Update machine assignment for task j
            h[picked_n] = picked_m

            # Update task run/process time for task j
            task_process_time[picked_n] = float(self.task_units[picked_n]) / float(self.machine_speeds[picked_m])
            t[picked_n][0] = machine_task_start
            t[picked_n][1] = machine_task_start + task_process_time[picked_n]
            machine_etf[picked_m] = machine_task_start + task_process_time[picked_n]
            
            done.append(picked_n)

            machine_task_list[picked_m].append(picked_n)

            A.remove(picked_n)

            for child in self.G.successors(picked_n):
                if child not in done:
                    ready = True
                    for parent in self.G.predecessors(child):
                        if parent not in done:
                            ready = False
                            break
                    if ready:
                        A.append(child)
                        task_etf[child] = max([t[i][1] for i in self.G.predecessors(child)])
                    
    
        return h, t, task_process_time, machine_task_list


    def get_permutation(self, machine=1):
        """
        f: Group Assignment Rule. Vector of size n
        tie_breaking_rule: Integer detemining which method to break tie
        """

        # Set of done tasks
        done = []

        # Initialize list of machine assignments for each task
        h = [0 for _ in range(self.n)] 

        # Initialize task_start and end time for each task
        t = [[0, 0] for _ in range(self.n)] 

        # Initialize earliest time each machine is free
        machine_etf = [0 for i in range(machine)]

        # Initialize total processing/run time for each task
        task_process_time = [0 for _ in range(self.n)]

        # Initialize constrained task_start times for each task
        task_etf = [None for _ in range(self.n)]

        # Initialize task list for each machine
        machine_task_list = [[] for i in range(machine)]

        # Append all ready tasks and update constrained task_start times for each task
        A = []
        for j in range(self.n):
            free = True
            for parent in self.G.predecessors(j):
                if parent not in done:
                    free = False
                    break
            if free:
                A.append(j)
                task_etf[j] = 0

        # While all tasks are not "done" 
        while len(done) < self.n:

            max_remaining = 0
            P = []
            for node in A:
                if self.remaining_nodes_list[node] > max_remaining:
                    max_remaining = self.remaining_nodes_list[node]
                    P = [node]
                elif self.remaining_nodes_list[node] == max_remaining:
                    P.append(node)

           


            num_children = {}
            children = []
            for node in P:
                num_children[node] = len(list(self.G.successors(node)))
                children.append(len(list(self.G.successors(node))))
    
            max_children = max(children)

            P_prime = [key for key in num_children if num_children[key] == max_children]
          

            picked_m = None

            possible_tasks = []
            global_earliest_time = np.inf
            for j in P_prime:
                
                # Earliest task_starting time for a given task over all possible machines
                earliest_time = np.inf
                picked_m = 0

                # for task j, find machine that gives earliest start time
                for m in range(machine):

                    if len(machine_task_list[m]) != 0:
                        last_node = machine_task_list[len(machine_task_list) - 1]
                        if last_node in self.G.predecessors(j):
                            picked_m = m
                            break

                    machine_task_start = machine_etf[m]
                    task_start = task_etf[j]

                    if task_start > machine_task_start:
                        machine_task_start = task_start

                    # Check if this task_start time for a task on a machine is
                    # a minimum across the feasible machines
                    if machine_task_start < earliest_time:
                        earliest_time = machine_task_start
                        picked_m = m

                # print("Prospesctive -- Task " + str(j) + " task_starts at " + str(earliest_time) + " on machine " + str(earliest_machine))
                possible_tasks.append([j, earliest_time, picked_m])

                # Check/update earliest start time among tasks j in A
                if earliest_time < global_earliest_time:
                    global_earliest_time = earliest_time


            # Get list of all tasks that start at global_earliest_time
            B = []
            for i in possible_tasks:
                j, tj, mj = i
                if tj == global_earliest_time:
                    B.append([j, tj, mj])

            # Use tie_breaking_rule to determine the task to schedule
            b = self.break_tie(B)
            picked_n, machine_task_start, picked_m = b
            picked_m = int(picked_m)


            # Update machine assignment for task j
            h[picked_n] = picked_m

            # Update task run/process time for task j
            task_process_time[picked_n] = float(self.task_units[picked_n]) / float(self.machine_speeds[picked_m])
            t[picked_n][0] = machine_task_start
            t[picked_n][1] = machine_task_start + task_process_time[picked_n]
            machine_etf[picked_m] = machine_task_start + task_process_time[picked_n]
            
            done.append(picked_n)

            machine_task_list[picked_m].append(picked_n)

            A.remove(picked_n)

            for child in self.G.successors(picked_n):
                if child not in done:
                    ready = True
                    for parent in self.G.predecessors(child):
                        if parent not in done:
                            ready = False
                            break
                    if ready:
                        A.append(child)
                        task_etf[child] = max([t[i][1] for i in self.G.predecessors(child)])
                    
    
        return machine_task_list[0]



if __name__ == "__main__":

    # Process some input data file
    # Givens:
    # task_units: Set of n tasks and their respective computation "units"
    #   they take to complete
    # task_transfer_units: set of edges denoting the dependencies between
    #   various tasks, number of information units needed to be transferred
    # machine_speeds: Set of m machines and their respective computational powers
    # machine_comm_bandwidths: m x m matrix denoting communication bandwidth
    #   i.e. how many information units per second can be transfered between
    #   two different machines
    #   For ETF case, this is inf on diagonal, and 1 on upper triangle

    #

    # Build the graph, done in getf_test.py
    pass