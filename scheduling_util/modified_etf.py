import networkx as nx
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from scheduling_util.optimization_functions import *


class Mod_ETF:

    def __init__ (self, \
        G, \
        w, \
        s, \
        num_machines, \
        tie_breaking_rule, \
        plot=False, is_etf=True):

        """
        Initialize parameters to run modified ETF.

        :param G: DAG to schedule
        :param w: List of task sizes/weights
        :param s: List of task speeds
        :param num_machines: total number of machines
        :param tie_breaking_rule: tie breaking rule in ETF. See function for
        different options.
        :param plot: Boolean variable; if True, the constructed schedule will
        be plotted.
        :param is_etf: Flag to toggle from GETF to pure ETF. (No need to change
        this for now.. code breaks rn if you set this to be False)
        """
    
        self.G = G
        self.w = w
        self.s = s

        self.num_tasks = len(self.w)
        self.num_machines = num_machines

        self.tie_breaking_rule = tie_breaking_rule

        # Use speeds to define pseudosizes
        self.pseudosize = [0 for _ in range(self.num_tasks)]
        for j in list(self.G.nodes):
            self.pseudosize[j] = (s[j])**2

        # Construct group assignment f
        # In our ETF case, tasks can be assigned to any machine, not just a
        # subset like GETF
        # Reason why we have this set up is because code was modified from
        # simulations.py
        self.group_assignment(is_etf)

        self.task_process_time = [self.w[i] / self.s[i] for i in range(self.num_tasks)]

        # Run modified ETF on the data
        # As done in paper, h maps tasks to machines, t maps tasks to the
        # interval that they run at
        self.h, self.t, self.order = self.etf()

        self.obj_value = self.compute_cost()

        if plot:
            color_palette = [(0, 0, 255 / 256), (0, 255 / 256, 0), (255 / 256, 255 / 256, 0), (255 / 256, 0, 0),
                (255 / 256, 128 / 256, 0),
                (255 / 256, 0, 127 / 256), (0, 255 / 256, 255 / 256), (127 / 256, 0, 255 / 256),
                (128 / 256, 128 / 256, 128 / 256),
                (255 / 256, 255 / 256, 255 / 256), (0, 0, 0)]
            metadata = make_task_metadata(self.order, self.num_tasks, self.t)
            plot_gantt(metadata, self.obj_value, color_palette)



    def power(self, speed):
        """
        Given speed, compute power used.
        :param speed: Speed input
        :return: power
        """
        return speed ** 2

    def compute_cost(self):
        """
        Compute objective function value.
        :return: objective function value
        """
        total_cost = 0
        energy = 0
        mrt = 0
        for j in range(self.num_tasks):
            mrt += self.t[j][1]
            energy += ((self.w[j] * self.power(self.s[j]))/ self.s[j])
            total_cost += (self.t[j][1] + ((self.w[j] * self.power(self.s[j]))/ self.s[j]))

        self.mrt = mrt
        self.energy = energy
        return total_cost


    def group_assignment(self, is_etf):
        """
        Assigns each task to the group of machines that it can be assigned to
        when performing GETF (or mod ETF in our case)

        :param is_etf: If true, than tasks can be assigned to any of the machines
        :return: f as vector of length num_tasks; each entry f[j] holds the
        list of machines that task j can be assigned to.
        """
        self.f = [set() for i in range(self.num_tasks)]

        # In the ETF case, all machines assigned to the same group
        # So the set of feasible machines for each task is everything
        if is_etf:
            for j in range(self.num_tasks):
                self.f[j] = range(self.num_machines)
            return


    def break_tie(self, B, order):
        """
        Breaks the tie based on the desired tie_breaking_rule input
        B: set of tasks to choose to schedule from
            Each element of b is a tuple consisting of task, task_starting time, machine
        tie_breaking_rule: How we choose to break the ties
            0: random
            1: highest machine speed
            2: lowest task size

        Returns: A task b to schedule
        """

        value = None

        if len(B) == 0:
            print("Error")
            return value

        # Handle base case when there is only one item
        if len(B) == 1:
            value = 0


        if self.tie_breaking_rule == 0:
            value = random.randint(0, len(B)-1)


        if self.tie_breaking_rule == 1:
            maxI = 0
            maxSpeed = 0
            for i in range(len(B)):
                _, _, mi = B[i]
                if mi > maxSpeed:
                    maxSpeed = mi
                    maxI = i
            value = maxI

        # In the heuristic, maining working with this tie-breaker
        if self.tie_breaking_rule == 2:
            minI = 0
            minLength = np.infty
            for i in range(len(B)):
                task, _, _ = B[i]
                if self.w[task] < minLength:
                    minLength = self.w[task]
                    minI = i
            value = minI

        star_task, machine_task_start, machine = B[value]

        poss_machines = []

        for b in B:
            j, tj, mj = b

            if j == star_task and tj == machine_task_start:
                poss_machines.append(mj)


        star_machine = None
        for machine in poss_machines:
            if len(order[machine]) > 0:
                if order[machine][len(order[machine]) - 1] in self.G.predecessors(star_task):

                    star_machine = machine
                    break

        if star_machine == None:
            star_machine = min(poss_machines)


        return star_task, machine_task_start, star_machine


    def etf(self):

        """ Use modified ETF to schedule the tasks on the machines.
        """

        # Set of tasks that have already been scheduled
        done = []

        # Initialize list for task/machine assignments; each entry h[j] will
        # hold the corresponding machine that task j is running on
        h = [0 for _ in range(self.num_tasks)]

        # Initialize the time interval in which each task runs on
        t = [[0, 0] for _ in range(self.num_tasks)]

        # Initialize earliest time each machine is free
        machine_etf = [0 for i in range(self.num_machines)]

        # Initialize constrained task_start times for each task
        task_etf = [None for _ in range(self.num_tasks)]

        # Initialize ordering
        order = [[] for i in range(self.num_machines)]

        # Initialize list of available tasks that are ready to run
        A = []

        # Append all available tasks
        for j in range(self.num_tasks):
            free = True
            # Check if parents have been scheduled yet
            for parent in self.G.predecessors(j):
                if parent not in done:
                    free = False
                    break
            if free:
                A.append(j)

                # Update constrained task earliest start times
                task_etf[j] = 0


        while len(done) < self.num_tasks:

            # Construct B as a list of tasks with the largest pseudosizes from A
            max_pseudosize = 0
            B = []
            for node in A:
                if self.pseudosize[node] > max_pseudosize:
                    max_pseudosize = self.pseudosize[node]
                    B = [node]
                elif self.pseudosize[node] == max_pseudosize:
                    B.append(node)

            # Construct B_prime as a list of tasks with most children from B
            num_children = {}
            children = []
            for node in B:
                num_children[node] = len(list(self.G.successors(node)))
                children.append(len(list(self.G.successors(node))))


            if children != []:
                max_children = max(children)
                B_prime = [key for key in num_children if num_children[key] == max_children]
            else:
                B_prime = B
            # Chosen machine for scheduling
            star_machine = None

            # List of possible tasks for scheduling
            possible_tasks = []
            global_earliest_time = np.inf
            for j in B_prime:

                # Earliest task_starting time for task j
                earliest_time = np.inf
                star_machine = 0

                # Find machine that gives earliest start time
                for m in range(self.num_machines):


                    machine_task_start = machine_etf[m]
                    task_start = task_etf[j]

                    if task_start > machine_task_start:
                        machine_task_start = task_start

                    # Check if this task_start time for a task on a machine is
                    # a minimum across the feasible machines
                    if machine_task_start < earliest_time:
                        earliest_time = machine_task_start
                        star_machine = m

                possible_tasks.append([j, earliest_time, star_machine])

                # Check/update earliest start time among tasks j in A
                if earliest_time < global_earliest_time:
                    global_earliest_time = earliest_time

            # Get list of all tasks that start at global_earliest_time
            P = []
            for i in possible_tasks:
                j, tj, mj = i
                if tj == global_earliest_time:
                    P.append([j, tj, mj])


            # Use tie_breaking_rule to determine the task to schedule


            b = self.break_tie(P, order)

            # if b == None:
            #     print(b)
            #     print(P)
            #     print(B_prime)
            #     print(B)
            star_task, machine_task_start, star_machine = b
            star_machine = int(star_machine)

            h[star_task] = star_machine
            t[star_task][0] = machine_task_start
            t[star_task][1] = machine_task_start + self.task_process_time[star_task]
            machine_etf[star_machine] = machine_task_start + self.task_process_time[star_task]

            done.append(star_task)
            order[star_machine].append(star_task)
            A.remove(star_task)

            # Update list of avaliable tasks ready to be scheduled
            for child in self.G.successors(star_task):
                if child not in done:
                    ready = True
                    for parent in self.G.predecessors(child):
                        if parent not in done:
                            ready = False
                            break
                    if ready:
                        A.append(child)
                        task_etf[child] = max([t[i][1] for i in self.G.predecessors(child)])

        return h, t, order

if __name__ == "__main__":
    pass