import numpy as np
import networkx as nx
from gekko import GEKKO
import matplotlib.pyplot as plt
import random

# Initialize GEKKO solver to solve for T + E, where T is makespan
def init_solver(G, w, num_tasks, task_prev, task_ordering):
    m = GEKKO()

    # Use IPOPT solver (default)
    m.options.SOLVER = 3

    # Change to parallel linear solver
    m.solver_options = ['linear_solver ma97']

    # variable array dimension
    
    # create array
    s = m.Array(m.Var, num_tasks)
    for i in range(num_tasks):
                s[i].value = 2.0
                s[i].lower = 0

    M = m.Var(value=5, lb=0)
    P = m.Var(value=5, lb=0)

    # define completion time of each task
    c = m.Array(m.Var, num_tasks)
    for i in range(num_tasks):
                c[i].value = 0
                c[i].lower = 0

    #1b
    # task's completion time must be later than the time to run task itself
    for i in range(num_tasks):
            m.Equation(w[i] / s[i]  <= c[i])

    #1c
    # task must start later than all ancestors
    for i in range(num_tasks):
        for j in nx.algorithms.ancestors(G, i):
            m.Equation(c[j] + (w[i] / s[i]) <= c[i])

    # task must start later than previous task on machine
    for task, prev in task_prev.items():
        if prev != None:
            m.Equation(c[prev] + (w[task] / s[task]) <= c[task])

    #1d
    # Total load assigned to each machine should not be greater than the makespan
    for lst in task_ordering:
        m.Equation(sum([w[i] / s[i] for i in lst]) <= M)

    #1e (define M in objective function)
    for i in range(num_tasks):
        m.Equation(c[i] <= M)

    # define P in objective function
    m.Equation(sum([s[i] for i in range(num_tasks)]) == P)

    return m, s, c, P, M


def optimize_makespan_power(w, m, s, c, P, M, task_ordering, verbose=True):
    try:
        m.Obj(P + M) # Objective
        m.options.IMODE = 3 # Steady state optimization
        m.solve(disp=False) # Solve

        if verbose:
            print('Results')
            for i in range(len(s)):
                print(str(i) + " " + str(s[i].value) + " " + str(c[i].value))
            print('Objective: ' + str(m.options.objfcnval))

        task_process_time = [float(w[i] / s[i].value[0]) for i in range(len(s))]
    except ImportError:
        print(task_ordering)
        return task_ordering

    return s, M, P, task_process_time, m.options.objfcnval

# Example of using GEKKO

# m = GEKKO() # Initialize gekko

# # Use IPOPT solver (default)
# m.options.SOLVER = 3

# # Change to parallel linear solver
# m.solver_options = ['linear_solver ma97']

# # Initialize variables
# x1 = m.Var(value=1,lb=1,ub=5)
# x2 = m.Var(value=5,lb=1,ub=5)
# x3 = m.Var(value=5,lb=1,ub=5)
# x4 = m.Var(value=1,lb=1,ub=5)
# # Equations
# m.Equation(x1*x2*x3*x4>=25)
# m.Equation(x1**2+x2**2+x3**2+x4**2==40)
# m.Obj(x1*x4*(x1+x2+x3)+x3) # Objective
# m.options.IMODE = 3 # Steady state optimization
# m.solve(disp=False) # Solve
# print('Results')

# print('x1: ' + str(x1.value))
# print('x2: ' + str(x2.value))
# print('x3: ' + str(x3.value))
# print('x4: ' + str(x4.value))
# print('Objective: ' + str(m.options.objfcnval))

