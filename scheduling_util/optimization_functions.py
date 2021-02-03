from gekko import GEKKO
import networkx as nx
import random
import plotly.figure_factory as ff
from fractions import Fraction as frac


def init_ordering_solver(mrt, G, num_tasks, w, order, task_scaling=False):
    """
    prepares the solver (given ordering) by adding the necessary constraints
    :param mrt: Boolean variable that is True if objective is to optimize for
                MRT + E, False if objective is to optimize Makespan + E.
    :param G: DAG to schedule
    :param num_tasks: total number of tasks
    :param order: ordering
    :param task_scaling: Boolean whether to scale tasks or not
    :return: m, s, c
    """
    m = GEKKO()

    # Use IPOPT solver (default)
    m.options.SOLVER = 3

    # Change to parallel linear solver
    m.solver_options = ['linear_solver ma97']

    # create array
    s = m.Array(m.Var, num_tasks)
    for i in range(num_tasks):
        s[i].value = 2.0
        s[i].lower = 0

    # define completion time of each task
    c = m.Array(m.Var, num_tasks)
    for i in range(num_tasks):
        c[i].value = 0
        c[i].lower = 0

    # 1b
    # task's completion time must be later than the time to run task itself
    for i in range(num_tasks):
        m.Equation(w[i] / s[i] <= c[i])

    # 1c
    # task must start later than all ancestors
    for i in range(num_tasks):
        for j in nx.algorithms.ancestors(G, i):
            m.Equation(c[j] + (w[i] / s[i]) <= c[i])


    # task must start later than previous task on machine
    resource_constraints = get_resource_constraints(order)
    for constraint in resource_constraints:
        task = constraint[1]
        prev = constraint[0]
        m.Equation(c[prev] + (w[task] / s[task]) <= c[task])

    # # all tasks on single machine must run at same speed
    # if not task_scaling:
    #     for machine in order:
    #         for i in range(len(machine)):
    #             if i != len(machine)-1:
    #                 m.Equation(s[machine[i]] == s[machine[i+1]])

    P = m.Var(value=5, lb=0)
    m.Equation(m.sum([w[j] * s[j] for j in range(num_tasks)]) == P)

    M = m.Var(value=5, lb=0)
    MRT = m.Var(value=5, lb=0)


    for j in range(num_tasks):
            m.Equation(c[j] <= M)

    for lst in order:
        m.Equation(sum([w[i] / s[i] for i in lst]) <= M)

    # define MRT
    m.Equation(m.sum([c[j] for j in range(num_tasks)]) == MRT)

    if mrt:
        m.Obj(MRT + P)

    else:

        m.Obj(P + M) # Objective


    return m, s, c

def init_relaxed_opt_solver(mrt, G, num_tasks, num_machines, w):
    """
    prepares the relaxed optimal solver by adding the necessary constraints
    :param mrt: Boolean variable that is True if objective is to optimize for
                MRT + E, False if objective is to optimize Makespan + E.
    :param G: DAG to schedule
    :param num_tasks: total number of tasks
    :param w: weights
    :return: m, s, c
    """
    m = GEKKO()

    # Use IPOPT solver (default)
    m.options.SOLVER = 3

    # Change to parallel linear solver
    #m.solver_options = ['minlp_max_iter_with_int_sol 10000']

    # create array
    s = m.Array(m.Var, num_tasks)
    for i in range(num_tasks):
        s[i].value = 2.0
        s[i].lower = 0

    # define completion time of each task
    c = m.Array(m.Var, num_tasks)
    for i in range(num_tasks):
        c[i].value = 0
        c[i].lower = 0

    # relaxation of variables to take on a value in [0,1]
    x = [[m.Var(0,lb=0,ub=1) for j in range(num_tasks)] for i in range(num_machines)]

    # 1a
    # each task will be assigned to exactly one machine
    for j in range(num_tasks):
        m.Equation(m.sum([x[i][j] for i in range(num_machines)]) == 1)

    # 1b
    # task's completion time must be later than the time to run task itself
    for j in range(num_tasks):
        m.Equation( w[j] / s[j]  <= c[j])

    # 1c
    # task must start later than all ancestors
    for j in range(num_tasks):
        for k in nx.algorithms.ancestors(G, j):
            m.Equation(c[k] + (w[j] / s[j]) <= c[j])

    M = m.Var(value=5, lb=0)
    P = m.Var(value=5, lb=0)
    MRT = m.Var(value=5, lb=0)

    # Total load assigned to each machine should not be greater than the makespan
    for i in range(num_machines):
        m.Equation(m.sum([w[j] * x[i][j] / s[j] for j in range(num_tasks)]) <= M)

    # 1e (define M in objective function)
    for j in range(num_tasks):
        m.Equation(c[j] <= M)

    # define P in objective function
    m.Equation(m.sum([w[j] * s[j] for j in range(num_tasks)]) == P)

    # define MRT
    m.Equation(m.sum([c[j] for j in range(num_tasks)]) == MRT)

    if mrt:
        m.Obj(MRT + P)

    else:

        m.Obj(P + M) # Objective

    return x, m, s, c


def init_opt_solver(G, num_tasks, num_machines, w):
    """
    prepares the optimization equation by adding the necessary constraints
    :param mrt: Boolean variable that is True if objective is to optimize for
                MRT + E, False if objective is to optimize Makespan + E.
    :param G: DAG to schedule
    :param num_tasks: total number of tasks
    :param w: weights
    :return: m, s, c
    """
    m = GEKKO()

    # Use IPOPT solver (default)
    m.options.SOLVER = 3

    # Change to parallel linear solver
    #m.solver_options = ['minlp_max_iter_with_int_sol 10000']

    # create array
    s = m.Array(m.Var, num_tasks)
    for i in range(num_tasks):
        s[i].value = 2.0
        s[i].lower = 0

    # define completion time of each task
    c = m.Array(m.Var, num_tasks)
    for i in range(num_tasks):
        c[i].value = 0
        c[i].lower = 0

    x = [[m.Var(0,lb=0,ub=1) for j in range(num_tasks)] for i in range(num_machines)]

    #Yu's constraints that you can uncomment
    p = [[m.Var(0,lb=0,ub=1, integer=True) for j in range(num_tasks)] for j_prime in range(num_tasks)]
    b = [[m.Var(0,lb=0,ub=1, integer=True) for j in range(num_tasks)] for j_prime in range(num_tasks)]

    # 2a
    # each task will be assigned to exactly one machine
    for j in range(num_tasks):
        m.Equation(m.sum([x[i][j] for i in range(num_machines)]) == 1)

    # 2b
    # task's completion time must be later than the time to run task itself
    for j in range(num_tasks):
        m.Equation( w[j] / s[j]  <= c[j])

    # 2c
    # task must start later than all ancestors
    for j in range(num_tasks):
        for k in nx.algorithms.ancestors(G, j):
            m.Equation(c[k] + (w[j] / s[j]) <= c[j])

    P = m.Var(value=5, lb=0)
    MRT = m.Var(value=5, lb=0)

    
    # # Total load assigned to each machine should not be greater than the makespan
    # for i in range(num_machines):
    #     m.Equation(m.sum([w[j] * x[i][j] / s[j] for j in range(num_tasks)]) <= M)

    # # 1e (define M in objective function)
    # for j in range(num_tasks):
    #     m.Equation(c[j] <= M)

    # define MRT
    # constraint in place of 2d, 2e
    m.Equation(m.sum([c[j] for j in range(num_tasks)]) == MRT)

    
    # 2f
    # define P in objective function
    m.Equation(m.sum([w[j] * s[j] for j in range(num_tasks)]) == P)

    # define MRT
    m.Equation(m.sum([c[j] for j in range(num_tasks)]) == MRT)



    # 2g
    for j_prime in range(num_tasks):
        for j in range(num_tasks):
            if j != j_prime:
                m.Equation(m.sum([x[i][j] * x[i][j_prime] for i in range(num_machines)]) == p[j][j_prime])

    for j_prime in range(num_tasks):
        for j in range(num_tasks):
            if j != j_prime:
                m.Equation(p[j][j_prime] * (c[j] - c[j_prime] + (w[j_prime] / s[j_prime])) <= b[j][j_prime] * (
                            MRT - c[j_prime] + (w[j_prime] / s[j_prime])))
                m.Equation(b[j][j_prime] * (c[j_prime] + (w[j]/s[j])) <= p[j][j_prime] * c[j])
                m.Equation(b[j][j_prime] <= p[j][j_prime])
                b[j][j_prime] = m.if3(b[j_prime][j] - 1, 1, 0)




    m.Obj(MRT + P)

    


    # Old Objective
    # m.Obj(sum([int(v[i]) / s[i] + s[i] for i in range(len(v))]))

    # objective for mean completion time

    return x, m, s, c


def get_resource_constraints(order):
    """
    gets resource constraints for a given ordering
    :param order:
    :return: resource constraints
    """
    resource_constraints = []
    for machine in order:
        for i in range(len(machine)):
            if i != len(machine) -1:
                task = machine[i]
                next_task = machine[i+1]
                resource_constraints.append([task, next_task])

    return resource_constraints


def solver_results(x, s, m, c, w, order=False,  verbose=True):
    """
    solves the optimization equation
    :param s: speeds
    :param m: gekko model
    :param c: completion times
    :param verbose: boolean to print or not
    :param order: optional order to print or not
    :return: task_process_time, ending times, intervals, speeds, objective value
    """

    #m.Obj(O) # Objective

    try:
        m.options.IMODE = 3  # Steady state optimization
        m.solve(disp=verbose) # Solve

    except:
        # print("Did not work")
        # if order!=False:
            # print("Order is ", order)
        return order, [-1]*len(s), [-1]*len(s), [-1,-1]*len(s),[-1]*len(s), 10000000

    task_process_time = [frac(w[i] / frac(s[i].value[0])) for i in range(len(s))]
    ending_time = [frac(c[i].value[0]) for i in range(len(c))]
    intervals = [[end - process_time, end] for (process_time, end) in zip(task_process_time, ending_time)]
    speeds = [frac(s[i].value[0]) for i in range(len(s))]

    task_process_time = [float(process_time.__round__(5)) for process_time in task_process_time]
    ending_time = [float(end_time.__round__(5)) for end_time in ending_time]
    intervals = [[float(interval[0].__round__(5)), float(interval[1].__round__(5))] for interval in intervals]
    speeds = [float(speed.__round__(5)) for speed in speeds]

    if verbose:
        print('Results')
        for i in range(len(s)):
            print(str(i) + " Speed: " + str(s[i].value) + " Ending Time: " + str(c[i].value) + " Interval: " +
                  str(intervals[i]) + " Task process time: " + str(task_process_time[i]))
        print('Objective: ' + str(m.options.objfcnval))

    if x != None:
        order = create_order(x, c)
    else:
        order = None

    return order, task_process_time, ending_time, intervals, speeds, float(frac(m.options.objfcnval).__round__(5))

def create_order(x, c):

    order = []

    num_machines = len(x)
    num_tasks = len(c)

    for i in range(num_machines):
        machine_unordered = []
        for j in range(num_tasks):
            if int(round(x[i][j].value[0])) == 1:
                machine_unordered.append((c[j].value[0], j))

        machine_sorted = sorted(machine_unordered)
        machine_order = []

        for k in range(len(machine_sorted)):

            machine_order.append(machine_sorted[k][1])
        order.append(machine_order)


    return order

def get_objective(ending_times, speeds):
    """
    Calculates objective given a schedule
    :param ending_times: completion times for a task
    :param speeds: speeds of tasks running
    :return:
    """
    return sum([ending_times[i] + speeds[i] for i in range(len(speeds))])


def get_machines(order, num_tasks):
    """
    returns list of task machine mappings
    :param order: order of tasks across machines
    :param num_tasks: number of total tasks
    :return:
    """
    machines = num_tasks * [-1]
    for machine_index in range(len(order)):
        for task in order[machine_index]:
            machines[task] = machine_index
    return machines


def make_task_metadata(order, num_tasks, intervals):
    """
    makes data ready to be plotted on a pretty gantt chart
    :param order: machine task ordering
    :param num_tasks: total number of tasks for the dag
    :param intervals: start end time 2d list
    :return: a dict of metadata with subdict for fields
    """
    task_metadata = {}
    machines = get_machines(order, num_tasks)
    for task_name in range(len(intervals)):

        task_metadata[task_name] = {'start': intervals[task_name][0], 'end': intervals[task_name][1],
                                    'task': task_name, 'machine': machines[task_name]}
    return task_metadata


def plot_gantt(task_metadata, objective_value, color_palette):
    """
    plots the task_speed_scaling gantt chart given the metadata
    :param task_metadata: metadata
    :param objective_value: value of objective for current value
    :param color_palette: rgb tuples for colors to use
    :return:
    """
    df = []
    colors = {}
    # print(task_metadata)
    for task_key in task_metadata:
        task = task_metadata[task_key]
        df.append(dict(Task=str("Machine " + str(task['machine'])), Start=task['start'], Finish=task['end'], Machine=task['task']))
        if task['task'] < len(color_palette):
            color = color_palette[task['task']]
        else:
            color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
            color_palette.append(color)
        colors[task['task']] = color
    title = "Speed Scaling Gantt Chart for Objective: " + str(objective_value)
    fig = ff.create_gantt(df, colors=colors, index_col='Machine', show_colorbar=True, group_tasks=True, showgrid_x=True, showgrid_y=True, title=title)
    fig.update_xaxes(type='linear')
    fig.show("notebook")
    return color_palette

