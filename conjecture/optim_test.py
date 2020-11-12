from gekko import GEKKO
import networkx as nx


# Initialize GEKKO solver to solve for T + E, where T is total weighted completion time
def init_solver(G, v,  resource_constraints, machine_task_list):
    m = GEKKO()

    # Use IPOPT solver (default)
    m.options.SOLVER = 3

    # Change to parallel linear solver
    m.solver_options = ['linear_solver ma97']

    # variable array dimension
    num_tasks = len(v)
    # create array
    s = m.Array(m.Var, num_tasks)
    for i in range(num_tasks):
        s[i].value = 2.0
        s[i].lower = 0

    O = m.Var(value=5, lb=0)
    # define completion time of each task
    c = m.Array(m.Var, num_tasks)
    for i in range(num_tasks):
        c[i].value = 0
        c[i].lower = 0

    # 1b
    # task's completion time must be later than the time to run task itself
    for i in range(num_tasks):
        m.Equation(1 / s[i] <= c[i])

    # 1c
    # task must start later than all ancestors
    for i in range(num_tasks):
        for j in nx.algorithms.ancestors(G, i):
            m.Equation(c[j] + (1 / s[i]) <= c[i])

    # task must start later than previous task on machine
    for constraint in resource_constraints:
        task = constraint[1]
        prev = constraint[0]
        m.Equation(c[prev] + (1 / s[task]) <= c[task])

    # all tasks on single machine must run at same speed
    for machine in machine_task_list:
        for i in range(len(machine)):
            if i != len(machine)-1:
                m.Equation(s[machine[i]] == s[machine[i+1]])

    # Objective
    m.Equation(sum([int(v[i]) / s[i] + s[i] for i in range(len(v))]) == O)

    # # 1d
    # # Total load assigned to each machine should not be greater than the makespan
    # for lst in task_ordering:
    #     m.Equation(sum([1 / s[i] for i in lst]) <= M)
    #
    # # 1e (define M in objective function)
    # for i in range(num_tasks):
    #     m.Equation(c[i] <= M)
    #
    # # define P in objective function
    # m.Equation(sum([s[i] for i in range(num_tasks)]) == P)


    return m, s, c, O


def optimize(m, s, c, O):
    m.Obj(O)  # Objective
    m.options.IMODE = 3  # Steady state optimization
    m.solve(disp=False)  # Solve


    print('Results')
    for i in range(len(s)):
        print(str(i) + " " + str(s[i].value) + " " + str(c[i].value))
    print('Objective: ' + str(m.options.objfcnval))

    return

def get_resource_constraints(machine_task_list):
    resource_constraints = []
    for machine in machine_task_list:
        for i in range(len(machine)):
            if i != len(machine) -1:
                task = machine[i]
                next_task = machine[i+1]
                resource_constraints.append([task, next_task])

    return resource_constraints


if __name__ == "__main__":
    dag = nx.DiGraph()
    dag.add_nodes_from(range(9))
    dag.add_edges_from([(0, 2), (2, 3), (2, 4), (3, 5), (4, 5), (5, 6), (6, 7), (7, 8)])
    machine_task_list = [[0, 2, 3, 5, 6, 7, 8], [4, 1]]
    v = [9, 1, 8, 5, 2, 4, 3, 2, 1]
    resource_constraints = get_resource_constraints(machine_task_list)
    m, s, c, O = init_solver(dag, v, resource_constraints, machine_task_list)
    optimize(m, s, c, O)