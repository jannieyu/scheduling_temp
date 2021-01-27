import numpy as np
import networkx as nx
from gekko import GEKKO
import matplotlib.pyplot as plt

def init_solver(v, O_value=5):
    m = GEKKO()

    # Use IPOPT solver (default)
    m.options.SOLVER = 3

    # Change to parallel linear solver
    m.solver_options = ['linear_solver ma97']

    # variable array dimension
    n = len(v) # rows

    # create array
    s = m.Array(m.Var, n)
    for i in range(n):
                s[i].value = 2.0
                s[i].lower = 0
                
    # Optimal value for ibjective
    O = m.Var(value=O_value, lb=0)
    
    # The objective basically
    m.Equation(sum([int(v[i]) / s[i] + s[i] for i in range(len(v))]) == O)

    return m, s, O

def solver_results(s, m, O, verbose=True):

    m.Obj(O) # Objective
    m.options.IMODE = 3 # Steady state optimization
    m.solve(disp=False) # Solve

    if verbose:
        print('Results')
        for i in range(len(s)):
            print(str(i) + " " + str(s[i].value))
        print('Objective: ' + str(m.options.objfcnval))

    task_process_time = [float(1 / s[i].value[0]) for i in range(len(s))]

    return s, task_process_time, m.options.objfcnval


def make_graph_visual(G, num_tasks):
       
        labels = {}

        for i in range(0, num_tasks):
            labels[i] = str(i)

        pos=nx.circular_layout(G)
        nx.draw(G, pos, nodecolor='y',edge_color='k')
        nx.draw_networkx_labels(G, pos, labels, font_size=20, font_color='y')
        plt.axis('off')
        
        plt.show()


def make_assignment_visual(task_process_time, dependencies, machine_task_list):

        t = create_start_end_times(task_process_time, dependencies)
        print("t is ", t)
        m = len(machine_task_list)
        # multi-dimensional data 
        machine_data = [[] for _ in range(m)]
        machine_labels = [[] for _ in range(m)]

        for i in range(m):
            machine_etfd = 0
            task_list = machine_task_list[i]
            for j in task_list:
                if machine_etfd < t[j][0]:
                    idle_time = t[j][0] - machine_etfd
                    machine_data[i].append(idle_time)
                    machine_labels[i].append('idle')
                process_time = task_process_time[j]
                machine_etfd = t[j][1]
                machine_data[i].append(process_time)
                machine_labels[i].append(str(j))

        segments = max([len(task_list) for task_list in machine_data])

        for i in range(len(machine_data)):
            for j in range(len(machine_data[i]), segments):
                machine_data[i].append(0)
                machine_labels[i].append('')

        data = []
        for s in range(segments):
            section = [machine_data[j][s] for j in range(m)]
            data.append(section)

        y_pos = np.arange(m)

        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)

        colors ='yg'
        patch_handles = []
        # left alignment of data task_starts at zero
        left = np.zeros(m) 
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
                ax.text(x,y, machine_labels[i][j], ha='center')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(np.arange(m))
        ax.set_xlabel('Time')
        plt.show()

def create_start_end_times(task_process_time, dependencies):

    num_tasks = len(task_process_time)
    t = [[None, None] for _ in range(num_tasks)]


    for task in range(num_tasks):
        t[task][0] = sum(task_process_time[i] for i in dependencies[task] if i != task)
        t[task][1] = t[task][0] + task_process_time[task]

    return t

def get_makespan(task_process_time, machine_task_list):
    makespan = 0
    for t in machine_task_list[1]:
        makespan += task_process_time[t]


    return makespan

# The following helper functions solve makepan + energy, not T + E
def init_solver_original(v, machine_task_list):
    m = GEKKO()

    # Use IPOPT solver (default)
    m.options.SOLVER = 3

    # Change to parallel linear solver
    m.solver_options = ['linear_solver ma97']

    # variable array dimension
    n = len(v) # rows

    # create array
    s = m.Array(m.Var, n)
    for i in range(n):
                s[i].value = 2.0
                s[i].lower = 0

    P = m.Var(value=5, lb=0)
    m.Equation(sum([s[i] for i in range(len(v))]) == P)

    M1 = m.Var(value=5, lb=0)
    M2 = m.Var(value=5, lb=0)
    m.Equation(sum([1 / s[i]  for i in machine_task_list[0]]) == M1)
    m.Equation(sum([1 / s[i]  for i in machine_task_list[1]]) == M2)

    
    return m, s, P, M1, M2


def solver_results_original(m, s, P, M1, M2, verbose=True):

    m.Obj(P + m.max2(M1, M2)) # Objective
    m.options.IMODE = 3 # Steady state optimization
    m.solve(disp=False) # Solve

    if verbose:
        print('Results')
        for i in range(len(s)):
            print(str(i+1) + " " + str(s[i].value))
        print('Objective: ' + str(m.options.objfcnval))

    task_process_time = [float(1 / s[i].value[0]) for i in range(len(s))]

    return s, M1, M2, P,  task_process_time

def v_helper(G, num_tasks, task_permutation_dict, machine_task_list, t):
    #print("Inside v_helper")
    v = [0 for _ in range(num_tasks)]
    dependencies = [[] for _ in range(num_tasks)]
    #print("machine_task_list is: ", machine_task_list)
    #print("num_tasks is: ", num_tasks)
    #print("t is: ", t)
    #print("task_permutation_dict is: ", task_permutation_dict)
    for task, prev_task in task_permutation_dict.items():
      
        if prev_task == None:
            parents = G.predecessors(task)

            for p in parents:
                if t[task][0] == t[p][1]:
                    dependencies[task].extend(dependencies[p])
                    break

        # not first task on machine
        else:
            if t[task][0] == t[prev_task][1]:
                dependencies[task].extend(dependencies[prev_task])
            

            if t[task][0]!= t[prev_task][1]:
                parents = G.predecessors(task)
                
                for p in parents:
                    if t[task][0] == t[p][1]:
                        dependencies[task].extend(dependencies[p])
                        break
                        
        dependencies[task].append(task)
        for d in dependencies[task]:
            v[d] += 1

    return v, dependencies


def add_constraints(m, s, equality_constraints, relaxing_constraints):
   
    # print('constraints:')
    # print(equality_constraints, relaxing_constraints)

    for constraint in equality_constraints:
        if isinstance(constraint[0], int):
            m.Equation(1 / s[constraint[0]] == 1 / s[constraint[1]]) 
        else:
            m.Equation(sum([1 / s[i] for i in constraint[0]]) == sum([1 / s[j] for j in constraint[1]]) )
        # 

    
    for constraint in relaxing_constraints:
     
        m.Equation(sum([1 / s[j] for j in constraint[0]]) <= sum([1 / s[j] for j in constraint[1]]) )

    return m,s

def define_constraints(G, time_chunks, dependencies):
    num_machines = len(time_chunks)
    num_max_tasks = len(time_chunks[0])
    equality_constraints = []
    relaxing_constraints = []
    # index i

    for i in range(1, num_max_tasks - 1):
        for j in range(num_machines):

            # key task that we are checking on 
            task = time_chunks[j][i]

            if task != "idle":

                parents = G.predecessors(task)
                children = G.successors(task)


                # find tasks that need to run at the same time as other tasks
                share_parent_tasks = []
                share_children_tasks = []
                for z in range(num_machines):
                           
                            if z != j and time_chunks[z][i] != 'idle':
                               
                                concurr_task = time_chunks[z][i]
                                other_parents = G.predecessors(concurr_task)
                                other_children = G.successors(concurr_task)

                                share_parent = False
                                share_child = False
                                for a in range(num_machines):
                                    prev_task = time_chunks[a][i - 1]
                                    if prev_task in parents and prev_task in other_parents:
                                        share_parent = True


                                    next_task = time_chunks[a][i + 1]
                                    if next_task in children and next_task in other_children:
                                        share_child = True


                                if share_child and share_parent:
                                    equality_constraints.append([task, concurr_task])
                            

                # check if task can be relaxed to the right into idle time
                if time_chunks[j][i + 1] == 'idle':
                    min_machine = None
                    min_relax = num_max_tasks
                    children = G.successors(task)
                    for s in range(num_machines):
                        if s != j:
                            relax = 0

                            for v in range(i + 1, num_max_tasks):

                                if time_chunks[s][v] not in children:
                                    relax += 1
                                else:
                                    break

                            if relax < min_relax:
                                min_relax = relax
                                min_machine = s

                    own_relax = 0
                    for x in range(i + 1, num_max_tasks):
                        if time_chunks[j][x] == 'idle':
                            own_relax += 1
                        else:
                            break

                    end_task = time_chunks[min_machine][i + min(min_relax, own_relax)]
                    relaxing_constraints.append([dependencies[task], dependencies[end_task]])

                # make sure that all parents finish before task starts
                parents = G.predecessors(task)
                parent_list = []

                prev_task = time_chunks[j][i-1]

                if prev_task != 'idle':

                    for k in range(len(time_chunks)):
                        p = time_chunks[k][i-1]
                        if p in parents and (j != k):
                            parent_list.append(p)


                    for parent in parent_list:

                        left = dependencies[parent]
                        right = dependencies[prev_task]
                        equality_constraints.append([left, right])

    return equality_constraints, relaxing_constraints



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


