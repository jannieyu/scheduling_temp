import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import plotly.figure_factory as ff



def make_graph_visual(G, num_tasks):
    '''
    Visualize graph G
    '''
    if nx.check_planarity(G)[0]:
        pos=nx.planar_layout(G)
    else:
        pos=nx.shell_layout(G)
    nx.draw(G, pos, node_color='k', node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=20, font_color='y')
    plt.axis('off')
    plt.show()

def plot_gantt(task_metadata, objective_value, color_palette):
    """
     Plots the task_speed_scaling gantt chart given the metadata
    :param task_metadata: metadata
    :param objective_value: value of objective for current value
    :param color_palette: rgb tuples for colors to use
    :return:
    """
    df = []
    colors = {}
  
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

def plot_schedule(num_tasks, order, t, cost):
    color_palette = [(0, 0, 255 / 256), (0, 255 / 256, 0), (255 / 256, 255 / 256, 0), (255 / 256, 0, 0),
        (255 / 256, 128 / 256, 0),
        (255 / 256, 0, 127 / 256), (0, 255 / 256, 255 / 256), (127 / 256, 0, 255 / 256),
        (128 / 256, 128 / 256, 128 / 256),
        (255 / 256, 255 / 256, 255 / 256), (0, 0, 0)]
    metadata = make_task_metadata(order, num_tasks, t)
    plot_gantt(metadata, cost, color_palette)
    
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

