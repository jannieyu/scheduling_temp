import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random


def make_graph_visual(G, num_tasks):
    '''
    Visualize graph G
    '''

    labels = {}

    for i in range(0, num_tasks):
        labels[i] = str(i)
    if nx.check_planarity(G)[0]:
        pos=nx.planar_layout(G)
    else:
        pos=nx.draw_kamada_kawai(G)

    nx.draw(G, pos, nodecolor='y',edge_color='k')
    nx.draw_networkx_labels(G, pos, labels, font_size=20, font_color='y')
    plt.axis('off')
    plt.show()
