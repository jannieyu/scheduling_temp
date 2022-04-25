import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
from copy import deepcopy
import numpy as np
import statsmodels.api as sm
import random
import pickle
import json
import ast

from graph_util.visualization_util import make_graph_visual
from scheduling_util.modified_etf import Mod_ETF
from scheduling_util.consolidated_functions import opt_schedule_given_ordering
from scheduling_util.heuristics import *
from graph_util.random_graph_functions import random_all_fork, random_all_join
from graph_util.erdos_renyi_dag import er_dag
from scheduling_util.approx_pseudosizes import speed_to_psize



def create_random_graph_data(filename):
    
    # init parameters
    num_dags = 5
    num_machines = 2
    num_tasks_list = [5, 10, 15]
    probability = 0.3
    input_df = pd.DataFrame(columns = [
            "graph_object", 
        ])
    for num_tasks in num_tasks_list:
        for i in range(num_dags):

            # create single DAG
            G, _ = er_dag(num_tasks, probability) #create graph object
            assert(nx.algorithms.dag.is_directed_acyclic_graph(G))

            entry_dict = {
            "graph_object": nx.node_link_data(G), # save graph object in json form
            }


            input_df = input_df.append(entry_dict, ignore_index = True)

    input_df.to_csv(filename, index=False)

create_random_graph_data("small_graph_training_data.csv")
create_random_graph_data("small_graph_testing_data.csv")