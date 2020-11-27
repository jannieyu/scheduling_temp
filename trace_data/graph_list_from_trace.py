import networkx as nx
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import json
# Construct DAGs

file_dict = {'1000genome':['1000genome-chameleon-2ch-100k-001.json',
'1000genome-chameleon-2ch-250k-001.json',
'1000genome-chameleon-4ch-100k-001.json',
'1000genome-chameleon-4ch-250k-001.json',
'1000genome-chameleon-6ch-100k-001.json',
'1000genome-chameleon-6ch-250k-001.json',
'1000genome-chameleon-8ch-100k-001.json',
'1000genome-chameleon-8ch-250k-001.json',
'1000genome-chameleon-10ch-100k-001.json',
'1000genome-chameleon-10ch-250k-001.json',
'1000genome-chameleon-12ch-100k-001.json',
'1000genome-chameleon-12ch-250k-001.json',
'1000genome-chameleon-14ch-100k-001.json',
'1000genome-chameleon-14ch-250k-001.json',
'1000genome-chameleon-16ch-100k-001.json',
'1000genome-chameleon-16ch-250k-001.json',
'1000genome-chameleon-18ch-100k-001.json',
'1000genome-chameleon-18ch-250k-001.json',
'1000genome-chameleon-20ch-100k-001.json',
'1000genome-chameleon-20ch-250k-001.json',
'1000genome-chameleon-22ch-100k-001.json',
'1000genome-chameleon-22ch-250k-001.json']}

def graph_list_from_trace(trace_name):
    file_list = file_dict[trace_name]
    G_list = []

    if trace_name not in ['1000genome']:
        return "Not valid trace name"

    for file in file_list:
        
        # Opening JSON file 
        f = open('pegasus-traces-master/' + trace_name + '/chameleon-cloud/' + file) 

        # returns JSON object as  
        # a dictionary 
        data = json.load(f) 

        taskname_to_id = {}
        id_count = 0

        G = nx.DiGraph()

        for task in data['workflow']['jobs']:

            child_id = id_count
            id_count += 1
            taskname_to_id[task['name']] = child_id

            if not G.has_node(child_id):
                    G.add_node(child_id)

            for parent in task['parents']:
                if parent in taskname_to_id:

                    parent_id = taskname_to_id[parent]

                else:

                    parent_id = id_count
                    id_count += 1
                    taskname_to_id[parent] = parent_id


                if not G.has_edge(parent_id, child_id):
                        G.add_edge(parent_id, child_id)
        print(len(G.nodes))

        
    
        G_list.append(G)
        
        # Closing file 
        f.close()
    return G_list