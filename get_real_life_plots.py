import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
from copy import deepcopy
import numpy as np
import statsmodels.api as sm
import random
import ast
import pickle
from graph_util.visualization_util import make_graph_visual
from scheduling_util.modified_etf import Mod_ETF
from scheduling_util.consolidated_functions import opt_schedule_given_ordering
from scheduling_util.heuristics import *
from graph_util.random_graph_functions import random_all_fork, random_all_join
from graph_util.erdos_renyi_dag import er_dag
from scheduling_util.approx_pseudosizes import speed_to_psize
from knockknock import slack_sender


# Function to get list of feature sets. A feature set is in the form [x1, x2, ..., y] where y is psize
def get_feature_set(G):
    lst = []
    
    # Get attributes for graph G
    out_bet_lst = nx.algorithms.out_degree_centrality(G)
    trophic_lst = nx.algorithms.trophic_levels(G)
    
    # Create feature set for each node in G
    for node in G.nodes:
        out_bet = out_bet_lst[node]
        trophic = trophic_lst[node]
        descendants = list(nx.algorithms.dag.descendants(G, node))
        descendants.append(node)
        
        lst.append({
            "constant": 1,
            "num_descendants": float(len(descendants)),
            "out_degree_betweenness_centrality": float(out_bet),
            "trophic_levels": float(trophic),
        })   
        
    return lst

def reset_psizes(psizes):
    revised_lst = [abs(x) for x in psizes]
    assert(all([z > 0 for z in revised_lst]))
    return revised_lst

# Function to create dataset
def create_dataset(num_machines, csv_file):
    
    df = pd.DataFrame(columns = [
        "graph_object",
        "num_tasks",
        "num_machines",
        "weights",
        "order",
        "features",
        "psize",
        "GD_cost",
        "LR_cost",
        "RLP_cost",
        "ETF-H_cost",
        "weak_strongman_cost"
    ])
    tie_breaking_rule = 2
    count = 0

    csv_df = pd.read_csv(csv_file) 
    for index, row in csv_df.iterrows():
        
        dict_dag = ast.literal_eval(row["graph_object"])
        G = nx.node_link_graph(dict_dag)

        num_tasks = len(G)
        
        _, _, h1_cost, _ = heuristic_algorithm(G, num_machines)
        w = [1 for _ in range(num_tasks)]
        s = [1 for _ in range(num_tasks)]
        p = [1 for _ in range(num_tasks)]
        etf = Mod_ETF(G, w, s, num_machines, tie_breaking_rule, plot=False)
        weak_strongman_cost = naive_2(G, num_machines)
        intervals, speeds, opt_cost = opt_schedule_given_ordering(True, G, w, p, etf.order, plot=False, compare=False)
        if speeds[0] != -1:
            entry_dict = {
                "graph_object": nx.node_link_data(G),
                "num_tasks": num_tasks,
                "num_machines": num_machines,
                "weights": w,
                "order": etf.order,
                "features": get_feature_set(G),
                "psize": speed_to_psize(speeds),
                "GD_cost": np.inf,
                "LR_cost": np.inf,
                "RLP_cost": opt_cost,
                "ETF-H_cost": h1_cost,
                "weak_strongman_cost": weak_strongman_cost
            }
            df = df.append(entry_dict, ignore_index = True)
                    
    return df

def compute_cost(w, t, s):
    '''
    Given weights w, time intervals t, and speeds s, compute the cost of the schedule; 
    returns total cost as well as separate power and time components.
    '''
    power = 0
    time = 0
    #print(f"compute cost intervals are {t}")
    for j in range(len(s)):
        if t[j] == -1:
            return -1, -1, -1
        power += w[j] * s[j]
        time += t[j][1]
    total_cost = power + time
    return total_cost, power, time

def psize_to_speed(psize):
    return [np.sqrt(psize[i]) for i in range(len(psize))]

#predict list of psizes for one graph G
def predict(coef, features):
    m = []
    for f in features:
        m.append(list(f.values()))
    return np.matmul(m, coef)


# Functions for Gradient Descent Approach
def single_weight_update(coef, G, w, features, order, curr_cost, step_size):
    
    m = []
    for f in features:
        m.append(list(f.values()))
        
    curr_min_coef = deepcopy(coef)
    min_cost = curr_cost
    all_combinations = list(product((-1, 1, 0), repeat=len(coef)))
    for combo in all_combinations:
        temp_coef = deepcopy(coef)
        for i in range(len(combo)):
            temp_coef[i] += step_size * combo[i]
        psizes = reset_psizes(np.matmul(m, temp_coef))
        speeds = psize_to_speed(psizes)
        time_intervals = native_rescheduler(deepcopy(G), deepcopy(speeds), deepcopy(w), deepcopy(order))
        cost, _, _ = compute_cost(w, time_intervals, speeds)
        if cost < min_cost:
            min_cost = cost
            curr_min_coef = deepcopy(temp_coef)
        
    return min_cost, curr_min_coef

def gd_algorithm(lr_coefficients, df):
    MAX_ITER = 1000
    step_size = 0.05
    i = 0
    
    coef = lr_coefficients

    stopping_condition = 0.001

    #iterate until the objective function cost change lowers to stopping point
    while i < MAX_ITER:

        cost_lst = []
        max_change = - np.infty
        for index, row in df.iterrows():
            new_cost, new_coef = single_weight_update(
                coef, 
                nx.node_link_graph(row["graph_object"]),
                row["weights"], 
                row["features"], 
                row["order"],
                row["GD_cost"],
                step_size
            )

            if row["GD_cost"] == np.inf: 
                change = - np.inf

            else:
                change = row["GD_cost"] - new_cost

            cost_lst.append(new_cost)
            max_change = max(change, max_change)
            coef = new_coef
        print(max_change)

        # update costs
        new_df = pd.DataFrame({'GD_cost': cost_lst})
        df.update(new_df)
        if max_change < stopping_condition and i > 1:
            print("Hit stopping condition with ", max_change)
            break
        i += 1
    
    return df, coef


def save_df(df, save_file):
    with open(save_file, "wb") as f:
        pickle.dump(df, save_file)

@slack_sender(webhook_url="https://hooks.slack.com/services/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", channel="Vivek Anand", user_mentions=["U03DY5GB464"])
def main():
    # All the train test files
    traces = ["1000genome", "cycles", "epigenomics", "montage", "seismology", "soykb"]
    feature_id = ['constant', 'num_descendants', 'out_degree_betweenness_centrality', 'trophic_levels']
    num_machine_combos = [3, 10, 50, 100]
    for num_machines in num_machine_combos:
        for trace in traces:
            train_file_name = f"real_life_traces/{trace}_machine_{num_machines}_train.csv"
            test_file_name = f"real_life_traces/{trace}_machine_{num_machines}_test.csv"
            # Make the df_train_dataset
            df_train = create_dataset(num_machines, train_file_name)
            ##### Linear Regression #####
            # Create X, Y dataset
            df_features = pd.DataFrame(columns = feature_id)
            df_psize = pd.DataFrame(columns = ["psize"])
            for index, row in df_train.iterrows():
                for feature in row["features"]:
                    df_features = df_features.append(feature, ignore_index=True)
                for psize in row["psize"]:
                    df_psize = df_psize.append({"psize": psize}, ignore_index=True)   
            X = df_features[feature_id]
            Y = df_psize[["psize"]]
            model=sm.OLS(Y, X.astype(float)).fit()
            print_model=model.summary()
            print(print_model)

            # weights learned from LR
            lr_coefficients = np.array(model.params)
            print(lr_coefficients)

            lr_lst = []

            for index, row in df_train.iterrows():
                # predict using LR model
                psizes = reset_psizes(predict(lr_coefficients, row["features"]))
                speeds = psize_to_speed(psizes)
                G = nx.node_link_graph(row["graph_object"])
        
                time_intervals = native_rescheduler(deepcopy(G), deepcopy(speeds), deepcopy(row["weights"]), deepcopy(row["order"]))
                cost, power, time = compute_cost(row["weights"], time_intervals, speeds)
                lr_lst.append(cost)
    
            # update costs
            new_df = pd.DataFrame({'LR_cost': lr_lst})
            df_train.update(new_df)

            # for initialization purposes only
            new_df = pd.DataFrame({'GD_cost': lr_lst})
            df_train.update(new_df)

            ##### GD ALGORITHM TRAIN#####
            df_train_results, coef_train = gd_algorithm(lr_coefficients, df_train)
            print('GD Coefficients Train: {coef_train}')

            ## Save df_train_results
            save_train_name = f"real_life_traces/results/{trace}_machine_{num_machines}_train.pkl"
            save_df(df_train_results, save_train_name)
            ### TESTING
            count = 0

            df_test = create_dataset(num_machines, test_file_name)

            lr_lst = []
            gd_lst = []

            for index, row in df_test.iterrows():
                # predict using LR model
                psizes_lr = reset_psizes(predict(lr_coefficients, row["features"]))
                speeds_lr = psize_to_speed(psizes_lr)
                G = nx.node_link_graph(row["graph_object"])
                time_intervals_lr = native_rescheduler(deepcopy(G), deepcopy(speeds_lr), deepcopy(row["weights"]), deepcopy(row["order"]))
                cost_lr, power_lr, time_lr = compute_cost(row["weights"], time_intervals_lr, speeds_lr)
                lr_lst.append(cost_lr)

            # update costs
            new_df_lr = pd.DataFrame({'LR_cost': lr_lst})
            df_test.update(new_df_lr)    

            ##### GD ALGORITHM TEST #####
            df_results, coef_test = gd_algorithm(lr_coefficients, df_test)

            ## Save df_results
            save_test_name = f"real_life_traces/results/{trace}_machine_{num_machines}_test.pkl"
            save_df(df_results, save_test_name)
            print("________________________________________________________________________________")
    
    return


if __name__ == "__main__":
    main()