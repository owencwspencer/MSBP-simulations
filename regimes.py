import networkx as nx
import numpy as np
import csv
import json
import multiprocessing
import random

n = 1000
initial_activated_count = 10
p_values = np.arange(0.005, 0.3, 0.005)
k1_values = [2, 3, 4]
sigma_values = np.arange(0.4, 0.9, 0.1)

def increase_degree(G, delta, rng):
    stubs = [node for node in G.nodes() for _ in range(delta)]
    rng.shuffle(stubs)
    for i in range(0, len(stubs) - 1, 2):
        u, v = stubs[i], stubs[i+1]
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)

def spread_activation(_G, node_states, k1, k2, sigma):
    nodes_k2 = set()
    nodes_k1 = set()
    for node in node_states:
        if node_states[node] != 1:
            total = sum(
                1 if node_states[nbr] == 1
                else sigma if node_states[nbr] == sigma
                else 0
                for nbr in ADJ[node]
            )
            if total >= k2:
                nodes_k2.add(node)
            elif total >= k1:
                nodes_k1.add(node)
    inactive_to_strong = sum(1 for node in nodes_k2 if node_states[node] == 0)
    weak_to_strong     = sum(1 for node in nodes_k2 if node_states[node] == sigma)
    inactive_to_weak   = sum(1 for node in nodes_k1 if node_states[node] == 0)
    for node in nodes_k2:
        node_states[node] = 1
    for node in nodes_k1:
        if node_states[node] != 1:
            node_states[node] = sigma
    return inactive_to_weak, inactive_to_strong, weak_to_strong

def worker_init(adj_dict, initial_nodes):
    global ADJ, NODES, initial_global
    ADJ = adj_dict
    NODES = list(adj_dict.keys())
    initial_global = initial_nodes

def run_simulation(k1, k2, sigma):
    node_states = {node: (sigma if node in initial_global else 0) for node in NODES}
    i_w_list = []
    i_s_list = []
    w_s_list = []
    n_inactive_list = []
    n_weak_list = []
    n_strong_list = []
    while True:
        i_w, i_s, w_s = spread_activation(None, node_states, k1, k2, sigma)
        if i_w == 0 and i_s == 0 and w_s == 0:
            break
        i_w_list.append(i_w)
        i_s_list.append(i_s)
        w_s_list.append(w_s)
        vals = list(node_states.values())
        n_inactive_list.append(vals.count(0))
        n_weak_list.append(vals.count(sigma))
        n_strong_list.append(vals.count(1))
    n_steps = len(i_w_list)
    return n_steps, i_w_list, i_s_list, w_s_list, n_inactive_list, n_weak_list, n_strong_list

def worker_task(params):
    exp, p, k1, k2, sigma = params
    n_steps, i_w, i_s, w_s, ni, nw, ns = run_simulation(k1, k2, sigma)
    return [
        k1, k2, sigma, p, exp, n_steps,
        json.dumps(i_w), json.dumps(i_s), json.dumps(w_s),
        json.dumps(ni), json.dumps(nw), json.dumps(ns)
    ]

if __name__ == '__main__':
    multiprocessing.freeze_support()
    num_experiments = 30
    csv_filename = 'regimes_results.csv'
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'k1','k2','sigma','p','experiment','n_steps',
            'i_w','i_s','w_s','n_inactive','n_weak','n_strong'
        ])
        for exp in range(1, num_experiments+1):
            graph_seed = exp
            degree0 = int(round(p_values[0] * (n - 1)))
            G = nx.random_regular_graph(degree0, n, seed=graph_seed)
            prev_degree = degree0
            nodes = list(G.nodes())
            initials = []
            for init_idx in range(num_experiments):
                seed_init = exp * num_experiments + init_idx
                rng_init = np.random.RandomState(seed_init)
                initials.append(
                    rng_init.choice(nodes, initial_activated_count, replace=False)
                )
            for p_idx, p in enumerate(p_values, start=1):
                print(f"Exp {exp}: p {p_idx}/{len(p_values)}")
                degree = int(round(p * (n - 1)))
                if p_idx > 1:
                    rng_py = random.Random(graph_seed + p_idx)
                    increase_degree(G, degree - prev_degree, rng_py)
                    prev_degree = degree
                adjacency = {u: list(G.adj[u]) for u in G.nodes()}
                for initial in initials:
                    with multiprocessing.Pool(
                        initializer=worker_init,
                        initargs=(adjacency, initial)
                    ) as pool:
                        tasks = [
                            (exp, p, k1, k2, sigma)
                            for k1 in k1_values
                            for k2 in (k1+1, k1+2, k1+3)
                            for sigma in sigma_values
                        ]
                        results = pool.map(worker_task, tasks)
                        writer.writerows(results)
                        f.flush()
    print(f"Processing complete. Results saved to {csv_filename}")
