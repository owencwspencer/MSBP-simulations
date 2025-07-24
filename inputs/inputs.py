import networkx as nx
import numpy as np
import csv
import json
import multiprocessing
import random
import numba

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

def convert_adj_to_csr_format(adj_dict, num_nodes):
    adj_indices = np.zeros(num_nodes + 1, dtype=np.int64)
    all_neighbors = []
    
    current_pos = 0
    for i in range(num_nodes):
        neighbors = adj_dict.get(i, [])
        adj_indices[i] = current_pos
        all_neighbors.extend(neighbors)
        current_pos += len(neighbors)
    adj_indices[num_nodes] = current_pos
    
    adj_data = np.array(all_neighbors, dtype=np.int64)
    return adj_data, adj_indices

@numba.jit(nopython=True)
def _spread_activation_numba(node_states, k1, k2, sigma, adj_data, adj_indices):
    nodes_to_k1 = []
    nodes_to_k2 = []

    for node in range(len(node_states)):
        if node_states[node] != 1.0:
            total_influence = 0.0
            start, end = adj_indices[node], adj_indices[node+1]
            for i in range(start, end):
                nbr = adj_data[i]
                nbr_state = node_states[nbr]
                if nbr_state == 1.0:
                    total_influence += 1.0
                elif nbr_state == sigma:
                    total_influence += sigma
            
            if total_influence >= k2:
                nodes_to_k2.append(node)
            elif total_influence >= k1:
                nodes_to_k1.append(node)

    weak_by_weak, weak_by_strong = 0, 0
    strong_by_weak, strong_by_strong = 0, 0
    inactive_to_weak, inactive_to_strong, weak_to_strong = 0, 0, 0

    if len(nodes_to_k1) > 0:
        k1_nodes_arr = np.array(nodes_to_k1, dtype=np.int64)
        for node in k1_nodes_arr:
            if node_states[node] == 0.0:
                inactive_to_weak += 1
                w_nbrs, s_nbrs = 0, 0
                start, end = adj_indices[node], adj_indices[node+1]
                for i in range(start, end):
                    nbr = adj_data[i]
                    if node_states[nbr] == sigma: w_nbrs += 1
                    elif node_states[nbr] == 1.0: s_nbrs += 1
                weak_by_weak += w_nbrs
                weak_by_strong += s_nbrs

    if len(nodes_to_k2) > 0:
        k2_nodes_arr = np.array(nodes_to_k2, dtype=np.int64)
        for node in k2_nodes_arr:
            if node_states[node] == 0.0: inactive_to_strong += 1
            else: weak_to_strong += 1
            
            w_nbrs, s_nbrs = 0, 0
            start, end = adj_indices[node], adj_indices[node+1]
            for i in range(start, end):
                nbr = adj_data[i]
                if node_states[nbr] == sigma: w_nbrs += 1
                elif node_states[nbr] == 1.0: s_nbrs += 1
            strong_by_weak += w_nbrs
            strong_by_strong += s_nbrs
    
    if len(nodes_to_k2) > 0:
        for node in k2_nodes_arr:
            node_states[node] = 1.0
    
    if len(nodes_to_k1) > 0:
        for node in k1_nodes_arr:
            if node_states[node] != 1.0:
                node_states[node] = sigma

    return (inactive_to_weak, inactive_to_strong, weak_to_strong,
            weak_by_weak, weak_by_strong, strong_by_weak, strong_by_strong)

def worker_init(adj_data, adj_indices):
    global ADJ_DATA, ADJ_INDICES
    ADJ_DATA = adj_data
    ADJ_INDICES = adj_indices

def run_simulation(k1, k2, sigma, initial_nodes):
    node_states = np.zeros(n, dtype=np.float64)
    node_states[initial_nodes] = sigma
    
    i_w_list, i_s_list, w_s_list = [], [], []
    w_by_w_list, w_by_s_list, s_by_w_list, s_by_s_list = [], [], [], []

    while True:
        res = _spread_activation_numba(node_states, k1, k2, sigma, ADJ_DATA, ADJ_INDICES)
        i_w, i_s, w_s, w_by_w, w_by_s, s_by_w, s_by_s = res
        if i_w == 0 and i_s == 0 and w_s == 0: break
        
        i_w_list.append(i_w)
        i_s_list.append(i_s)
        w_s_list.append(w_s)
        w_by_w_list.append(w_by_w)
        w_by_s_list.append(w_by_s)
        s_by_w_list.append(s_by_w)
        s_by_s_list.append(s_by_s)

    n_steps = len(i_w_list)
    return (n_steps, i_w_list, i_s_list, w_s_list,
            w_by_w_list, w_by_s_list, s_by_w_list, s_by_s_list)

def worker_task(params):
    exp, p, k1, k2, sigma, initial = params
    res = run_simulation(k1, k2, sigma, initial)
    n_steps, i_w, i_s, w_s, w_by_w, w_by_s, s_by_w, s_by_s = res
    return [
        k1, k2, sigma, p, exp, n_steps,
        json.dumps(i_w), json.dumps(i_s), json.dumps(w_s),
        json.dumps(w_by_w), json.dumps(w_by_s),
        json.dumps(s_by_w), json.dumps(s_by_s)
    ]

if __name__ == '__main__':
    multiprocessing.freeze_support()
    num_experiments = 30
    csv_filename = 'regimes_results_optimized.csv'
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'k1','k2','sigma','p','experiment','n_steps', 'i_w','i_s','w_s',
            'weak_by_weak', 'weak_by_strong', 'strong_by_weak', 'strong_by_strong'
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
                adj_data, adj_indices = convert_adj_to_csr_format(adjacency, n)
                
                with multiprocessing.Pool(
                    initializer=worker_init,
                    initargs=(adj_data, adj_indices)
                ) as pool:
                    tasks = [
                        (exp, p, k1, k2, sigma, initial)
                        for initial in initials
                        for k1 in k1_values
                        for k2 in (k1+1, k1+2, k1+3)
                        for sigma in sigma_values
                    ]
                    results = pool.map(worker_task, tasks)
                    writer.writerows(results)
                    f.flush()
    print(f"Processing complete. Results saved to {csv_filename}")
    