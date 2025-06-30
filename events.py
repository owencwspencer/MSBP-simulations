import networkx as nx
import numpy as np
import csv
import multiprocessing
import os
import random
import itertools

sigma = 0.5
initial_activated_count = 10

def increase_edges(G, delta, rng, absent_edges):
    new_edges = rng.sample(absent_edges, delta)
    for u, v in new_edges:
        G.add_edge(u, v)
        absent_edges.remove((u, v))

def spread_activation(node_states, k1_param, k2_param, sigma_param):
    nodes_k2 = set()
    nodes_k1 = set()
    for node in NODES:
        if node_states[node] != 1:
            total = sum(
                1 if node_states[nbr] == 1
                else sigma_param if node_states[nbr] == sigma_param
                else 0
                for nbr in ADJ[node]
            )
            if total >= k2_param:
                nodes_k2.add(node)
            elif k1_param is not None and total >= k1_param:
                nodes_k1.add(node)
    for node in nodes_k2:
        node_states[node] = 1
    for node in nodes_k1:
        if node_states[node] != 1:
            node_states[node] = sigma_param
    return nodes_k2, nodes_k1

def single_run(k1_param, k2_param, sigma_param, initial_nodes):
    node_states = {node: 1 if node in initial_nodes else 0 for node in NODES}
    step_count = 0
    early_strong = False
    self_activated = 0
    total_strong = 0
    prev_strong = sum(1 for s in node_states.values() if s == 1)
    prev_weak = sum(1 for s in node_states.values() if s == sigma_param)
    while True:
        step_count += 1
        old_states = node_states.copy()
        k2_set, k1_set = spread_activation(node_states, k1_param, k2_param, sigma_param)
        if step_count == 1 and k2_set:
            early_strong = True
        for node in k2_set:
            num_strong_neighbors = sum(1 for nbr in ADJ[node] if old_states[nbr] == 1)
            total_strong += 1
            if num_strong_neighbors >= k2_param:
                self_activated += 1
        curr_strong = sum(1 for s in node_states.values() if s == 1)
        curr_weak = sum(1 for s in node_states.values() if s == sigma_param)
        if curr_strong == prev_strong and curr_weak == prev_weak:
            break
        prev_strong, prev_weak = curr_strong, curr_weak
    if total_strong > 0:
        frac = self_activated / total_strong
    else:
        frac = 0.0
    full_strong = (curr_strong == len(NODES))
    full_weak = (curr_strong + curr_weak == len(NODES))
    return full_strong, early_strong, full_weak, frac

def worker_init(adjacency):
    global ADJ, NODES
    ADJ = adjacency
    NODES = list(adjacency.keys())

def worker_task(params):
    k1_param, k2_param, sigma_param, initial_nodes = params
    return single_run(k1_param, k2_param, sigma_param, initial_nodes)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    n_list = [50, 100, 200, 500, 1000]
    num_tests = 30
    num_p_values = 100
    p_array = np.linspace(0.01, 1.00, num_p_values)
    num_experiments_per_trial = 30
    k_configurations = [
        {'k1': 1, 'k2': 6}, {'k1': 1, 'k2': 7}, {'k1': 1, 'k2': 8},
        {'k1': 2, 'k2': 6}, {'k1': 2, 'k2': 7}, {'k1': 2, 'k2': 8},
        {'k1': 3, 'k2': 6}, {'k1': 3, 'k2': 7}, {'k1': 3, 'k2': 8}
    ]
    csv_filename = 'full_results_strong.csv'
    csv_headers = [
        'k1', 'k2', 'sigma', 'n', 'test_num', 'trial_num', 'experiment_num',
        'p_value', 'full_strong', 'early_strong', 'full_weak', 'prop_strong_via_strong'
    ]
    file_exists = os.path.exists(csv_filename)
    write_headers = not file_exists or os.path.getsize(csv_filename) == 0
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_headers:
            writer.writerow(csv_headers)
        for n_val in n_list:
            for test_idx in range(num_tests):
                seed_graph = test_idx + n_val * 100000
                base_G = nx.erdos_renyi_graph(n_val, p_array[0], seed=seed_graph)
                nodes = list(base_G.nodes())
                total_possible_edges = n_val * (n_val - 1) // 2
                base_absent_edges = set(itertools.combinations(nodes, 2)) - set(base_G.edges())
                base_prev_edges = base_G.number_of_edges()
                initials = []
                for exp_idx in range(num_experiments_per_trial):
                    seed_init = test_idx * num_experiments_per_trial + exp_idx
                    rng_init = np.random.RandomState(seed_init)
                    initials.append(rng_init.choice(nodes, initial_activated_count, replace=False))
                for config_run in k_configurations:
                    current_k1 = config_run['k1']
                    current_k2 = config_run['k2']
                    G = base_G.copy()
                    absent_edges = base_absent_edges.copy()
                    prev_edges = base_prev_edges
                    for p_idx, p_val in enumerate(p_array):
                        print(f"Config {current_k1}/{current_k2}  n={n_val}  test={test_idx+1}/{num_tests}  p={p_idx+1}/{len(p_array)} ({p_val:.2f})")
                        if p_idx > 0:
                            desired_edges = int(round(p_val * total_possible_edges))
                            delta = desired_edges - prev_edges
                            if delta > 0:
                                available = len(absent_edges)
                                if delta > available:
                                    print(f"  Warning: requested {delta} edges but only {available} available; capping to {available}")
                                    delta = available
                                rng_graph = random.Random(seed_graph + p_idx)
                                increase_edges(G, delta, rng_graph, absent_edges)
                            prev_edges = desired_edges
                        adjacency = {u: list(G.adj[u]) for u in G.nodes()}
                        tasks = [
                            (current_k1, current_k2, sigma, initials[exp_idx])
                            for exp_idx in range(num_experiments_per_trial)
                        ]
                        with multiprocessing.Pool(initializer=worker_init, initargs=(adjacency,)) as pool:
                            results = pool.map(worker_task, tasks)
                        rows = []
                        for exp_idx, res in enumerate(results):
                            full_strong, early_strong, full_weak, frac = res
                            rows.append([
                                current_k1, current_k2, sigma, n_val, test_idx,
                                p_idx, exp_idx, p_val,
                                full_strong, early_strong, full_weak, frac
                            ])
                        writer.writerows(rows)
                        file.flush()
    print(f"Processing complete. Results saved to {csv_filename}")
