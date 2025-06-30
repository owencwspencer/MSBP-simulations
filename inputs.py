import networkx as nx
import numpy as np
import random
import csv
import json
import multiprocessing

n = 1000
initial_activated_count = 10
p_values = np.arange(0.01, 0.2, 0.01)
k1_values = [2, 3, 4]
sigma_values = np.arange(0.05, 0.9, 0.05)
num_experiments = 10

def worker_init(graph, initial):
    global G_shared, initial_shared
    G_shared = graph
    initial_shared = initial

def spread_activation(G, node_states, k1, k2, sigma):
    nodes_k2 = set()
    nodes_k1 = set()
    for node in G.nodes():
        if node_states[node] != 1:
            total = sum(
                1 if node_states[nbr] == 1
                else sigma if node_states[nbr] == sigma
                else 0
                for nbr in G.neighbors(node)
            )
            if total >= k2:
                nodes_k2.add(node)
            elif total >= k1:
                nodes_k1.add(node)
    transitioned = nodes_k1 | nodes_k2
    weak_contrib = sum(
        sum(1 for nbr in G.neighbors(node) if node_states[nbr] == sigma)
        for node in transitioned
    )
    strong_contrib = sum(
        sum(1 for nbr in G.neighbors(node) if node_states[nbr] == 1)
        for node in transitioned
    )
    for node in nodes_k2:
        node_states[node] = 1
    for node in nodes_k1:
        if node_states[node] != 1:
            node_states[node] = sigma
    return weak_contrib, strong_contrib

def run_simulation(G, initial, k1, k2, sigma, strong_initial):
    if strong_initial:
        node_states = {node: 1 if node in initial else 0 for node in G.nodes()}
    else:
        node_states = {node: sigma if node in initial else 0 for node in G.nodes()}
    weak_list = []
    strong_list = []
    while True:
        weak, strong = spread_activation(G, node_states, k1, k2, sigma)
        if weak == 0 and strong == 0:
            break
        weak_list.append(weak)
        strong_list.append(strong)
    n_steps = len(weak_list)
    return n_steps, weak_list, strong_list

def increase_degree(G, delta, rng):
    stubs = [node for node in G.nodes() for _ in range(delta)]
    rng.shuffle(stubs)
    for i in range(0, len(stubs)-1, 2):
        u, v = stubs[i], stubs[i+1]
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)

def worker_task(params):
    graph_id, p, k1, k2, sigma, strong_initial = params
    n_steps, weak_list, strong_list = run_simulation(G_shared, initial_shared, k1, k2, sigma, strong_initial)
    return [k1, k2, sigma, p, graph_id, strong_initial, n_steps,
            json.dumps(weak_list), json.dumps(strong_list)]

if __name__ == '__main__':
    multiprocessing.freeze_support()
    csv_filename = 'inputs_results.csv'
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'k1','k2','sigma','p','graph','strong_initial','n_steps',
            'weak_contrib','strong_contrib'
        ])
        for graph_id in range(1, num_experiments+1):
            graph_seed = graph_id
            degree0 = int(round(p_values[0] * (n - 1)))
            G = nx.random_regular_graph(degree0, n, seed=graph_seed)
            prev_degree = degree0
            nodes = list(G.nodes())
            initials = []
            for init_idx in range(num_experiments):
                seed_init = graph_seed * num_experiments + init_idx
                rng_init = np.random.RandomState(seed_init)
                initials.append(rng_init.choice(nodes, initial_activated_count, replace=False))
            for p_idx, p in enumerate(p_values, start=1):
                degree = int(round(p * (n - 1)))
                if p_idx > 1:
                    rng_py = random.Random(graph_seed + p_idx)
                    increase_degree(G, degree - prev_degree, rng_py)
                    prev_degree = degree
                for initial in initials:
                    params = [
                        (graph_id, p, k1, k2, sigma, strong_initial)
                        for strong_initial in (False, True)
                        for k1 in k1_values
                        for k2 in (k1+1, k1+2, k1+3)
                        for sigma in sigma_values
                    ]
                    with multiprocessing.Pool(initializer=worker_init, initargs=(G, initial)) as pool:
                        results = pool.map(worker_task, params)
                    writer.writerows(results)
                    f.flush()
    print(f"Processing complete. Results saved to {csv_filename}")
