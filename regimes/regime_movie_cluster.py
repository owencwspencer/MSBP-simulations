import networkx as nx
import numpy as np
import csv
import json
import multiprocessing
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

n = 1000
initial_activated_count = 10

p = 0.39
k1 = 5
k2 = 6
sigma = 0.8

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
    history = [node_states.copy()]
    while True:
        i_w, i_s, w_s = spread_activation(None, node_states, k1, k2, sigma)
        if i_w == 0 and i_s == 0 and w_s == 0:
            break
        history.append(node_states.copy())
    return history

def worker_task(params):
    exp, p, k1, k2, sigma = params
    history = run_simulation(k1, k2, sigma)
    return history

if __name__ == '__main__':
    multiprocessing.freeze_support()
    graph_seed = 1
    degree = int(round(p * (n - 1)))
    G = nx.random_regular_graph(degree, n, seed=graph_seed)
    adjacency = {u: list(G.adj[u]) for u in G.nodes()}
    nodes = list(G.nodes())
    rng_init = np.random.RandomState(0)
    initial = rng_init.choice(nodes, initial_activated_count, replace=False)
    worker_init(adjacency, initial)
    history = run_simulation(k1, k2, sigma)
    pos = nx.spring_layout(G, seed=0)
    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame):
        ax.clear()
        state = history[frame]
        colors = []
        for node in G.nodes():
            val = state[node]
            if val == 0:
                colors.append('lightgray')
            elif val == sigma:
                colors.append('blue')
            elif val == 1:
                colors.append('red')
        nx.draw_networkx_nodes(G, pos, node_color=colors, ax=ax, node_size=20)
        nx.draw_networkx_edges(G, pos, ax=ax, width=0.2)
        ax.set_axis_off()
        ax.set_title(f"Step {frame}")

    ani = animation.FuncAnimation(fig, update, frames=len(history), interval=200)

    writer = animation.FFMpegWriter(fps=5)
    ani.save(f'cluster_{k1}_{k2}_{sigma}_{p}.mp4', writer=writer)
