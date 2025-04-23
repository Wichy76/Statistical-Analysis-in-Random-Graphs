# HypothesisTestFixednFixedp.py

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import community as community_louvain  # algoritmo de Louvain

# Importamos sólo la función de carga de grafos simulados
from SimulateG1G2 import simulate_or_load

# ------------------------
# Estadístico de triángulos normalizado
# ------------------------

def normalized_triangles(G):
    t = sum(nx.triangles(G).values()) / 3
    k = G.number_of_nodes()
    max_tri = k * (k - 1) * (k - 2) / 6
    return t / max_tri if max_tri > 0 else 0

# ------------------------
# Test de hipótesis no paramétrico
# ------------------------

def hypothesis_test_triangles(G1, G2, b=200, k=None):
    if k is None:
        k = int(round(G1.number_of_nodes()**(2/3)))

    stats_G1 = []
    for _ in range(b):
        sampled = np.random.choice(G1.nodes(), k, replace=False)
        subG = G1.subgraph(sampled)
        stats_G1.append(normalized_triangles(subG))

    ci_low, ci_high = np.quantile(stats_G1, [0.025, 0.975])

    sampled2 = np.random.choice(G2.nodes(), k, replace=False)
    subG2 = G2.subgraph(sampled2)
    stat_G2 = normalized_triangles(subG2)

    inside = ci_low <= stat_G2 <= ci_high

    print(f"Estadístico en G2: {stat_G2:.4f}")
    print(f"Intervalo 95% bootstrap G1: [{ci_low:.4f}, {ci_high:.4f}]")
    print("NO se rechaza H0" if inside else "Se rechaza H0")

    plt.scatter(range(b), stats_G1, color="gray")
    plt.axhline(ci_low, color="blue", linestyle="dashed")
    plt.axhline(ci_high, color="blue", linestyle="dashed")
    plt.scatter(b+1, stat_G2, color="green" if inside else "red")
    plt.text(b+2, stat_G2, "G2", color="green" if inside else "red")
    plt.xlabel("Bootstrap iteration")
    plt.ylabel("Triángulos normalizados")
    plt.title(f"b={b}, k={k}")
    plt.show()

# ------------------------
# Comparar densidades de vecinos comunes
# ------------------------

def compute_common_neighbor_counts(G):
    nodes     = list(G.nodes())
    neighbors = {u:set(G.neighbors(u)) for u in nodes}
    counts    = []
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            common = len(neighbors[nodes[i]] & neighbors[nodes[j]])
            counts.append(common)
    return np.array(counts)

# ------------------------
# MAIN
# ------------------------

if __name__ == "__main__":
    cache_file = "simulated_graphs.pkl"
    params = (
        1000,   # n
        26,     # p
        100000, # sigma1
        50,     # m1
        [1/50]*50,
        100000, # sigma2
        45,     # m2
        [1/45]*45,
        42      # seed
    )

    # 1) Simula o carga los grafos G1, G2
    G1, G2 = simulate_or_load(cache_file, params, force=False)

    print(f"n1={G1.number_of_nodes()}, n2={G2.number_of_nodes()}")
    print(f"m1={G1.number_of_edges()}, m2={G2.number_of_edges()}")

    # 2) Test de los triángulos
    hypothesis_test_triangles(G1, G2, b=200)

    # 3) Comunidades con Louvain
    nc1 = len(set(community_louvain.best_partition(G1).values()))
    nc2 = len(set(community_louvain.best_partition(G2).values()))
    print(f"Comunidades G1: {nc1}, G2: {nc2}")

    # 4) Densidades de vecinos comunes
    print("Vecinos comunes G1...")
    cc1 = compute_common_neighbor_counts(G1)
    print("Vecinos comunes G2...")
    cc2 = compute_common_neighbor_counts(G2)

    dens1 = gaussian_kde(cc1)
    dens2 = gaussian_kde(cc2)
    x = np.linspace(0, max(cc1.max(), cc2.max()), 500)

    plt.figure(figsize=(8,5))
    plt.plot(x, dens1(x), label="G1")
    plt.plot(x, dens2(x), label="G2")
    plt.xlabel("Vecinos comunes")
    plt.ylabel("Densidad")
    plt.title("Densidades de vecinos comunes")
    plt.legend()
    plt.show()
