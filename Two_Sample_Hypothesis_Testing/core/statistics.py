import numpy as np
import networkx as nx
from scipy.stats import gaussian_kde
import community as community_louvain


def modularity(G):
    partition = community_louvain.best_partition(G)
    return community_louvain.modularity(partition, G)


def kde_l1_linf_ratio(counts, num_points=1000):
    xs = np.linspace(0, counts.max(), num_points)
    dens = gaussian_kde(counts)(xs)
    dx = xs[1] - xs[0]
    l1 = np.sum(np.abs(dens)) * dx
    linf = np.max(np.abs(dens))
    return l1 / linf if linf > 0 else 0.0


def common_neighbor_density(G, num_samples=10000):
    nodes = list(G.nodes())
    n = len(nodes)
    k = 0
    counts = []
    seen = set()
    while k < num_samples:
        i, j = np.random.choice(n, 2, replace=False)
        if i > j:
            i, j = j, i
        if (i, j) in seen:
            continue
        seen.add((i, j))
        ni = set(G.neighbors(nodes[i]))
        nj = set(G.neighbors(nodes[j]))
        counts.append(len(ni & nj))
        k += 1
    return kde_l1_linf_ratio(np.array(counts))


def normalized_triangles(G):
    t = sum(nx.triangles(G).values()) / 3
    k = G.number_of_nodes()
    max_tri = k * (k - 1) * (k - 2) / 6
    return t / max_tri if max_tri > 0 else 0.0

def log_avg_degree(G):
    n = G.number_of_nodes()
    degrees = [d for _, d in G.degree()]
    return np.log(np.mean(degrees) + 1)

def f_edges_normalized(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    return 2 * m / (n * (n - 1)) if n > 1 else 0.0


AVAILABLE_STATS = {
    "f1": modularity,
    "f2": normalized_triangles,
    "f3": common_neighbor_density,
    "f4": log_avg_degree,
    "f5": f_edges_normalized
}


def compute_bivariate_stats(G, k=None, stat_names=["f1", "f2"], seed=100):
    np.random.seed(seed)
    if k is None:
        k = int(round(G.number_of_nodes() ** (2 / 3)))
    nodes = np.random.choice(list(G), k, replace=False)
    sub = G.subgraph(nodes)
    results = []
    for name in stat_names:
        stat_func = AVAILABLE_STATS[name]
        results.append(stat_func(sub))
    return np.array(results)
