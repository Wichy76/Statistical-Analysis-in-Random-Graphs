# datasets_simulation/vary_num_communities.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import numpy as np
from tqdm import tqdm
from core.generators import generate_mixture_sphere_sample, generate_random_spherical_graph


def create_basis_u_list(p, m):
    basis = [np.eye(p)[i] for i in range(p)]
    u_list = []
    i = 0
    while len(u_list) < m:
        u_list.append(basis[i % p])
        if len(u_list) < m:
            u_list.append(-basis[i % p])
        i += 1
    return u_list


def generate_graph_instance(n, m, seed):
    np.random.seed(seed)
    p = int(n / 2 + 1)
    mu = np.zeros(p - 1)
    u_list = create_basis_u_list(p, m)
    weights = [1 / m] * m
    Sigmas = [np.diag([1 / (5 * p)] * (p - 1)) for _ in range(m)]
    X = generate_mixture_sphere_sample(mu, Sigmas, u_list, weights, n)
    G = generate_random_spherical_graph(X)
    return G, {"n": n, "p": p, "m": m, "seed": seed}
 

def save_graph(G, metadata, folder, idx):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"graph_{idx:03d}.pkl")
    with open(path, "wb") as f:
        pickle.dump({"graph": G, "metadata": metadata}, f)


def generate_dataset_for_n(n, num_graphs=50):
    possible_ms = [1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50]
    folder = f"data/vary_num_communities/n{n}"
    for i in tqdm(range(num_graphs)):
        m = np.random.choice(possible_ms)
        G, metadata = generate_graph_instance(n, m, seed=1000 * n + i)
        metadata["dataset_type"] = "vary_num_communities"
        save_graph(G, metadata, folder, i)


if __name__ == "__main__":
    for n in [100, 500, 1000]:
        generate_dataset_for_n(n)
