# datasets_simulation/vary_sigma.py
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

def generate_graph_instance(n, m, s, seed, u_list):
    np.random.seed(seed)
    p = int(n / 2 + 1)
    mu = np.zeros(p - 1)
    weights = [1 / m] * m
    Sigmas = [np.diag([1 / s] * (p - 1)) for _ in range(m)]
    X = generate_mixture_sphere_sample(mu, Sigmas, u_list, weights, n)
    G = generate_random_spherical_graph(X)
    return G, {"n": n, "p": p, "m": m, "seed": seed, "sigma": s}

def save_graph(G, metadata, folder, idx):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"graph_{idx:03d}.pkl")
    with open(path, "wb") as f:
        pickle.dump({"graph": G, "metadata": metadata}, f)

def generate_dataset_for_n(n, num_graphs=50):
    p = int(n / 2 + 1)
    possible_sigmas = [3*p, 5*p, 7*p, 9*p, 11*p, 13*p, 15*p, 17*p, 19*p, 21*p, 23*p, 25*p, 27*p, 29*p, 31*p, 33*p, 35*p, 37*p, 39*p, 41*p, 43*p, 45*p]
    m = 15
    u_list = create_basis_u_list(p, m)
    folder = f"data/vary_sigma/n{n}"
    for i in tqdm(range(num_graphs)):
        s = np.random.choice(possible_sigmas)
        G, metadata = generate_graph_instance(n, m, s, seed=1000 * n + i, u_list=u_list)
        metadata["dataset_type"] = "vary_sigma"
        save_graph(G, metadata, folder, i)

if __name__ == "__main__":
    for n in [2000]:
        generate_dataset_for_n(n)
