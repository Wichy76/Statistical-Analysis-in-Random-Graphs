# datasets_simulation/vary_weights.py
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

def generate_weight_scenarios(m):
    scenarios = []
    scenarios.append([1/m]*m)  # uniforme
    scenarios.append(np.linspace(1, 2, m))
    scenarios.append(np.linspace(2, 1, m))
    scenarios.append(np.linspace(1, 5, m))
    scenarios.append(np.linspace(5, 1, m))
    scenarios.append([i+1 for i in range(m)])
    scenarios.append([1/(i+1) for i in range(m)])
    scenarios.append(np.random.dirichlet(np.ones(m)).tolist())
    scenarios.append(np.random.dirichlet(np.linspace(1, 2, m)).tolist())
    scenarios.append(np.random.dirichlet(np.linspace(2, 1, m)).tolist())
    scenarios.append([0.1]*5 + [0.9/(m-5)]*(m-5))
    scenarios.append([0.9/(m-5)]*(m-5) + [0.1]*5)
    scenarios.append([0.05]*10 + [0.5] + [0.45/(m-11)]*(m-11))
    scenarios.append([0.5] + [0.5/(m-1)]*(m-1))
    scenarios.append([0.6 if i % 2 == 0 else 0.4/(m-1) for i in range(m)])
    scenarios.append([0.01]*m); scenarios[-1][0] = 0.81
    scenarios.append([0.3] + [0.7/(m-1)]*(m-1))
    scenarios.append([0.7/(m-1)]*(m-1) + [0.3])
    scenarios.append(np.random.dirichlet(np.arange(1, m+1)).tolist())
    scenarios.append(np.random.dirichlet(np.ones(m)*10).tolist())
    scenarios = [np.array(w)/sum(w) for w in scenarios]
    return scenarios

def generate_graph_instance(n, m, weights, seed, u_list):
    np.random.seed(seed)
    p = int(n / 2 + 1)
    mu = np.zeros(p - 1)
    Sigmas = [np.diag([1 / (5 * p)] * (p - 1)) for _ in range(m)]
    X = generate_mixture_sphere_sample(mu, Sigmas, u_list, weights, n)
    G = generate_random_spherical_graph(X)
    return G, {"n": n, "p": p, "m": m, "weights": weights.tolist(), "seed": seed}

def save_graph(G, metadata, folder, idx):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"graph_{idx:03d}.pkl")
    with open(path, "wb") as f:
        pickle.dump({"graph": G, "metadata": metadata}, f)

def generate_dataset_for_n(n, num_graphs=50):
    m = 15
    p = int(n / 2 + 1)
    u_list = create_basis_u_list(p, m)
    weight_scenarios = generate_weight_scenarios(m)
    folder = f"data/vary_weights/n{n}"
    for i in tqdm(range(num_graphs)):
        weights = weight_scenarios[i % len(weight_scenarios)]
        G, metadata = generate_graph_instance(n, m, weights, seed=2000 * n + i, u_list=u_list)
        metadata["dataset_type"] = "vary_weights"
        save_graph(G, metadata, folder, i)

if __name__ == "__main__":
    for n in [2000]:
        generate_dataset_for_n(n)
