# datasets_simulation/vary_weights_ruido.py
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

def generate_weight_scenarios_with_noise(m):
    base_weights = []
    base_weights.append([1/m]*m)
    base_weights.append(np.linspace(1, 2, m))
    base_weights.append(np.linspace(2, 1, m))
    base_weights.append(np.linspace(1, 5, m))
    base_weights.append(np.linspace(5, 1, m))
    base_weights.append([i+1 for i in range(m)])
    base_weights.append([1/(i+1) for i in range(m)])
    base_weights.append([0.1]*5 + [0.9/(m-5)]*(m-5))
    base_weights.append([0.9/(m-5)]*(m-5) + [0.1]*5)
    base_weights.append([0.05]*10 + [0.5] + [0.45/(m-11)]*(m-11))

    base_weights = [np.array(w)/np.sum(w)*0.95 for w in base_weights]  # escalar por 0.95
    scenarios = [np.append(w, 0.05) for w in base_weights]  # agregar 5% de ruido
    scenarios = [w/np.sum(w) for w in scenarios]  # normalizar
    return scenarios

def generate_graph_instance(n, m, weights, seed, u_list):
    np.random.seed(seed)
    p = int(n / 2 + 1)
    mu = np.zeros(p - 1)
    noise_direction = np.random.normal(size=p)
    noise_direction /= np.linalg.norm(noise_direction)
    u_list = u_list + [noise_direction]
    Sigmas = [np.diag([1 / (5 * p)] * (p - 1)) for _ in range(m)] + [np.identity(p - 1)]
    X = generate_mixture_sphere_sample(mu, Sigmas, u_list, weights, n)
    G = generate_random_spherical_graph(X)
    return G, {"n": n, "p": p, "m": m, "weights": weights.tolist(), "seed": seed, "noise": True}

def save_graph(G, metadata, folder, idx):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"graph_{idx:03d}.pkl")
    with open(path, "wb") as f:
        pickle.dump({"graph": G, "metadata": metadata}, f)

def generate_dataset_for_n(n, num_graphs=50):
    m = 15
    p = int(n / 2 + 1)
    u_list = create_basis_u_list(p, m)
    weight_scenarios = generate_weight_scenarios_with_noise(m)
    folder = f"data/vary_weights_ruido/n{n}"
    for i in tqdm(range(num_graphs)):
        weights = weight_scenarios[i % len(weight_scenarios)]
        G, metadata = generate_graph_instance(n, m, weights, seed=4000 * n + i, u_list=u_list)
        metadata["dataset_type"] = "vary_weights_ruido"
        save_graph(G, metadata, folder, i)

if __name__ == "__main__":
    for n in [2000]:
        generate_dataset_for_n(n)
