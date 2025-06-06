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
    s = [0.01]*m; s[0] = 0.81; scenarios.append(s)
    scenarios.append([0.3] + [0.7/(m-1)]*(m-1))
    scenarios.append([0.7/(m-1)]*(m-1) + [0.3])
    scenarios.append(np.random.dirichlet(np.arange(1, m+1)).tolist())
    scenarios.append(np.random.dirichlet(np.ones(m)*10).tolist())
    scenarios = [np.array(w)/sum(w) for w in scenarios]
    return scenarios

def generate_graph_instance(n, m, weights, s, u_list, seed, weights_id):
    np.random.seed(seed)
    p = int(n / 2 + 1)
    mu = np.zeros(p - 1)

    weights = [w * 0.95 for w in weights[:m]]
    weights.append(0.05)
    weights = np.array(weights) / np.sum(weights)

    Sigmas = [np.diag([1 / s] * (p - 1)) for _ in range(m)]
    Sigmas.append(np.identity(p - 1))

    noise_direction = np.random.normal(size=p)
    noise_direction /= np.linalg.norm(noise_direction)
    u_list = u_list + [noise_direction]

    X = generate_mixture_sphere_sample(mu, Sigmas, u_list, weights, n)
    G = generate_random_spherical_graph(X)
    return G, {"n": n, "p": p, "m": m, "seed": seed, "weights_id": weights_id, "sigma": s, "noise": True}

def save_graph(G, metadata, folder, idx):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"graph_{idx:03d}.pkl")
    with open(path, "wb") as f:
        pickle.dump({"graph": G, "metadata": metadata}, f)

def generate_dataset_for_n(n, num_graphs=50):
    possible_ms = [1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 24, 28, 32, 36, 40, 45, 50, 55, 60, 66, 72, 78, 84, 91, 98, 105, 112, 120, 128]
    possible_sigmas = [3*n, 5*n, 7*n, 9*n, 11*n, 13*n, 15*n, 17*n, 19*n, 21*n, 23*n, 25*n, 27*n, 29*n, 31*n, 33*n, 35*n, 37*n, 39*n, 41*n, 43*n, 45*n]

    folder = f"data/vary_all_ruido/n{n}"
    for i in tqdm(range(num_graphs)):
        m = np.random.choice(possible_ms)
        s = np.random.choice(possible_sigmas)
        weights_list = generate_weight_scenarios(m)
        weights_id = np.random.randint(len(weights_list))
        weights = weights_list[weights_id]
        p = int(n / 2 + 1)
        u_list = create_basis_u_list(p, m)
        G, metadata = generate_graph_instance(n, m, weights, s, u_list, seed=8000 * n + i, weights_id=weights_id)
        metadata["dataset_type"] = "vary_all_ruido"
        save_graph(G, metadata, folder, i)

if __name__ == "__main__":
    for n in [2000]:
        generate_dataset_for_n(n)
