import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import random
import numpy as np
import pickle
import yaml
from sklearn.covariance import EmpiricalCovariance
from scipy.stats import chi2
from core.statistics import compute_bivariate_stats

# Armónicos esféricos en S^1: cos(kθ), sin(kθ) para k=3,4,5,6
def spherical_harmonics(theta):
    return np.array([
        np.cos(3 * theta), np.sin(3 * theta),
        np.cos(4 * theta), np.sin(4 * theta),
        np.cos(5 * theta), np.sin(5 * theta),
        np.cos(6 * theta), np.sin(6 * theta),
    ])

def compute_Zn_squared(X, epsilon=0.05):
    n = len(X)
    mean = np.mean(X, axis=0)
    cov = np.cov(X.T)
    inv_sqrt_cov = np.linalg.inv(np.linalg.cholesky(cov)).T
    Y = (X - mean) @ inv_sqrt_cov.T

    norms = np.linalg.norm(Y, axis=1)
    rn = np.quantile(norms, epsilon)
    mask = norms > rn
    Y_kept = Y[mask]
    W = Y_kept / np.linalg.norm(Y_kept, axis=1)[:, None]

    theta = np.arctan2(W[:,1], W[:,0])
    H_vals = np.array([spherical_harmonics(t) for t in theta])
    Qn = H_vals.mean(axis=0)
    Zn_squared = len(W) * np.sum(Qn**2)
    return Zn_squared

def load_graph_dataset(path):
    files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(".pkl")])
    dataset = []
    for f in files:
        with open(f, "rb") as g:
            obj = pickle.load(g)
            dataset.append((obj["graph"], obj["metadata"]))
    return dataset

def test_symmetry_on_graph(G, stat_names, k, seed):
    np.random.seed(seed)
    stats = np.array([compute_bivariate_stats(G, k, stat_names, seed+i) for i in range(200)])
    return compute_Zn_squared(stats)

def generate_elliptical_data(mean, cov, n=200, seed=0):
    np.random.seed(seed)
    return np.random.multivariate_normal(mean, cov, size=n)

def generate_elliptical_with_noise(mean, cov, noise_std=0.1, n=200, seed=0):
    np.random.seed(seed)
    X = np.random.multivariate_normal(mean, cov, size=n)
    noise = np.random.normal(0, noise_std, size=X.shape)
    return X + noise

def run_symmetry_test(config):
    dataset_path = config["dataset_path"]
    stat_names = config["stat_names"]
    dataset = load_graph_dataset(dataset_path)

    results = []
    alpha = config.get("alpha", 0.05)
    threshold = (1 - 0.05) * chi2.ppf(1 - alpha, df=8)

    print("\n--- Test sobre grafos del dataset ---")
    selected_graphs = random.sample(dataset, 10)
    for i, (G, _) in enumerate(selected_graphs):
        k = int(round(G.number_of_nodes() ** (2/3)))
        Zn2 = test_symmetry_on_graph(G, stat_names, k, seed=1000+i)
        rejected = Zn2 > threshold
        results.append((Zn2, rejected))
        print(f"Graph {i+1}: Z^2 = {Zn2:.2f} → {'Reject' if rejected else 'Accept'}")

    print("\n--- Test sobre datos simulados elipsoidales ---")
    mean = [0, 0]
    cov = [[2.0, 0.7], [0.7, 1.0]]
    for i in range(5):
        X_sim = generate_elliptical_data(mean, cov, n=200, seed=999 + i)
        Zn2_sim = compute_Zn_squared(X_sim)
        rejected_sim = Zn2_sim > threshold
        print(f"Elliptical Test {i+1}: Z^2 = {Zn2_sim:.2f} → {'Reject' if rejected_sim else 'Accept'}")

    print("\n--- Test sobre datos elipsoidales + ruido ---")
    for noise_std in [0.05, 0.1, 0.2]:
        for i in range(3):
            X_noisy = generate_elliptical_with_noise(mean, cov, noise_std=noise_std, n=200, seed=1200 + int(noise_std*100) + i)
            Zn2_noisy = compute_Zn_squared(X_noisy)
            rejected_noisy = Zn2_noisy > threshold
            print(f"Elliptical + noise {noise_std:.2f} [{i+1}]: Z^2 = {Zn2_noisy:.2f} → {'Reject' if rejected_noisy else 'Accept'}")

    return results

if __name__ == "__main__":
    with open("C:/Users/tejon/Documents/Statistical-Analysis-in-Random-Graphs/Two_Sample_Hypothesis_Testing/experiments/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_symmetry_test(config)
