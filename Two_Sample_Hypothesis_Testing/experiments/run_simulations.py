import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import yaml
import sys
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.covariance import EllipticEnvelope
from core.statistics import compute_bivariate_stats
from core.hypothesis_test import run_bivariate_test

def load_graph_dataset(path):
    files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(".pkl")])
    dataset = []
    for f in files:
        with open(f, "rb") as g:
            obj = pickle.load(g)
            dataset.append((obj["graph"], obj["metadata"]))
    return dataset


def plot_trial(S, stat2, accepted, ax):
    ax.scatter(S[:, 0], S[:, 1], s=10, alpha=0.6, label="Bootstrap")
    color = 'green' if accepted else 'red'
    ax.scatter(stat2[0], stat2[1], color=color, edgecolor='black', s=40, label="G2")

    # Elipse de confianza
    clf = EllipticEnvelope(contamination=0.05).fit(S)
    center = clf.location_
    cov = clf.covariance_
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * np.sqrt(5.991 * vals)  # 95% ellipse
    ellipse = Ellipse(xy=center, width=width, height=height, angle=angle, edgecolor='blue', facecolor='none', linestyle='--')
    ax.add_patch(ellipse)
    ax.set_xticks([])
    ax.set_yticks([])


def simulate_and_test(dataset, stat_names, trial_id):
    if trial_id < len(dataset) // 2:
        G, _ = random.choice(dataset)
        G1 = G2 = G
    else:
        # Encuentra dos grafos con diferentes m
        attempts = 0
        while True:
            g1, meta1 = random.choice(dataset)
            g2, meta2 = random.choice(dataset)
            if meta1["m"] != meta2["m"]:
                G1, G2 = g1, g2
                break
            attempts += 1
            if attempts > 1000:
                raise RuntimeError("No se encontraron pares con diferente m después de 1000 intentos")

    n = G1.number_of_nodes()
    #K = int(round(n ** (2 / 3)))
    K = 200
    
    S = np.array([compute_bivariate_stats(G1, K, stat_names, n+it) for it in range(200)])
    stat2 = compute_bivariate_stats(G2, K, stat_names)
    _, _, accepted2d = run_bivariate_test(S, stat2)
    return accepted2d, S, stat2


def run_batch(config):
    dataset_path = config["dataset_path"]
    stat_names = config["stat_names"]
    create_images = config.get("create_images", False)
    dataset_name = os.path.basename(dataset_path.rstrip("/\\"))
    output_dir = os.path.join("visualization", f"{dataset_name}_results")
    os.makedirs(output_dir, exist_ok=True)

    dataset = load_graph_dataset(dataset_path)
    random.seed(42)
    np.random.seed(42)
    random.shuffle(dataset)

    n_trials = 30
    results = []
    all_data = []

    for i in range(n_trials):
        accepted, S, stat2 = simulate_and_test(dataset, stat_names, i)
        results.append(accepted)
        all_data.append((S, stat2, accepted))

    if create_images:
        fig, axes = plt.subplots(6, 5, figsize=(30, 18))
        for i, ax in enumerate(axes.flat):
            S, stat2, accepted = all_data[i]
            plot_trial(S, stat2, accepted, ax)
        fig.suptitle(f"Dataset: {dataset_name}", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(output_dir, f"{dataset_name}.png")
        fig.savefig(save_path)
        plt.close(fig)

    return results


def summarize(results):
    n = len(results)
    labels = np.array([0] * (n // 2) + [1] * (n // 2))  # 0: H0, 1: H1
    biv_rej = np.logical_not(results)

    type_I_biv = np.mean(biv_rej[:n // 2])
    type_II_biv = 1 - np.mean(biv_rej[n // 2:])

    print("\nResults Summary:")
    print(f"Bivariate   → Type I: {type_I_biv:.2f}, Type II: {type_II_biv:.2f}")


if __name__ == '__main__':
    with open("C:/Users/tejon/Documents/Statistical-Analysis-in-Random-Graphs/Two_Sample_Hypothesis_Testing/experiments/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    results = run_batch(config)
    summarize(results)