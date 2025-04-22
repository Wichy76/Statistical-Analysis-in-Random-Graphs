import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import multinomial
from scipy.linalg import null_space
import os

# Ruta donde se guardarán las imágenes
output_dir = r"C:\Users\tejon\Documents\Statistical-Analysis-in-Random-Graphs\Random_Graphs_Complex_Inference_in_R\tests"
os.makedirs(output_dir, exist_ok=True)

def calculate_inverse_projection(u, p):
    norm_p_u_squared = np.sum((p + u)**2)
    scaling_factor = 2 / norm_p_u_squared
    x = scaling_factor * (p + u) - u
    return x

def generate_orthogonal_sample(u, mu, Sigma, n):
    basis = null_space(u.reshape(1, -1))
    samples = np.random.multivariate_normal(mu, Sigma, size=n)
    return samples @ basis.T

def generate_mixture_sphere_sample(mu, Sigma_list, u_list, weights, n):
    k = len(weights)
    p = len(u_list[0])
    sample_counts = multinomial.rvs(n, weights)
    samples_all = np.zeros((n, p))
    row_idx = 0
    for i in range(k):
        ni = sample_counts[i]
        if ni > 0:
            sample_orth = generate_orthogonal_sample(u_list[i], mu, Sigma_list[i], ni)
            mapped = np.array([calculate_inverse_projection(u_list[i], x) for x in sample_orth])
            samples_all[row_idx:row_idx+ni, :] = mapped
            row_idx += ni
    return samples_all

def generate_random_spherical_graph(sample_sphere):
    n = sample_sphere.shape[0]
    dot_product_matrix = sample_sphere @ sample_sphere.T
    arccos_matrix = np.arccos(np.clip(dot_product_matrix, -1, 1))
    P = np.cos(0.5 * arccos_matrix)**14
    np.fill_diagonal(P, 0)
    upper_tri = np.random.uniform(size=(n, n))
    adj = (upper_tri < P).astype(int)
    adj = np.triu(adj, 1)
    adj = adj + adj.T
    return nx.from_numpy_array(adj)

def generate_test_graph(n, p, sigma, m, weights):
    mu = np.zeros(p - 1)
    canonical_basis = [np.eye(p)[i] for i in range(p)]
    u_list = []
    i = 0
    while len(u_list) < m:
        u_list.append(canonical_basis[i % p])
        if len(u_list) < m:
            u_list.append(-canonical_basis[i % p])
        i += 1
    Sigma_list = [(1 / sigma) * np.eye(p - 1) for _ in range(m)]
    sample_sphere = generate_mixture_sphere_sample(mu, Sigma_list, u_list, weights, n)
    return generate_random_spherical_graph(sample_sphere)

def normalized_triangles(G):
    t = sum(nx.triangles(G).values()) / 3
    k = G.number_of_nodes()
    max_tri = k * (k - 1) * (k - 2) / 6
    return t / max_tri if max_tri > 0 else 0

def hypothesis_test_triangles(G1, G2, b=200, k=None, ax=None):
    if k is None:
        k = int(round(G1.number_of_nodes()**(2/3)))
    stats_G1 = []
    for _ in range(b):
        nodes = np.random.choice(G1.nodes(), k, replace=False)
        subG = G1.subgraph(nodes)
        stats_G1.append(normalized_triangles(subG))
    ci_low = np.quantile(stats_G1, 0.025)
    ci_high = np.quantile(stats_G1, 0.975)
    nodes_G2 = np.random.choice(G2.nodes(), k, replace=False)
    subG2 = G2.subgraph(nodes_G2)
    stat_G2 = normalized_triangles(subG2)
    inside = ci_low <= stat_G2 <= ci_high

    if ax:
        ax.scatter(range(b), stats_G1, color="gray", s=5)
        ax.axhline(ci_low, color="blue", linestyle="--")
        ax.axhline(ci_high, color="blue", linestyle="--")
        ax.scatter(b + 10, stat_G2, color="green" if inside else "red", s=20)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("H0 " + ("NO rechazada" if inside else "rechazada"), fontsize=6)
    return not inside

def run_experiments():
    ns = [100, 200, 500]
    sigma = 1e12
    num_trials = 30

    for n in ns:
        p = int(2.5 * np.sqrt(n)) + 2
        m_options = [max(2, n // 20), max(3, n // 15), max(4, n // 10)]
        fig, axs = plt.subplots(6, 5, figsize=(15, 18))
        axs = axs.flatten()
        fp, fn = 0, 0
        subplot_idx = 0

        for i in range(num_trials):
            m1 = np.random.choice(m_options)
            m2 = m1 if i < num_trials // 2 else np.random.choice([m for m in m_options if m != m1])
            weights1 = [1/m1] * m1
            weights2 = [1/m2] * m2

            G1 = generate_test_graph(n, p, sigma, m1, weights1)
            G2 = generate_test_graph(n, p, sigma, m2, weights2)

            rejected = hypothesis_test_triangles(G1, G2, ax=axs[subplot_idx])
            axs[subplot_idx].set_title(f"m1={m1}, m2={m2}", fontsize=7)
            subplot_idx += 1

            if m1 == m2 and rejected:
                fp += 1
            if m1 != m2 and not rejected:
                print(f"Aceptó H0 con parámetros m1 = {m1} y m2 = {m2}")
                fn += 1

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.suptitle(f"n = {n} — 30 pruebas", fontsize=14)
        plt.savefig(os.path.join(output_dir, f"test_n_{n}.png"))
        plt.close()

        print(f"\nResumen para n = {n}:")
        print(f"Falsos positivos (rechazo incorrecto con m1 = m2): {fp}")
        print(f"Falsos negativos (aceptó H0 cuando m1 ≠ m2): {fn}")

if __name__ == "__main__":
    run_experiments()
