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
    return scaling_factor * (p + u) - u

def generate_orthogonal_sample(u, mu, Sigma, n):
    basis = null_space(u.reshape(1, -1))
    samples = np.random.multivariate_normal(mu, Sigma, size=n)
    return samples @ basis.T

def generate_mixture_sphere_sample(mu, Sigma_list, u_list, weights, n):
    k = len(weights)
    p = len(u_list[0])
    sample_counts = multinomial.rvs(n, weights)
    samples_all = np.zeros((n, p))
    idx = 0
    for i, wi in enumerate(weights):
        ni = sample_counts[i]
        if ni > 0:
            orth = generate_orthogonal_sample(u_list[i], mu, Sigma_list[i], ni)
            mapped = np.array([calculate_inverse_projection(u_list[i], x) for x in orth])
            samples_all[idx:idx+ni] = mapped
            idx += ni
    return samples_all

def generate_random_spherical_graph(sample_sphere):
    n = sample_sphere.shape[0]
    D = sample_sphere @ sample_sphere.T
    A = np.arccos(np.clip(D, -1, 1))
    P = np.cos(0.5*A)**14
    np.fill_diagonal(P, 0)
    U = np.random.rand(n,n)
    adj = (U < P).astype(int)
    adj = np.triu(adj,1)
    adj = adj + adj.T
    return nx.from_numpy_array(adj)

def generate_test_graph(n, p, sigma, m, weights):
    mu = np.zeros(p-1)
    basis_vecs = [np.eye(p)[i] for i in range(p)]
    u_list = []
    idx0 = 0
    while len(u_list)<m:
        u_list.append(basis_vecs[idx0%p])
        if len(u_list)<m:
            u_list.append(-basis_vecs[idx0%p])
        idx0+=1
    Sigma_list = [(1/sigma)*np.eye(p-1) for _ in range(m)]
    sphere = generate_mixture_sphere_sample(mu, Sigma_list, u_list, weights, n)
    return generate_random_spherical_graph(sphere)

def normalized_triangles(G):
    t = sum(nx.triangles(G).values())/3
    k = G.number_of_nodes()
    max_tri = k*(k-1)*(k-2)/6
    return t/max_tri if max_tri>0 else 0

def hypothesis_test_triangles(G1, G2, b=200, k=None, ax=None):
    n = G1.number_of_nodes()
    if k is None:
        k = int(round(n**(2/3)))
    S = [normalized_triangles(G1.subgraph(
             np.random.choice(G1.nodes(), k, replace=False)))
         for _ in range(b)]
    lo, hi = np.quantile(S, 0.025), np.quantile(S, 0.975)
    stat2 = normalized_triangles(G2.subgraph(
              np.random.choice(G2.nodes(), k, replace=False)))
    rej = not (lo <= stat2 <= hi)
    if ax is not None:
        ax.scatter(range(b), S, c='gray', s=5)
        ax.axhline(lo, c='blue', ls='--')
        ax.axhline(hi, c='blue', ls='--')
        ax.scatter(b+5, stat2, c='green' if not rej else 'red', s=20)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("H₀ " + ("no rechazada" if not rej else "rechazada"),
                     fontsize=6)
    return rej

def run_experiments():
    ns = [100, 200, 500]
    sigma = 1e12
    trials = 30

    for n in ns:
        p = int(2.5*np.sqrt(n)) + 2
        m_opts = [max(2,n//20), max(3,n//15), max(4,n//10)]
        fig, axs = plt.subplots(6,5,figsize=(15,18))
        axs = axs.flatten()
        fp = fn = 0
        idx = 0

        for i in range(trials):
            m1 = np.random.choice(m_opts)
            m2 = m1 if i<trials//2 else np.random.choice([m for m in m_opts if m!=m1])
            w1 = [1/m1]*m1
            w2 = [1/m2]*m2

            G1 = generate_test_graph(n, p, sigma, m1, w1)
            G2 = generate_test_graph(n, p, sigma, m2, w2)

            rejected = hypothesis_test_triangles(G1, G2, ax=axs[idx])
            axs[idx].set_title(f"m1={m1}, m2={m2}", fontsize=7)
            idx += 1

            if m1==m2 and rejected:    fp += 1
            if m1!=m2 and not rejected: fn += 1

        plt.tight_layout(); plt.subplots_adjust(top=0.92)
        plt.suptitle(f"[Triangles] n={n}, {trials} pruebas", fontsize=14)
        plt.savefig(os.path.join(output_dir, f"test_triangles_{n}.png"))
        plt.close()

        print(f"[Triangles] n = {n} → falsos positivos: {fp}, falsos negativos: {fn}\n")

if __name__ == "__main__":
    run_experiments()
