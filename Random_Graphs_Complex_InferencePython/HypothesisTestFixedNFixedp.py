import numpy as np
from scipy.stats import multinomial, norm, gaussian_kde
from scipy.linalg import null_space
from numpy.linalg import norm
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain  # algoritmo de Louvain


# ------------------------
# Simulación del modelo esférico
# ------------------------

def calculate_inverse_projection(u, p):  # O(p)
    norm_p_u_squared = np.sum((p + u)**2)
    scaling_factor = 2 / norm_p_u_squared
    x = scaling_factor * (p + u) - u
    return x

def generate_orthogonal_sample(u, mu, Sigma, n):  # O(n·p²)
    basis = null_space(u.reshape(1, -1))  # O(p²)
    samples = np.random.multivariate_normal(mu, Sigma, size=n)  # O(n·p²)
    orthogonal_samples = samples @ basis.T  # O(n·p²)
    return orthogonal_samples

def generate_mixture_sphere_sample(mu, Sigma_list, u_list, weights, n):  # O(n·p²)
    k = len(weights)
    p = len(u_list[0])
    sample_counts = multinomial.rvs(n, weights)
    samples_all = np.zeros((n, p))
    row_idx = 0
    for i in range(k):
        ni = sample_counts[i]
        if ni > 0:
            sample_orth = generate_orthogonal_sample(u_list[i], mu, Sigma_list[i], ni)  # O(ni·p²)
            mapped = np.array([calculate_inverse_projection(u_list[i], x) for x in sample_orth])  # O(ni·p)
            samples_all[row_idx:row_idx+ni, :] = mapped
            row_idx += ni
    return samples_all

def generate_random_spherical_graph(sample_sphere):  # O(n²·p)
    n = sample_sphere.shape[0]
    dot_product_matrix = sample_sphere @ sample_sphere.T  # O(n²·p)
    arccos_matrix = np.arccos(np.clip(dot_product_matrix, -1, 1))  # O(n²)
    P = np.cos(0.5 * arccos_matrix)**14  # O(n²)
    np.fill_diagonal(P, 0)
    upper_tri = np.random.uniform(size=(n, n))  # O(n²)
    adj = (upper_tri < P).astype(int)  # O(n²)
    adj = np.triu(adj, 1)
    adj = adj + adj.T  # O(n²)
    G = nx.from_numpy_array(adj)  # O(n²)
    return G

def generate_test_graph(n, p, sigma, m, weights):  # O(n²·p)
    mu = np.zeros(p - 1)
    canonical_basis = [np.eye(p)[i] for i in range(p)]  # O(p²)

    # Construir u_list intercalando canónicos y sus negativos
    u_list = []
    i = 0
    while len(u_list) < m:
        u_list.append(canonical_basis[i % p])
        if len(u_list) < m:
            u_list.append(-canonical_basis[i % p])
        i += 1

    Sigma_list = [(1 / sigma) * np.eye(p - 1) for _ in range(m)]  # O(m·p²)
    sample_sphere = generate_mixture_sphere_sample(mu, Sigma_list, u_list, weights, n)  # O(n·p²)
    return generate_random_spherical_graph(sample_sphere)  # O(n²·p)


def count_communities_louvain(G):  # O(n log n)
    partition = community_louvain.best_partition(G)
    num_communities = len(set(partition.values()))
    return num_communities

# ------------------------
# Estadístico de triángulos normalizado
# ------------------------

def normalized_triangles(G):  # O(n³) en grafos densos
    t = sum(nx.triangles(G).values()) / 3  # O(n³) worst case
    k = G.number_of_nodes()
    max_tri = k * (k - 1) * (k - 2) / 6
    return t / max_tri if max_tri > 0 else 0

# ------------------------
# Test de hipótesis no paramétrico
# ------------------------

def hypothesis_test_triangles(G1, G2, b=200, k=None):  # O(b·k³)
    if k is None:
        k = int(round(G1.number_of_nodes()**(2/3)))
    
    stats_G1 = []
    for _ in range(b):
        sampled_nodes = np.random.choice(G1.nodes(), k, replace=False)  # O(k)
        subG = G1.subgraph(sampled_nodes)  # O(k²)
        stats_G1.append(normalized_triangles(subG))  # O(k³)

    ci_low = np.quantile(stats_G1, 0.025)  # O(b log b)
    ci_high = np.quantile(stats_G1, 0.975)

    sampled_nodes_G2 = np.random.choice(G2.nodes(), k, replace=False)  # O(k)
    subG2 = G2.subgraph(sampled_nodes_G2)  # O(k²)
    stat_G2 = normalized_triangles(subG2)  # O(k³)

    inside = ci_low <= stat_G2 <= ci_high

    print(f"Estadístico en G2: {stat_G2:.4f}")
    print(f"Intervalo 95% del bootstrap de G1: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Resultado del test: {'NO se rechaza H0' if inside else 'Se rechaza H0'}")

    plt.scatter(range(b), stats_G1, color="gray")
    plt.axhline(ci_low, color="blue", linestyle="dashed")
    plt.axhline(ci_high, color="blue", linestyle="dashed")
    plt.scatter(b + 10, stat_G2, color="green" if inside else "red")
    plt.text(b + 12, stat_G2, "G2", color="green" if inside else "red")
    plt.xlabel("Índice de muestra bootstrap")
    plt.ylabel("Conteo de triángulos normalizado")
    plt.title(f"Test de hipótesis con b={b}, k={k}")
    plt.show()

# ------------------------
# Comparar densidades de vecinos comunes
# ------------------------

def compute_common_neighbor_counts(G):  # O(n³) en grafos densos
    nodes = list(G.nodes())
    neighbors = {u: set(G.neighbors(u)) for u in nodes}  # O(n²)
    counts = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            ni = neighbors[nodes[i]]
            nj = neighbors[nodes[j]]
            common = len(ni.intersection(nj))  # O(n)
            counts.append(common)
    return np.array(counts)

# ------------------------
# Ejecutar prueba y visualización
# ------------------------

n = 1000
p = 26         # CONDICIÓN: p/2 >= m_1, m_2
sigma1 = 100000
m1 = 50
weights1 = [1/m1] * m1

sigma2 = 100000
m2 = 45
weights2 = [1/m2] * m2

np.random.seed(42)
G1 = generate_test_graph(n, p, sigma1, m1, weights1)
G2 = generate_test_graph(n, p, sigma2, m2, weights2)

hypothesis_test_triangles(G1, G2)

num_comms_G1 = count_communities_louvain(G1)
num_comms_G2 = count_communities_louvain(G2)

print(f"Número de comunidades en G1: {num_comms_G1}")
print(f"Número de comunidades en G2: {num_comms_G2}")

# Visualización
# fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# # Calcular layouts una sola vez (evita inconsistencias)
# pos_G1 = nx.spring_layout(G1, seed=42)
# pos_G2 = nx.spring_layout(G2, seed=42)

# # Dibujar G1
# nx.draw(
#     G1, pos=pos_G1, ax=axs[0],
#     node_size=5, node_color="black",
#     edge_color="gray", width=0.5, alpha=0.4,
#     with_labels=False
# )
# axs[0].set_title("Grafo G1")

# # Dibujar G2
# nx.draw(
#     G2, pos=pos_G2, ax=axs[1],
#     node_size=5, node_color="black",
#     edge_color="gray", width=0.5, alpha=0.4,
#     with_labels=False
# )
# axs[1].set_title("Grafo G2")

# plt.tight_layout()
# plt.show()


# ------------------------
# Comparar densidades de vecinos comunes
# ------------------------


print("Calculando vecinos comunes para G1...")
common_counts_G1 = compute_common_neighbor_counts(G1)

print("Calculando vecinos comunes para G2...")
common_counts_G2 = compute_common_neighbor_counts(G2)

# Calcular densidades suavizadas
density_G1 = gaussian_kde(common_counts_G1)
density_G2 = gaussian_kde(common_counts_G2)

x_range = np.linspace(0, max(common_counts_G1.max(), common_counts_G2.max()), 500)

main_title = f"Densidad de vecinos comunes entre pares de nodos\nG1: m={m1}, sigma={sigma1} | G2: m={m2}, sigma={sigma2}, n={n}, p={p}"

plt.figure(figsize=(10, 6))
plt.plot(x_range, density_G1(x_range), label="G1", color="black", linewidth=2)
plt.plot(x_range, density_G2(x_range), label="G2", color="red", linewidth=2)
plt.xlabel("Número de vecinos comunes")
plt.ylabel("Densidad")
plt.title(main_title)
plt.legend()
plt.grid(True)
plt.show()





