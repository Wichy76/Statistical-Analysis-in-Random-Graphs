import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os

from scipy.stats import multinomial
from scipy.linalg import null_space

# ---------------------------------------------------------------------
# Aquí asumo que ya tienes definidas:
#  - generate_test_graph
#  - normalized_triangles
#  - degree_assortativity
#  - average_shortest_path
#  - percentile_diameter
#  - hypothesis_test_triangles
#  - hypothesis_test_benchmark
#  - hypothesis_test
# ---------------------------------------------------------------------

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


def run_experiments(benchmark=False):
    """
    Ejecuta la batería de experimentos para varios n,
    usando el test univariado (benchmark=False) o el triple
    test de control (benchmark=True).
    """
    ns = [100, 200, 500]
    sigma = 1e12
    num_trials = 30

    for n in ns:
        p = int(2.5 * np.sqrt(n)) + 2
        m_options = [max(2, n // 20), max(3, n // 15), max(4, n // 10)]
        # Preparamos la figura de 6×5 subplots
        fig, axs = plt.subplots(6, 5, figsize=(15, 18))
        axs = axs.flatten()

        fp = 0  # falsos positivos
        fn = 0  # falsos negativos
        idx_subplot = 0

        for i in range(num_trials):
            # m1 y m2
            m1 = np.random.choice(m_options)
            if i < num_trials // 2:
                m2 = m1
            else:
                # aseguramos m2 != m1
                m2 = np.random.choice([m for m in m_options if m != m1])

            w1 = [1/m1] * m1
            w2 = [1/m2] * m2

            # Generamos los grafos de prueba
            G1 = generate_test_graph(n, p, sigma, m1, w1)
            G2 = generate_test_graph(n, p, sigma, m2, w2)

            # Ejecutamos el test, pasando el flag benchmark
            rejected = hypothesis_test(G1, G2, b=200, benchmark=benchmark)

            # Trazamos en el subplot correspondiente
            ax = axs[idx_subplot]
            title = f"m1={m1}, m2={m2}"
            ax.set_title(title, fontsize=7)
            # para el univariado de triángulos podemos reusar tu función de dibujo,
            # si lo deseas podrías crear una versión que pinte también los benchmarks.
            # Aquí simplemente coloreo el fondo:
            ax.set_facecolor('mistyrose' if rejected else 'honeydew')
            ax.set_xticks([])
            ax.set_yticks([])

            # Actualizamos contadores de error
            if m1 == m2 and rejected:
                fp += 1
            if m1 != m2 and not rejected:
                fn += 1

            idx_subplot += 1

        # Ajustes finales de la figura
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        titulo = f"{'Benchmark' if benchmark else 'Triángulos'} — n = {n} — 30 pruebas"
        fig.suptitle(titulo, fontsize=14)

        # Guardamos
        fname = f"test_{'benchmark' if benchmark else 'triangles'}_n_{n}.png"
        fig.savefig(os.path.join(output_dir, fname))
        plt.close(fig)

        # Resumen en consola
        print(f"\n=== Resultados para n = {n} — {'Benchmark' if benchmark else 'Triángulos'} ===")
        print(f"Falsos positivos (rechaza con m1=m2): {fp}")
        print(f"Falsos negativos (acepta con m1≠m2): {fn}")

if __name__ == "__main__":
    # Llamamos dos veces, para comparar test de triángulos vs benchmarks
    run_experiments(benchmark=False)
    run_experiments(benchmark=True)
