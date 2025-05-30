import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import Random_Graphs_Complex_InferencePython.RSG_simulation.SimulateG1G2 as SimulateG1G2  # tu módulo con simulate_or_load()

np.random.seed(42+1)
def compute_common_neighbor_counts(G):
    """
    Devuelve un array con el número de vecinos comunes para cada par (i<j) de nodos en G.
    """
    # Obtenemos la matriz de adyacencia (no es la forma más eficiente para grafos muy grandes,
    # pero replica exactamente tu lógica de adj %*% adj y extraer upper triangle)
    A = nx.to_numpy_array(G, dtype=int)
    # common_mat[i,j] = número de vecinos comunes entre i y j
    common_mat = A @ A
    # Extraemos sólo la parte superior (i<j)
    iu = np.triu_indices_from(common_mat, k=1)
    return common_mat[iu]

def sample_common_counts(G, S):
    """
    Muestrea S pares (i<j) de nodos *sin* reemplazo y calcula el número de vecinos comunes.
    Devuelve un array de longitud S.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    counts = np.empty(S, dtype=int)
    seen = set()
    k = 0
    while k < S:
        i, j = np.random.choice(n, 2, replace=False)
        if i > j:
            i, j = j, i
        if (i,j) in seen:
            continue
        seen.add((i,j))
        # vecinos comunes:
        ni = set(G.neighbors(nodes[i]))
        nj = set(G.neighbors(nodes[j]))
        counts[k] = len(ni & nj)
        k += 1
    return counts


if __name__ == "__main__":
    # ---------------------------
    # 1) Generar o cargar G1, G2
    # ---------------------------
    cache = "simulated_graphs.pkl"
    # Ajusta estos parámetros a lo que uses
    n = 500
    p = n//20 + 1
    sigma   = 1e5
    m1= 50  
    m2 = 10
    params = (
        n,    
        p,     
        1e5,     # sigma1
        m1,      # m1
        [1/m1]*m1,
        1e5,     # sigma2
        m2,      # m2
        [1/m2]*m2,
        42       # seed
    )
    G1, G2 = SimulateG1G2.simulate_or_load(cache, params, force=True)


    # --------------------------------------------------
    # 2) Calcular vectores de vecinos comunes en G1 y G2
    # --------------------------------------------------

    S = min(2000, n*(n-1)//2)

    print("Calculando vecinos comunes para G1...")
    common_G1 = compute_common_neighbor_counts(G1)
    common_G1_sampled = sample_common_counts(G1, S)  # muestreo de 1000 pares
    print("Calculando vecinos comunes para G2...")
    common_G2 = compute_common_neighbor_counts(G2)
    common_G2_sampled = sample_common_counts(G2, S)  # muestreo de 1000 pares

    # --------------------------------------------------
    # 3) Estimar densidades (KDE) via gaussian_kde
    # --------------------------------------------------
    # datos ≥ 0, así que fijamos grid desde 0 hasta max posible
    xmin, xmax = 0, max(common_G1.max(), common_G2.max())
    #xmin, xmax = 0, max(common_G1.max(), common_G1_sampled.max())
    xs = np.linspace(xmin, xmax, 1000)

    kde1 = gaussian_kde(common_G1)
    kde1_sampled = gaussian_kde(common_G1_sampled)
    kde2 = gaussian_kde(common_G2)
    kde2_sampled = gaussian_kde(common_G2_sampled)

    dens1 = kde1(xs)
    dens1_sampled = kde1_sampled(xs)
    dens2 = kde2(xs)
    dens2_sampled = kde2_sampled(xs)    

    # --------------------------------------------------
    # 4) Ploteo superpuesto estilo ggplot2/R
    # --------------------------------------------------
    title = (
        f"Densidad de vecinos comunes entre pares de nodos\n"
        f"G1: m={m1} "
        f"G2: m={m2} "
        f"n={n}, p={p}"
    )

    plt.figure(figsize=(8,5))
    plt.plot(xs, dens1, color="black", linewidth=2, label="G1")
    plt.plot(xs, dens2, color="red",   linewidth=2, label="G2")
    plt.fill_between(xs, dens1_sampled, color="black", alpha=0.2, label="G1 (muestreo)")
    plt.fill_between(xs, dens2_sampled, color="red", alpha=0.2, label="G2 (muestreo)")
    plt.title(title)
    plt.xlabel("Número de vecinos comunes")
    plt.ylabel("Densidad")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

     # --------------------------------------------------
    # 4) Cálculo aproximado de distancias L1 y L∞
    # --------------------------------------------------
    dx = xs[1] - xs[0]
    L1_approx  = np.sum(np.abs(dens1 - dens2)) * dx
    Linf_approx = np.max(np.abs(dens1 - dens2))
    print(f"L1 ≈ {L1_approx:.4f}, L∞ ≈ {Linf_approx:.4f}")

    # --------------------------------------------------
    # 5) Plot superpuesto estilo ggplot2/R
    # --------------------------------------------------
    title = (
        f"Densidades KDE sobre {S} pares muestreados\n"
        f"G1: m={m1}, G2: m={m2}, n={n}, p={p}\n"
        f"L1≈{L1_approx:.3f}, L∞≈{Linf_approx:.3f}"
    )

    plt.figure(figsize=(8,5))
    plt.plot(xs, dens1, color="black", linewidth=2, label="G1")
    plt.plot(xs, dens2, color="red",   linewidth=2, label="G2")
    plt.fill_between(xs, dens1_sampled, color="black", alpha=0.2, label="G1 (muestreo)")
    plt.fill_between(xs, dens2_sampled, color="red", alpha=0.2, label="G2 (muestreo)")
    plt.title(title)
    plt.xlabel("Número de vecinos comunes")
    plt.ylabel("Densidad KDE")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()