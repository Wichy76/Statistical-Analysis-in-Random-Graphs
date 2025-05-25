import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import SimulateG1G2              # tu módulo con simulate_or_load()
from sklearn.covariance import EllipticEnvelope
from scipy.stats import chi2
import community as community_louvain
import os

# Ruta donde se guardarán las imágenes
output_dir = r"C:\Users\tejon\Documents\Statistical-Analysis-in-Random-Graphs\Tests\CN_Linf+AvgDist+Triangles"
os.makedirs(output_dir, exist_ok=True)

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

def compute_stats_subgraph(G, k):
    nodes = np.random.choice(list(G), k, replace=False)
    sub = G.subgraph(nodes)
    # f1 = common_neighbors (sobre no todos los nodos) 
    
    
    # f2 = distancia media (sobre componente gigante)
    partition = community_louvain.best_partition(sub)
    f2 = community_louvain.modularity(partition, sub)
    # f3 = triángulos normalizados
    t = sum(nx.triangles(sub).values())/3
    k0 = sub.number_of_nodes()
    max_t = k0*(k0-1)*(k0-2)/6
    f3 = t/max_t if max_t>0 else 0.0
    return np.array([f1, f2, f3])

def fit_mve_ee(S):
    ee = EllipticEnvelope(contamination=0.05).fit(S)
    return ee

def test_point(ee, stat2):
    m2 = ee.mahalanobis(stat2.reshape(1,-1))[0]
    thr = chi2.ppf(0.95, df=stat2.size)
    return m2, thr, (m2 <= thr)

def run_experiments_mv():
    ns      = [100, 200, 500]
    sigma   = 1e5
    trials  = 30
    b       = 200

    for n in ns:
        p = n//20 + 1
        m_opts = [max(2,n//20), max(3,n//15), max(4,n//10)]

        fig, axes = plt.subplots(6,5, figsize=(18,11))
        axes = axes.flatten()

        fp = fn = 0
        for i in range(trials):
            np.random.seed(42+i)
            # elegir m1, m2 y pesos
            m1 = np.random.choice(m_opts)
            m2 = m1 if i < trials//2 else np.random.choice([m for m in m_opts if m!=m1])
            #Pesos decrecientes
            w1 = [1/m1]*m1
            w2 = [1/m2]*m2

            # simular G1, G2
            cache = 'simulated_graphs.pkl'
            params = (n, p, sigma, m1, w1, sigma, m2, w2, 42)
            G1, G2 = SimulateG1G2.simulate_or_load(cache, params, force=True)

            # calcular subgrafitos bootstrap en G1
            k = int(round(n**(2/3)))
            S = np.vstack([ compute_stats_subgraph(G1, k) for _ in range(b) ])

            # ajustar EllipticEnvelope
            ee = fit_mve_ee(S)

            # calcular distancias Mahalanobis² de los b puntos
            ms_boot = ee.mahalanobis(S)
            # estadística y test para G2
            stat2 = compute_stats_subgraph(G2, k)
            m2_val, thr, accept = test_point(ee, stat2)

            ax = axes[i]
            # plot de distancias bootstrap
            ax.scatter(np.arange(b), ms_boot, c='gray', s=5)
            # umbral
            ax.axhline(thr, color='blue', linestyle='--')
            # punto de la prueba
            ax.scatter(b+5, m2_val, c='green' if accept else 'red', s=20)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"m1={m1}, m2={m2}", fontsize=7)

            # contabilizar FP/FN
            if m1==m2 and not accept: fn += 1
            if m1!=m2 and     accept: fp += 1

        # remate de la figura
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.suptitle(f"[MVE] n={n}, {trials} pruebas  FP={fp}, FN={fn}", fontsize=14)
        plt.savefig(os.path.join(output_dir, f"test_mve_mv_fixed_psigmaulistweights_{n}.png"))
        plt.close(fig)

        print(f"[MVE] n = {n} → falsos positivos: {fp}, falsos negativos: {fn}\n")

if __name__ == "__main__":
    run_experiments_mv()
