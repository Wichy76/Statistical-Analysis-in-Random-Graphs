import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import multinomial
from scipy.linalg import null_space
import os
from mpl_toolkits.mplot3d import Axes3D  # necesario para 3D

# Ruta donde se guardarán las imágenes
output_dir = r"C:\Users\tejon\Documents\Statistical-Analysis-in-Random-Graphs\Random_Graphs_Complex_Inference_in_R\tests"
os.makedirs(output_dir, exist_ok=True)

# ——————————————————————————————————————————————————————————
# Ajuste del MVE (elipsoide minimax) al 95% — algoritmo de Khachiyan
# ——————————————————————————————————————————————————————————
def mve_khachiyan(P, tol=1e-5, max_iter=500):
    N, d = P.shape
    Q = np.hstack((P, np.ones((N,1))))  # (N, d+1)
    u = np.ones(N) / N
    for _ in range(max_iter):
        X = Q.T @ (Q * u[:,None])
        M = np.einsum('ij,jk,ki->i', Q, np.linalg.inv(X), Q.T)
        j = np.argmax(M)
        maxM = M[j]
        step = (maxM - d - 1) / ((d+1)*(maxM - 1))
        new_u = (1 - step)*u
        new_u[j] += step
        if np.linalg.norm(new_u - u) < tol:
            u = new_u
            break
        u = new_u
    c = P.T @ u
    cov = ((P - c).T * u) @ (P - c) / d
    A = np.linalg.inv(cov)
    return c, A

# ——————————————————————————————————————————————————————————
# Generación de grafos (igual que antes)
# ——————————————————————————————————————————————————————————
def calculate_inverse_projection(u, p):
    norm_p_u_squared = np.sum((p + u)**2)
    return 2/(norm_p_u_squared)*(p+u) - u

def generate_orthogonal_sample(u, mu, Sigma, n):
    basis = null_space(u.reshape(1,-1))
    X = np.random.multivariate_normal(mu, Sigma, size=n)
    return X @ basis.T

def generate_mixture_sphere_sample(mu, Sigma_list, u_list, weights, n):
    k = len(weights)
    p = len(u_list[0])
    counts = multinomial.rvs(n, weights)
    out = np.zeros((n,p))
    idx = 0
    for i,wi in enumerate(weights):
        ni = counts[i]
        if ni>0:
            orth = generate_orthogonal_sample(u_list[i], mu, Sigma_list[i], ni)
            mapped = np.array([calculate_inverse_projection(u_list[i], x)
                               for x in orth])
            out[idx:idx+ni] = mapped
            idx += ni
    return out

def generate_random_spherical_graph(sample_sphere):
    n = sample_sphere.shape[0]
    D = sample_sphere @ sample_sphere.T
    A = np.arccos(np.clip(D,-1,1))
    P = np.cos(0.5*A)**14
    np.fill_diagonal(P,0)
    U = np.random.rand(n,n)
    adj = (U < P).astype(int)
    adj = np.triu(adj,1)
    adj = adj + adj.T
    return nx.from_numpy_array(adj)

def generate_test_graph(n, p, sigma, m, weights):
    mu = np.zeros(p-1)
    basis = [np.eye(p)[i] for i in range(p)]
    u_list = []
    idx0 = 0
    while len(u_list)<m:
        u_list.append(basis[idx0%p])
        if len(u_list)<m:
            u_list.append(-basis[idx0%p])
        idx0+=1
    Sigma_list = [(1/sigma)*np.eye(p-1) for _ in range(m)]
    sphere = generate_mixture_sphere_sample(mu, Sigma_list, u_list, weights, n)
    return generate_random_spherical_graph(sphere)

def compute_stats_subgraph(G, k):
    nodes = np.random.choice(list(G.nodes()), k, replace=False)
    sub = G.subgraph(nodes)
    # (i) asortatividad de grado
    f1 = nx.degree_assortativity_coefficient(sub) if sub.number_of_edges()>0 else 0.0
    # (ii) distancia media
    if nx.is_connected(sub):
        lengths = dict(nx.all_pairs_shortest_path_length(sub))
        dists = [d for u in lengths for v,d in lengths[u].items() if u<v]
    else:
        comp = max(nx.connected_components(sub), key=len)
        sg = sub.subgraph(comp)
        lengths = dict(nx.all_pairs_shortest_path_length(sg))
        dists = [d for u in lengths for v,d in lengths[u].items() if u<v]
    f2 = np.mean(dists) if dists else 0.0
    # (iii) diámetro al 95 pct
    f3 = np.percentile(dists,95) if dists else 0.0
    return np.array([f1,f2,f3])

# ——————————————————————————————————————————————————————————
# Test multivariado con MVE al 95% y scatter 3D de la "nube" + punto G2
# ——————————————————————————————————————————————————————————
def multivariate_hypothesis_test(G1, G2, b=200, k=None, ax=None):
    n = G1.number_of_nodes()
    if k is None:
        k = int(round(n**(2/3)))

    # (1) bootstrap de G1
    S = np.array([compute_stats_subgraph(G1,k) for _ in range(b)])  # (b,3)

    # (2) recortamos el 5% más lejano (Euclídeo)
    cent0 = S.mean(0)
    euc = np.linalg.norm(S-cent0,axis=1)
    cut = np.quantile(euc,0.95)
    S95 = S[euc<=cut]

    # (3) ajustamos MVE
    center, A = mve_khachiyan(S95)

    # (4) estadístico de G2
    stat2 = compute_stats_subgraph(G2,k)
    diff = stat2 - center
    val = diff @ A @ diff
    rej = val>1.0

    # (5) si hay ejes 3D, lo pintamos
    if ax is not None:
        ax.scatter(S95[:,0], S95[:,1], S95[:,2],
                   c='gray', s=15, alpha=0.6, label='bootstrap')
        # punto G2
        ax.scatter([stat2[0]], [stat2[1]], [stat2[2]],
                   c='red' if rej else 'green', s=100, label='G2')
        ax.set_xlabel('f1=asort.')
        ax.set_ylabel('f2=dist media')
        ax.set_zlabel('f3=diam95')
        ax.legend(fontsize=6)
        ax.set_title("H₀ "+("rechazada" if rej else "aceptada"), fontsize=8)

    return rej

# ——————————————————————————————————————————————————————————
# Experimentos principales
# ——————————————————————————————————————————————————————————
def run_experiments_mv():
    ns = [100,200,500]
    sigma = 1e12
    trials = 30

    for n in ns:
        p = int(2.5*np.sqrt(n))+2
        m_opts = [max(2,n//20), max(3,n//15), max(4,n//10)]
        fig = plt.figure(figsize=(15,18))
        axs = [fig.add_subplot(6,5,i+1,projection='3d') for i in range(30)]
        fp=fn=0; idx=0

        for i in range(trials):
            m1 = np.random.choice(m_opts)
            m2 = m1 if i<trials//2 else np.random.choice([m for m in m_opts if m!=m1])
            w1=[1/m1]*m1; w2=[1/m2]*m2

            G1=generate_test_graph(n,p,sigma,m1,w1)
            G2=generate_test_graph(n,p,sigma,m2,w2)

            rejected = multivariate_hypothesis_test(G1,G2,b=200,k=None,ax=axs[idx])
            axs[idx].set_title(f"m1={m1}, m2={m2}",fontsize=7)
            idx+=1

            if m1==m2 and rejected:     fp+=1
            if m1!=m2 and not rejected: fn+=1

        plt.tight_layout(); fig.suptitle(f"[MV‑MVE] n={n}, {trials} pruebas", y=1.02, fontsize=14)
        plt.savefig(os.path.join(output_dir, f"test_mv_{n}.png"), bbox_inches='tight')
        plt.close(fig)

        print(f"[MV‑MVE] n = {n} → falsos positivos: {fp}, falsos negativos: {fn}\n")

if __name__=="__main__":
    run_experiments_mv()
