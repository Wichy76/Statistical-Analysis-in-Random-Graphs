import os
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ------------------------------------------------
# 1) Simulación y cacheo de los grafos
# ------------------------------------------------

def calculate_inverse_projection(u, p):
    norm_p_u_squared = np.sum((p + u)**2)
    scaling_factor   = 2 / norm_p_u_squared
    return scaling_factor*(p + u) - u

def generate_orthogonal_sample(u, mu, Sigma, n):
    from scipy.linalg import null_space
    basis   = null_space(u.reshape(1,-1))
    samples = np.random.multivariate_normal(mu, Sigma, size=n)
    return samples @ basis.T

def generate_mixture_sphere_sample(mu, Sigma_list, u_list, weights, n):
    from scipy.stats import multinomial
    k = len(weights)
    p = len(u_list[0])
    counts = multinomial.rvs(n, weights)
    out    = np.zeros((n,p))
    idx    = 0
    for i in range(k):
        ni = counts[i]
        if ni>0:
            ort = generate_orthogonal_sample(u_list[i], mu, Sigma_list[i], ni)
            mapped = np.array([calculate_inverse_projection(u_list[i], x)
                               for x in ort])
            out[idx:idx+ni] = mapped
            idx += ni
    return out

def generate_random_spherical_graph(X):
    # este es el mismo kernel de adyacencia que antes
    D = X@X.T
    A = np.cos(0.5*np.arccos(np.clip(D,-1,1)))**14
    np.fill_diagonal(A,0)
    U = np.random.rand(*A.shape)
    adj = ((U<A)|(U.T<A)).astype(int)
    return nx.from_numpy_array(adj)

def generate_test_graph(n, p, sigma, m, weights, seed=None):
    """
    Genera un grafo de tamaño n, dimensión p, parámetro sigma,
    mezcla de m componentes con pesos=weights.
    """
    if seed is not None:
        np.random.seed(seed)
    # construye u_list
    basis = [np.eye(p)[i] for i in range(p)]
    u_list=[]
    i=0
    while len(u_list)<m:
        u_list.append( basis[i%p] )
        if len(u_list)<m:
            u_list.append(-basis[i%p])
        i+=1
    mu     = np.zeros(p-1)
    Sigmas = [(1/sigma)*np.eye(p-1) for _ in range(m)]
    S      = generate_mixture_sphere_sample(mu,Sigmas,u_list,weights,n)
    return generate_random_spherical_graph(S)

def simulate_or_load(path, params, force=True):
    """
    Si existe el archivo path y force=False, lo carga con pickle.
    Sino, genera los grafos con los parámetros params (tupla)
    y los guarda en path.
    params = (n,p,sigma1,m1,weights1,sigma2,m2,weights2,seed)
    Devuelve G1, G2.
    """
    if (not force) and os.path.exists(path):
        with open(path,'rb') as f:
            G1, G2 = pickle.load(f)
        print("Grafos cargados de cache.")
    else:
        n,p,sigma1,m1,weights1,sigma2,m2,weights2,seed = params
        G1 = generate_test_graph(n,p,sigma1,m1,weights1, seed=seed)
        G2 = generate_test_graph(n,p,sigma2,m2,weights2, seed=seed+1)
        with open(path,'wb') as f:
            pickle.dump((G1,G2), f)
    return G1, G2

n=500
p = int(n / 2 + 1)  # p es la dimensión del espacio
i = 8
possible_ms = [1, 2, 3, 4, 6, 8, 10, 12, 15, 18]
m1 = possible_ms[i]
m2 = possible_ms[i+1]
if __name__=="__main__":
    cache_file = 'simulated_graphs.pkl'
    params = (
        n,    # n
        p,      # p
        10*p,     # sigma1
        m1,      # m1
        [1/m1]*m1,
        10*p,     # sigma2
        m2,      # m2
        [1/m2]*m2,
        42       # seed
    )
    G1, G2 = simulate_or_load(cache_file, params, force=True)

    #Visualización
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Calcular layouts una sola vez (evita inconsistencias)
    pos_G1 = nx.spring_layout(G1, seed=42)
    pos_G2 = nx.spring_layout(G2, seed=42)

    # Dibujar G1
    nx.draw(
        G1, pos=pos_G1, ax=axs[0],
        node_size=5, node_color="black",
        edge_color="gray", width=0.5, alpha=0.4,
        with_labels=False
        )
    axs[0].set_title("Grafo G1")

    # Dibujar G2
    nx.draw(
        G2, pos=pos_G2, ax=axs[1],
        node_size=5, node_color="black",
        edge_color="gray", width=0.5, alpha=0.4,
        with_labels=False
    )
    axs[1].set_title("Grafo G2")

    plt.tight_layout()
    plt.show()

    # Número de comunidades
    import community as community_louvain
    print(f"n: {n}, p: {p}, m1: {m1}, m2: {m2}")
    print(f"Comunidades en G1: {m1}, Comunidades en G2: {m2}\n")
    partition1 = community_louvain.best_partition(G1)
    partition2 = community_louvain.best_partition(G2)
    num_communities_G1 = len(set(partition1.values()))
    num_communities_G2 = len(set(partition2.values()))
    print(f"Comunidades detectadas en G1: {num_communities_G1}, G2: {num_communities_G2}")