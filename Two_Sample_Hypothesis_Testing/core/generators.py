import numpy as np
import networkx as nx
from scipy.stats import multinomial
from scipy.linalg import null_space


def calculate_inverse_projection(u, p):
    norm_p_u_squared = np.sum((p + u) ** 2)
    scaling_factor = 2 / norm_p_u_squared
    return scaling_factor * (p + u) - u


def generate_orthogonal_sample(u, mu, Sigma, n):
    basis = null_space(u.reshape(1, -1))
    samples = np.random.multivariate_normal(mu, Sigma, size=n)
    return samples @ basis.T


def generate_mixture_sphere_sample(mu, Sigma_list, u_list, weights, n):
    k = len(weights)
    p = len(u_list[0])
    counts = multinomial.rvs(n, weights)
    out = np.zeros((n, p))
    idx = 0
    for i in range(k):
        ni = counts[i]
        if ni > 0:
            ort = generate_orthogonal_sample(u_list[i], mu, Sigma_list[i], ni)
            mapped = np.array([calculate_inverse_projection(u_list[i], x) for x in ort])
            out[idx:idx+ni] = mapped
            idx += ni
    return out


def generate_random_spherical_graph(X):
    D = X @ X.T
    A = np.cos(0.5 * np.arccos(np.clip(D, -1, 1))) ** 14
    np.fill_diagonal(A, 0)
    U = np.random.rand(*A.shape)
    adj = ((U < A) | (U.T < A)).astype(int)
    return nx.from_numpy_array(adj)


