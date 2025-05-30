#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_mv_elliptic_envelope.py

1) Simula o carga G1, G2
2) Calcula b subgrafos bootstrap de tamaño k en G1 → matriz S (b×3)
3) Ajusta EllipticEnvelope(contamination=0.05) a S
4) Test multivariante en G2 (Mahalanobis² ≤ χ²₀.₉₅(3))
5) Plot 3D: S, elipsoide 95% y punto de G2
"""

import numpy as np
import networkx as nx
import Random_Graphs_Complex_InferencePython.RSG_simulation.SimulateG1G2 as SimulateG1G2              # tu módulo con simulate_or_load()
from sklearn.covariance import EllipticEnvelope
from scipy.stats import chi2
import plotly.graph_objs as go
import community as community_louvain
from scipy.stats import gaussian_kde

np.random.seed(42+8)

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


def normalized_triangles(G):
    """Número de triángulos normalizado en [0,1]."""
    t = sum(nx.triangles(G).values()) / 3
    k = G.number_of_nodes()
    max_t = k*(k-1)*(k-2)/6
    return t/max_t if max_t>0 else 0.0

def compute_stats_subgraph(G, k):
    """Devuelve el triplete (f1,f2,f3) de un subgrafo aleatorio de k vértices."""
    nodes = np.random.choice(list(G), k, replace=False)
    sub = G.subgraph(nodes)
    # f1 = common_neighbours subsampled
    S = min(10000, n*(n-1)//2)
    common_sampled = sample_common_counts(sub, S) 
    xmin, xmax = 0, max(common_sampled.max(), common_sampled.max())
    xs = np.linspace(xmin, xmax, 1000)

    kde1_sampled = gaussian_kde(common_sampled)
    dens1 = kde1_sampled(xs)
    dx = xs[1] - xs[0]
    L1_approx  = np.sum(np.abs(dens1)) * dx
    Linf_approx = np.max(np.abs(dens1))
    f1 = L1_approx / Linf_approx if Linf_approx > 0 else 0.0
    # f2 = distancia media (sobre componente gigante)
    partition = community_louvain.best_partition(sub)
    f2 = community_louvain.modularity(partition, sub)
    # f3 = triángulos normalizados
    f3 = normalized_triangles(sub)* 15 if sub.number_of_edges()>0 else 0.0
    return np.array([f1, f2, f3])

def compute_stats_subgraph_fast(G, k):
    # 1) extraemos subgrafo y su matriz
    nodes = np.random.choice(list(G), k, replace=False)
    A = nx.to_numpy_array(G.subgraph(nodes), dtype=int)  # k×k

    # 2) producto matricial para vecinos comunes
    common_mat = A.dot(A)                              # O(k^3)

    # 3) histogram de los C(k,2) conteos
    iu = np.triu_indices(k, k=1)
    common_counts = common_mat[iu]                     # O(k^2)

    # 4) KDE & L1/L∞
    xs = np.linspace(0, common_counts.max(), 1000)
    dens = gaussian_kde(common_counts)(xs)
    dx  = xs[1] - xs[0]
    L1  = np.sum(np.abs(dens))*dx
    Linf= np.max(np.abs(dens))
    f1 = L1 / Linf if Linf>0 else 0.0

    # 5) modularidad
    sub = G.subgraph(nodes)
    partition = community_louvain.best_partition(sub)
    f2 = community_louvain.modularity(partition, sub)

    # 6) triángulos normalizados
    f3 = normalized_triangles(sub) if sub.number_of_edges()>0 else 0.0

    return np.array([f1, f2, f3])

def fit_mve_ee(S):
    """
    Ajusta EllipticEnvelope al 95% de los datos S (b×d).
    contamination=0.05 → deja fuera el 5% más extremo.
    """
    ee = EllipticEnvelope(contamination=0.05).fit(S)
    center = ee.location_        # (d,)
    cov    = ee.covariance_      # (d,d)
    return ee, center, cov

def test_point(ee, stat2, d):
    """
    Calcula Mahalanobis² de stat2 con ee, y lo compara
    con el cuantil χ²(0.95,d).
    """
    m2 = ee.mahalanobis(stat2.reshape(1,-1))[0]
    thr = chi2.ppf(0.95, df=d)
    return m2, (m2 <= thr)

if __name__=='__main__':
    # — 1) Simula o carga tus grafos G1, G2
    cache = 'simulated_graphs.pkl'
    n = 500
    p = n//20 + 1
    sigma   = 1e5
    m1= 25  
    m2 = 33
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

    # — 2) Bootstrap en G1
    b = 200
    n = G1.number_of_nodes()
    k = int(round(n**(2/3)))
    S = np.vstack([ compute_stats_subgraph_fast(G1,k) for _ in range(b) ])

    # — 3) Ajusta elipsoidEE al 95%
    ee, center, cov = fit_mve_ee(S)

    # — 4) Estadístico Mahalanobis² y test en G2
    stat2 = compute_stats_subgraph(G2,k)
    m2, accept = test_point(ee, stat2, d=S.shape[1])
    print(f"Test MVE (EE): H₀ {'ACEPTADA' if accept else 'RECHAZADA'} (M²={m2:.3f})")

    # — 5) Prepara elipsoide para plot
    #    Para dibujar elipsoide con Mah² = χ²₀.₉₅
    #    x = center + V · diag(√(λ_i·thr)) · sphere_coords
    vals, vecs = np.linalg.eigh(cov)
    thr = chi2.ppf(0.95, df=len(vals))
    axes = np.sqrt(vals * thr)
    Vsub = vecs

    # Mesh spherical coords
    u = np.linspace(0,2*np.pi,30)
    v = np.linspace(0,   np.pi,15)
    U,V = np.meshgrid(u,v)
    xs =  np.cos(U)*np.sin(V)
    ys =  np.sin(U)*np.sin(V)
    zs =  np.cos(V)
    pts = np.vstack([xs.ravel(), ys.ravel(), zs.ravel()])   # 3×m

    # Transform
    E = Vsub @ np.diag(axes) @ pts
    E = E + center.reshape(3,1)
    Xe = E[0].reshape(xs.shape)
    Ye = E[1].reshape(xs.shape)
    Ze = E[2].reshape(xs.shape)

    # — 6) Plot 3D con plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
      x=S[:,0], y=S[:,1], z=S[:,2],
      mode='markers',
      marker=dict(size=3, color='steelblue', opacity=0.6),
      name='Bootstrap G1'
    ))
    fig.add_trace(go.Surface(
      x=Xe, y=Ye, z=Ze,
      opacity=0.3, colorscale='Greys', showscale=False,
      name='Elipsoide EE 95%'
    ))
    fig.add_trace(go.Scatter3d(
      x=[stat2[0]], y=[stat2[1]], z=[stat2[2]],
      mode='markers',
      marker=dict(size=8,
                  color=('green' if accept else 'red')),
      name='Stat G2'
    ))
    fig.update_layout(
      title=f"Test MVE 95% (EllipticEnvelope) — H₀ {'ACEPTADA' if accept else 'RECHAZADA'}",
      scene = dict(
        xaxis_title='f1',
        yaxis_title='f2: modularidad',
        zaxis_title='f3: triNorm',
        camera = dict(eye=dict(x=1.3,y=1.3,z=0.8))
      ),
      margin = dict(l=0,r=0,b=0,t=40)
    )
    fig.show()
