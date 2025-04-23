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
import SimulateG1G2              # tu módulo con simulate_or_load()
from sklearn.covariance import EllipticEnvelope
from scipy.stats import chi2
import plotly.graph_objs as go

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
    # f1 = assortativity
    a = nx.degree_assortativity_coefficient(sub)
    f1 = 0.0 if np.isnan(a) else a
    # f2 = distancia media (sobre componente gigante)
    if nx.is_connected(sub):
        lengths = dict(nx.all_pairs_shortest_path_length(sub))
        dists = [d for u in lengths for v,d in lengths[u].items() if u<v]
    else:
        comp = max(nx.connected_components(sub), key=len)
        lengths = dict(nx.all_pairs_shortest_path_length(sub.subgraph(comp)))
        dists = [d for u in lengths for v,d in lengths[u].items() if u<v]
    f2 = np.mean(dists) if dists else 0.0
    # f3 = triángulos normalizados
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
    params = (
      500, 26, 1e5, 50, [1/50]*50,
      1e5, 50, [1/50]*50, 42
    )
    G1, G2 = SimulateG1G2.simulate_or_load(cache, params, force=False)

    # — 2) Bootstrap en G1
    b = 200
    n = G1.number_of_nodes()
    k = int(round(n**(2/3)))
    S = np.vstack([ compute_stats_subgraph(G1,k) for _ in range(b) ])

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
        xaxis_title='f1: assort',
        yaxis_title='f2: avgDist',
        zaxis_title='f3: triNorm',
        camera = dict(eye=dict(x=1.3,y=1.3,z=0.8))
      ),
      margin = dict(l=0,r=0,b=0,t=40)
    )
    fig.show()
