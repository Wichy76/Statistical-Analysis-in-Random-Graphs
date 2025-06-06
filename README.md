# Statistical-Analysis-in-Random-Graphs

This project collects all the statistical analysis, simulation tools, and hypothesis testing procedures developed in my undergraduate mathematics thesis. The main objective is to explore non-parametric two-sample hypothesis testing for complex networks, particularly in a high-dimensional setting where graphs are modeled via latent positions on the sphere.

The core model is the **Random Spherical Graph (RSG)**, where nodes are embedded on the sphere as a Gaussian mixture projected onto subspaces, and edges are drawn based on spherical distances. The testing procedure is based on evaluating network statistics over bootstrapped subgraphs and comparing them through a Mahalanobis-type confidence region.

![masterfrontpage-background_page-0001](https://github.com/user-attachments/assets/4c4d906a-a61d-444d-91c1-b6a6a8768991)

---

## 📁 Project Structure

```
.
├── core/
│   ├── generators.py          # Functions to simulate latent positions and generate RSG graphs
│   ├── hypothesis_test.py     # Core two-sample hypothesis testing logic (bootstrap + Mahalanobis ellipse)
│   └── statistics.py          # Computation of various network statistics (e.g., triangles, modularity)
│
├── datasets_simulation/
│   ├── vary_num_communities.py         # Generates datasets with varying number of communities
│   ├── vary_weights.py                 # Generates datasets with different weight distributions
│   ├── vary_sigma.py                  # Generates datasets with varying intra-cluster variance
│   ├── vary_all.py                    # Combines variation in communities, weights, and sigma
│   ├── *_ruido.py                     # Versions of the above with additional "noise" components
│
├── data/
│   └── ...                            # Contains serialized .pkl graphs organized by dataset
│
├── experiments/
│   ├── run_simulations.py            # Runs the two-sample test for a given dataset and stat pair
│   ├── test_ellipticity_assumption.py # Verifies if statistic pairs follow elliptical symmetry
│   └── config.yaml                   # Controls which dataset and stats to use in experiments
│
├── visualization/
│   └── *_results/                    # Stores generated images of bootstrap regions + test stats
│
└── README.md                         # This file
```

---

## 🔬 Methods Overview

- Graphs are sampled from a latent Gaussian mixture on the sphere (`generate_mixture_sphere_sample`)
- Network statistics are computed over 200 bootstrapped subgraphs of graph G1
- A 95% confidence ellipse is fit to this distribution (via `EllipticEnvelope`)
- The statistic of graph G2 is tested against this region (rejected if outside)
- Results are summarized in terms of empirical Type I and Type II errors

---

## 📊 Available Statistics

Implemented network statistics include:

- Triangle density
- Modularity (Louvain)
- Assortativity
- Average degree
- Spectral features (e.g., top eigenvalue)
- Custom combinations for bivariate testing

---

## ⚙️ Running Experiments

1. Edit `experiments/config.yaml` to select:
   - `dataset_path`: folder with pre-generated graphs (e.g. `data/vary_num_communities/n2000`)
   - `stat_names`: list of two stat names to test (e.g. `["triangles", "modularity"]`)
   - `n_trials`: number of repetitions
   - `create_images`: whether to generate visualizations

2. Run the test:
   ```bash
   python experiments/run_simulations.py
   ```

3. Results:
   - Printed summary of Type I / Type II errors
   - Optionally, visual plots in `visualization/` folder

---

## 🧪 Dataset Generation

To generate synthetic graph datasets, run one of the scripts in `datasets_simulation/`. For example:

```bash
python datasets_simulation/vary_num_communities.py
```

Each script controls one experimental dimension: number of communities, weight balance, density, or combinations (with/without noise).

---

## 📚 Thesis Reference

This codebase supports the empirical component of the thesis:

> **Pruebas de hipótesis no paramétricas para redes complejas**  
> Undergraduate thesis in mathematics, Universidad de loa Andes de Colombia  
> Author: Luis Ernest Tejón Rojas
> Advisor: Adolfo Quiroz

For full theoretical context, methodology, and mathematical background, refer to the compiled thesis PDF.

---

## 📦 Requirements

- Python ≥ 3.9
- numpy
- networkx
- scikit-learn
- matplotlib
- tqdm
- pyyaml



---

## 📄 License

This repository is intended for academic and research use. Feel free to fork, cite, or adapt.
