library(igraph)
library(MASS)
library(pracma)

# ------------------------------------------
# Proyección estereográfica inversa (vectorizada)
calculate_inverse_projection <- function(u, P) {
  sum_sq <- rowSums((t(t(P) + u))^2)
  scale <- 2 / sum_sq
  x <- sweep(t(t(P) + u), 1, scale, "*") - matrix(rep(u, each = nrow(P)), ncol = length(u), byrow = FALSE)
  return(x)
}

# Muestras ortogonales a u con N(mu, Sigma)
generate_orthogonal_sample <- function(u, mu, Sigma, n) {
  basis <- Null(u)
  samples <- mvrnorm(n, mu, Sigma)
  return(samples %*% t(basis))
}

# Muestras en la esfera unidad como mezcla
generate_mixture_sphere_sample <- function(mu, Sigma_list, u_list, weights, n) {
  k <- length(weights)
  p <- length(u_list[[1]])
  sample_counts <- as.vector(rmultinom(1, n, weights))
  samples_all <- matrix(0, nrow = n, ncol = p)
  idx <- 1
  for (i in seq_len(k)) {
    ni <- sample_counts[i]
    if (ni > 0) {
      orthogonal_samples <- generate_orthogonal_sample(u_list[[i]], mu, Sigma_list[[i]], ni)
      projected <- calculate_inverse_projection(u_list[[i]], orthogonal_samples)
      samples_all[idx:(idx + ni - 1), ] <- projected
      idx <- idx + ni
    }
  }
  return(samples_all)
}

# Grafo aleatorio desde puntos en la esfera
generate_random_dot_product_graph <- function(sample_sphere) {
  n <- nrow(sample_sphere)
  dot_matrix <- sample_sphere %*% t(sample_sphere)
  arccos_matrix <- acos(pmin(pmax(dot_matrix, -1), 1))
  P <- (cos(0.5 * arccos_matrix))^13
  diag(P) <- 0
  rand_mat <- matrix(runif(n * n), nrow = n)
  adj <- matrix(0, n, n)
  upper_idx <- upper.tri(P)
  adj[upper_idx] <- as.numeric(rand_mat[upper_idx] < P[upper_idx])
  adj <- adj + t(adj)
  return(graph_from_adjacency_matrix(adj, mode = "undirected", diag = FALSE))
}

# Base canónica sin lapply
canonical_basis_list <- function(p, m) {
  basis_mat <- diag(p)
  lapply(1:m, function(i) basis_mat[, i])
}

# Grafo de prueba
generate_test_graph <- function(n, p, sigma, m, weights) {
  mu <- rep(0, p - 1)
  u_list <- canonical_basis_list(p, m)
  Sigma_list <- replicate(m, (1 / sigma) * diag(p - 1), simplify = FALSE)
  sample_sphere <- generate_mixture_sphere_sample(mu, Sigma_list, u_list, weights, n)
  generate_random_dot_product_graph(sample_sphere)
}

# Estadístico de triángulos
normalized_triangles <- function(g) {
  t <- sum(count_triangles(g)) / 3
  k <- vcount(g)
  max_tri <- choose(k, 3)
  return(ifelse(max_tri > 0, t / max_tri, 0))
}

# Test de hipótesis
hypothesis_test_triangles <- function(G1, G2, b = 200, k = round(vcount(G1)^(2/3))) {
  stats_G1 <- replicate(b, {
    nodes <- sample(V(G1), k)
    normalized_triangles(induced_subgraph(G1, nodes))
  })
  ci <- quantile(stats_G1, c(0.025, 0.975))
  stat_G2 <- normalized_triangles(induced_subgraph(G2, sample(V(G2), k)))
  
  cat("Estadístico en G2:", stat_G2, "\n")
  cat("Intervalo 95% de bootstrap:", ci, "\n")
  cat("Resultado del test:", ifelse(stat_G2 >= ci[1] && stat_G2 <= ci[2], "NO se rechaza H0", "Se rechaza H0"), "\n")
  
  plot(stats_G1, col = "gray40", pch = 16, main = sprintf("Test b=%d, k=%d", b, k),
       xlab = "Bootstrap index", ylab = "Conteo de triángulos normalizado")
  abline(h = ci, col = "blue", lty = 2)
  points(b + 10, stat_G2, col = ifelse(stat_G2 >= ci[1] && stat_G2 <= ci[2], "forestgreen", "red"), pch = 19, cex = 1.5)
}

# Vecinos comunes
compute_common_neighbor_counts <- function(g) {
  adj <- as_adj(g, sparse = FALSE)
  common_mat <- adj %*% adj
  common <- common_mat[upper.tri(common_mat)]
  return(common)
}

# ------------------------------------------
# Parámetros de los grafos (EDITABLES)
# ------------------------------------------
n <- 1000
p <- 200


# Grafo G1
sigma1 <- 100000
m1 <- 10
weights1 <- rep(1/m1, m1)

# Grafo G2
sigma2 <- 100000   # diferente
m2 <- 2
weights2 <- rep(1/m2, m2)

# ------------------------------------------
# Ejecutar test
# ------------------------------------------

set.seed(42)
G1 <- generate_test_graph(n, p, sigma1, m1, weights1)
G2 <- generate_test_graph(n, p, sigma2, m2, weights2)

hypothesis_test_triangles(G1, G2)


# ------------------------------------------
# Visualizar los grafos G1 y G2
# ------------------------------------------

# Layout común para comparación visual
layout1 <- layout_with_fr(G1)
layout2 <- layout_with_fr(G2)

par(mfrow = c(1, 2))  # Dividir ventana gráfica

# Grafo G1
plot(G1, layout = layout1,
     vertex.size = 3,               # Tamaño del nodo
     vertex.color = "black",        # Color negro para nodos
     vertex.label = NA,
     edge.color = "gray70",         # Aristas grises
     edge.width = 0.5,              # Grosor fino de aristas
     main = "Grafo G1")

# Grafo G2
plot(G2, layout = layout2,
     vertex.size = 3,
     vertex.color = "black",
     vertex.label = NA,
     edge.color = "gray70",
     edge.width = 0.5,
     main = "Grafo G2")

par(mfrow = c(1, 1))  # Restaurar ventana

# ------------------------------------------
# Comparar densidades de vecinos comunes entre G1 y G2
# ------------------------------------------

# Función para calcular el número de vecinos comunes entre pares
compute_common_neighbor_counts <- function(g) {
  neighbors_list <- adjacent_vertices(g, V(g))
  n <- vcount(g)
  counts <- numeric()
  
  for (i in 1:(n - 1)) {
    ni <- neighbors_list[[i]]
    for (j in (i + 1):n) {
      nj <- neighbors_list[[j]]
      counts <- c(counts, length(intersect(ni, nj)))
    }
  }
  
  return(counts)
}

# Calcular vectores de vecinos comunes
cat("Calculando vecinos comunes para G1...\n")
common_counts_G1 <- compute_common_neighbor_counts(G1)

cat("Calculando vecinos comunes para G2...\n")
common_counts_G2 <- compute_common_neighbor_counts(G2)

# Calcular densidades
dens_G1 <- density(common_counts_G1, from = 0)
dens_G2 <- density(common_counts_G2, from = 0)

# Título informativo
main_title <- sprintf("Densidad de vecinos comunes entre pares de nodos\nG1: m=%d, sigma=%d | G2: m=%d, sigma=%d, n=%d, p=%d",
                      m1, sigma1, m2, sigma2, n, p)

# Plot de densidades
plot(dens_G1, col = "black", lwd = 2, ylim = range(0, dens_G1$y, dens_G2$y),
     main = main_title,
     xlab = "Número de vecinos comunes",
     ylab = "Densidad")

lines(dens_G2, col = "red", lwd = 2)
legend("topright", legend = c("G1", "G2"), col = c("black", "red"), lwd = 2)
