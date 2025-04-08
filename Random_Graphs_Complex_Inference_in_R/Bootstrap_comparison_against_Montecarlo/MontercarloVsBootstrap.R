set.seed(3)

# Tamaño de las muestras
b = 200

l = 200   # n/k
# Tamaño de los grafos
k <- 65
n = k*l

# Parámetros de las distribuciones
sigma = 1000
m = 10
p = m
mu <- numeric(p - 1)
canonical_basis <- lapply(1:p, function(i) { v <- numeric(p)
v[i] <- 1  
return(v)
})
u_list <- canonical_basis[1:m] 
Sigma_list <- lapply(1:m, function(i) 1/(sigma)*diag(p - 1))  #List of covariance matrices
weights = c(rep(1/m, m)) 

# Generate data on the sphere and the base graph
sample_sphere <- generate_mixture_sphere_sample(mu, Sigma_list, u_list, weights, n)
graph <- generate_random_dot_product_graph(sample_sphere)

# Step 2: Compute the normalized triangle count statistic
normalized_triangles <- function(g) {
  t <- count_triangles(g)
  k <- vcount(g)
  max_tri <- choose(k, 3)
  return(t / max_tri)
}

# Step 3: Bootstrap from the observed graph
bootstrap_graphs <- function(graph, b, k) {
  nodes <- V(graph)
  stats <- numeric(b)
  
  for (i in 1:b) {
    sampled_nodes <- sample(nodes, k, replace = FALSE)
    subgraph <- induced_subgraph(graph, sampled_nodes)
    stats[i] <- normalized_triangles(subgraph)
  }
  
  return(stats)
}

# Step 4: Monte Carlo simulation from the original model
generate_from_model <- function(k, b, mu, Sigma_list, u_list, weights) {
  stats <- numeric(b)
  for (i in 1:b) {
    sample_sphere <- generate_mixture_sphere_sample(mu, Sigma_list, u_list, weights, k)
    g <- generate_random_dot_product_graph(sample_sphere)
    stats[i] <- normalized_triangles(g)
  }
  return(stats)
}

# ---------------------------
# Run bootstrap and model-based simulations
bootstrap_stats <- bootstrap_graphs(graph, b, k)
model_stats <- generate_from_model(k, b, mu, Sigma_list, u_list, weights)

diferences <- model_stats - bootstrap_stats
# ---------------------------
# Step 5: Plot both distributions for the value of n

main_title <- sprintf("Bootstrap vs. Model-based simulation\nn = %d, k = %d, b = %d, n/k = %d", n, k, b, l)

hist(bootstrap_stats, col = "gray", 
     main = main_title,
     xlab = "Normalized triangle count", ylab = "Frequency", 
     breaks = 50)
hist(model_stats, col = "red", add = TRUE, breaks = 50)

plot(density(bootstrap_stats),
     col = "gray",
     main = main_title,
     xlab = "Normalized triangle count",
     ylab = "Density")
lines(density(model_stats), col = "red")
