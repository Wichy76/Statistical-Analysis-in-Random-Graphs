set.seed(3)

# Tamaño de los grafos
n = 500

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
weights = c(1/2, 1/4, 1/8, rep(1/(8*(m-3)), m - 3)) 

# Generate data on the sphere and the base graph
sample_sphere <- generate_mixture_sphere_sample(mu, Sigma_list, u_list, weights, n)
graph <- generate_random_dot_product_graph(sample_sphere)
