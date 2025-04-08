# Function to calculate the inverse stereographic projection.
calculate_inverse_projection <- function(u, p) {
  if (!is.numeric(u) || !is.numeric(p) || length(u) != length(p)) {
    stop("Both mu and p must be numeric vectors of the same length.")
  }
  
  norm_p_u_squared <- sum((p + u)^2)  # Euclidean norm ||p + u||^2
  scaling_factor <- 2 / norm_p_u_squared
  
  # Compute the inverse stereographic projection
  x <- scaling_factor * (p + u) - u
  
  return(x)
}

# Function to generate sample of size n orthogonal to u with N(mu,Sigma) distribution.
generate_orthogonal_sample <- function(u, mu, Sigma, n) {
  p <- length(u)  # Dimension of R^p
  
  # Check if u is a unit vector
  if (abs(sum(u^2) - 1) > 1e-6) {
    stop("u must be a unit vector (norm 1).")
  }
  
  # Check dimensions
  if (length(mu) != (p - 1) || nrow(Sigma) != (p - 1) || ncol(Sigma) != (p - 1)) {
    stop("mu must be in R^(p-1) and Sigma must be a (p-1)x(p-1) covariance matrix.")
  }
  
  # Compute an orthonormal basis for the orthogonal complement of u
  basis <- Null(u)  # p x (p-1), each column is a basis vector of the orthogonal complement of u
  
  # Generate n samples from N(mu, Sigma) in R^(p-1)
  samples <- mvrnorm(n, mu, Sigma)  # n x (p-1), each row is point in R^(p-1)
  
  # Map samples to R^p using the orthogonal basis (n x (p-1)) %*% ((p-1) x p) -> (n x p)
  orthogonal_samples <- samples %*% t(basis)  # Each row is a point in R^p orthogonal to u
  
  return(orthogonal_samples)  # Each row is a sample in R^p, orthogonal to u
}

# Function to generate sample of size n on the unit sphere from inverses stereographic images
generate_mixture_sphere_sample <- function(mu, Sigma_list, u_list, weights, n) {
  k <- length(weights)         # Number of components
  p <- length(u_list[[1]])     # Dimension of R^p
  
  # Validate inputs
  if (length(mu) != (p - 1)) stop("mu must be in R^(p-1)a.")
  if (length(Sigma_list) != k || length(u_list) != k) stop("Mismatch in number of components.")
  if (abs(sum(weights) - 1) > 1e-6) stop("Weights must sum to 1.")
  
  # Precompute orthogonal bases for each u_i
  basis_list <- lapply(u_list, function(u) {
    if (abs(sum(u^2) - 1) > 1e-6) stop("Each u_i must be a unit vector.")
    Null(u)  # Basis of orthogonal complement
  })
  
  # Determine how many samples per component using multinomial draw
  sample_counts <- as.numeric(rmultinom(1, n, prob = weights))
  
  # Initialize result matrix
  samples_all <- matrix(0, nrow = n, ncol = p)
  
  row_idx <- 1  # Row index to fill samples
  for (i in 1:k) {
    ni <- sample_counts[i]  # Samples for component i
    
    if (ni > 0) {
      # Generate orthogonal samples for component i
      sample_orthogonal_to_ui <- generate_orthogonal_sample(u_list[[i]], mu, Sigma_list[[i]], ni)
      
      mapped_samples <- t(apply(sample_orthogonal_to_ui, 1, function(x) calculate_inverse_projection(u_list[[i]], x)))
      
      # Store in result matrix
      samples_all[row_idx:(row_idx + ni - 1), ] <- mapped_samples
      row_idx <- row_idx + ni
    }
  }
  
  return(samples_all)  # Each row is a sample in R^p
}

# Function to generate a random graph from a given probability matrix P
generate_random_graph <- function(P) {
  n <- nrow(P)
  
  # Generate upper triangle of adjacency matrix (excluding diagonal)
  upper_tri <- matrix(runif(n * n), nrow = n)
  adjacency_matrix <- (upper_tri < P) * 1  # 1 if random number < P_ij, else 0
  
  # Ensure symmetry and zero diagonal
  adjacency_matrix[lower.tri(adjacency_matrix)] <- t(adjacency_matrix)[lower.tri(adjacency_matrix)]
  diag(adjacency_matrix) <- 0
  
  # Create igraph object from adjacency matrix
  graph <- graph_from_adjacency_matrix(adjacency_matrix, mode = "undirected", diag = FALSE)
  
  return(graph)
}

#Function to generate a random dot product graph from a sample on the unit sphere
generate_random_dot_product_graph <- function(sample_sphere) {
  n <- nrow(sample_sphere)
  
  # Compute dot product matrix
  dot_product_matrix <- sample_sphere %*% t(sample_sphere)  # Matrix n x n, each entry is the dot product between two data points in the sphere S^{p-1}
  arccos_matrix <- acos(pmin(pmax(dot_product_matrix, -1), 1))  # Clamp to [-1,1] for numerical stability
  P <- cos(0.5 * arccos_matrix)**13  # Exponentiate to make probabilities closer to 0 or 1
  diag(P) <- 0  # Diagonal must be 0
  
  graph <- generate_random_graph(P)
  
  return(graph)
}

# Example of generating a random dot product graph

m<- 10   # Size of u_list (number of communities i hope to have)
p <- 10  # Dimension of R^p
mu <- numeric(p - 1)

# Example with u_list unformly distributed in the sphere

#U <- matrix(rnorm(m*p), nrow = m, ncol = p)
#U <- U / sqrt(rowSums(U^2))  # Normalize rows to unit vectors
#u_list <- asplit(U, 1)  # List of unit vectors

# Example with u_list having angle between any two vector bigger than 2pi in R^3
#u_list <- list(c(1,0,0), c(0,1,0), c(0,0,1), c(-1,0,0), c(0,-1,0), c(0,0,-1))  # List of unit vectors

# Example with u_list all orthogonal
# Condition: m <= p
canonical_basis <- lapply(1:p, function(i) { v <- numeric(p)
v[i] <- 1  
return(v)
})
u_list <- canonical_basis[1:m] 

# Example with u_list having it first 2p vector having angles bigger than 2pi and the rest randomly chosen
# Condition: 2p < m
#canonical_basis <- lapply(1:p, function(i) { v <- numeric(p)  
#v[i] <- 1  
#return(v)})
#canonical_antipodal_basis <- lapply(1:p, function(i) { v <- numeric(p) 
#v[i] <- -1 
#return(v)})
#U <- matrix(rnorm((m-2*p)*p), nrow = m-2*p, ncol = p)
#U <- U / sqrt(rowSums(U^2))  # Normalize rows to unit vectors
#u_list <- c(canonical_basis, canonical_antipodal_basis, asplit(U, 1) )

Sigma_list <- lapply(1:m, function(i) 1/(200000000000000000)*diag(p - 1))  #List of covariance matrices
weights = c(rep(1/m, m))                                           #Weights for each component
n = 200

# Matrix n x p, each row is a data point in the sphere S^{p-1}
sample_sphere <- generate_mixture_sphere_sample(mu, Sigma_list, u_list, weights, n)

# Generate a random dot product graph from the sample on the sphere
graph <- generate_random_dot_product_graph(sample_sphere)
plot(graph, vertex.size=3, vertex.label=NA, main="Random Graph from sample on the sphere")
