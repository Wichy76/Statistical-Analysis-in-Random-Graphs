install.packages("scatterplot3d")  # Run once
install.packages("rgl")  # Run once
install.packages("pracma")
install.packages("igraph")  # Run once
install.packages("MASS")  # Run once
install.packages("visNetwork")  # Run once

library(MASS)
library(pracma)
library(scatterplot3d)
library(rgl)
library(igraph)
library(visNetwork)


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