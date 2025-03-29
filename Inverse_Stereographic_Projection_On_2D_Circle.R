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
  basis <- Null(u)  # Each column is a basis vector in R^p
  
  # Generate n samples from N(mu, Sigma) in R^(p-1)
  samples <- mvrnorm(n, mu, Sigma)  # n x (p-1)
  
  # Map samples to R^p using the orthogonal basis (n x (p-1)) %*% ((p-1) x p) -> (n x p)
  orthogonal_samples <- samples %*% t(basis)
  
  return(orthogonal_samples)  # Each row is a sample in R^p, orthogonal to u
}


u <- c(0,-1)  
mu <- c(0)
Sigma <- 10*diag(1)

n <- 1000  # Number of samples

samples_orthogonal <- generate_orthogonal_sample(u, mu, Sigma, n)
samples <- t(apply(samples_orthogonal, 1, function(x) calculate_inverse_projection(u, x)))

# Graph of samples orthogonal to u
plot(samples_orthogonal, col = "blue", pch = 20, xlab = "X", ylab = "Y", main = "Samples Orthogonal to u")

# Graph of samples in the sphere
plot(samples, col = "red", pch = 20, xlab = "X", ylab = "Y", main = "Samples in the Sphere")
