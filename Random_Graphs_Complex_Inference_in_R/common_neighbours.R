 
g <- graph

neighbors_list <- adjacent_vertices(g, V(g))

# Inicializar vector de frecuencias
common_counts <- c()

# Para cada par de vértices (i < j)
n <- vcount(g)
for (i in 1:(n-1)) {
  ni <- neighbors_list[[i]]
  for (j in (i+1):n) {
    nj <- neighbors_list[[j]]
    
    # Contar vecinos comunes (intersección)
    n_common <- length(intersect(ni, nj))
    common_counts <- c(common_counts, n_common)
  }
}

# Hacer histograma
hist(common_counts,
     main = "Distribución de vecinos comunes entre pares de nodos",
     xlab = "Número de vecinos comunes",
     ylab = "Frecuencia",
     col = "steelblue",
     breaks = max(common_counts) + 1)
