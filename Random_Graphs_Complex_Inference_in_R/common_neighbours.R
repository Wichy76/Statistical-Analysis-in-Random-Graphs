 
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

main_title <- sprintf("Distribución de vecinos comunes entre pares de nodos\n con pesos (1/2, 1/4, 1/8, 1/56 ..., 1/56)/m \n cov = diag(%d), n = %d, m=%d, p=%d", sigma, n, m, p)

# Hacer histograma
hist(common_counts,
     main = main_title,
     xlab = "Número de vecinos comunes",
     ylab = "Frecuencia",
     col = "steelblue",
     breaks = max(common_counts) + 1)


pdf("test.pdf", width = 40, height = 40, bg = "white")  # bg puede ser "transparent"
plot(graph, vertex.size = 0.5, vertex.color = rgb(0, 0, 0, alpha = 1),
     vertex.frame.color = NA, edge.color = rgb(0, 0, 0, alpha = 0.2), edge.width = 0.7,
     vertex.label = NA, margins = 0,, main= main_title)

dev.off()  # Cerrar PDF
