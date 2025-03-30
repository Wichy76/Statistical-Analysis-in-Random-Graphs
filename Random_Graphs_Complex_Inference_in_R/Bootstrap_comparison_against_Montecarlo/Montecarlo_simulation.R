# Tamaño de las muestras
b = 1000

# Tamaño de los grafos
n = 800
k <- n/2
#n = k

# Parámetros de las distribuciones
sigma = 10000
m = 40
p = m
mu <- numeric(p - 1)
canonical_basis <- lapply(1:p, function(i) { v <- numeric(p)
v[i] <- 1  
return(v)
})
u_list <- canonical_basis[1:m] 
Sigma_list <- lapply(1:m, function(i) 1/(sigma)*diag(p - 1))  #List of covariance matrices
weights = c(rep(1/m, m)) 

# Generate the data
sample_sphere <- generate_mixture_sphere_sample(mu, Sigma_list, u_list, weights, n)

# Generate a random dot product graph from the sample on the sphere
graph <- generate_random_dot_product_graph(sample_sphere)



visualizar_interactivo <- function(graph, file_name = NULL, title = "Visualización Interactiva del Grafo") {
  if (!requireNamespace("visNetwork", quietly = TRUE)) install.packages("visNetwork")
  if (!requireNamespace("htmlwidgets", quietly = TRUE)) install.packages("htmlwidgets")
  if (!requireNamespace("igraph", quietly = TRUE)) install.packages("igraph")
  
  library(visNetwork)
  library(igraph)
  
  # Detectar comunidades (solo para agrupamiento)
  comunidades <- cluster_louvain(graph)
  membership <- membership(comunidades)
  
  # Calcular layout DRL y extraer coordenadas
  layout_coords <- layout_with_drl(graph)
  layout_df <- data.frame(id = 1:vcount(graph),
                          x = layout_coords[,1],
                          y = layout_coords[,2])
  
  # Crear nodos (sin color por comunidad)
  nodes <- data.frame(
    id = layout_df$id,
    group = as.character(membership),  # usado solo para agrupar en layout
    size = 3,
    color = "gray30",
    x = layout_df$x * 100,
    y = layout_df$y * 100
  )
  
  # Crear aristas con opacidad baja y líneas delgadas
  edges <- as_data_frame(graph, what = "edges")
  edges$color <- "rgba(128,128,128,0.2)"
  edges$width <- 0.1
  
  # Visualización con layout fijo
  vis <- visNetwork(nodes, edges, main = title) %>%
    visOptions(highlightNearest = FALSE, nodesIdSelection = FALSE) %>%
    visNodes(fixed = TRUE)
  
  print(vis)
  
  if (!is.null(file_name)) {
    htmlwidgets::saveWidget(vis, file = file_name, selfcontained = FALSE)
    message("Archivo guardado como: ", file_name)
  }
}


visualizar_interactivo(graph, file_name = "grafo_interactivo.html")


pdf("test.pdf", width = 40, height = 40, bg = "white")  # bg puede ser "transparent"
plot(graph, vertex.size = 0.5, vertex.color = rgb(0, 0, 0, alpha = 1),
     vertex.frame.color = NA, edge.color = rgb(0, 0, 0, alpha = 0.2), edge.width = 0.7,
     vertex.label = NA, margins = 0,, main="Random Graph from sample on the sphere")

dev.off()  # Cerrar PDF
