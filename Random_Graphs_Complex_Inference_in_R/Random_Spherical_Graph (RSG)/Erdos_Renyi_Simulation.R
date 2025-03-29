

# Parámetros
n <- 1000
p2 <- log(n)*(1-1/n) / n
p = 2/n
print(n**(2/3))

g <- erdos.renyi.game(n = n, p.or.m = p2, type = "gnp", directed = FALSE)

# Layout artístico
layout_coords <- layout_with_kk(g)

# Abrir PDF para guardar
pdf("Erdos_Renyi_Dense_Simulation.pdf", width = 20, height = 20, bg = "white")  # bg puede ser "transparent"

# Plot dentro del PDF
plot(g,
     layout = layout_coords,
     vertex.size = 0.5,
     vertex.color = rgb(0, 0, 0, alpha = 0.7),
     vertex.frame.color = NA,
     edge.color = rgb(0, 0, 0, alpha = 0.3),
     edge.width = 0.7,
     vertex.label = NA,
     margins = 0,)

dev.off()  # Cerrar PDF

