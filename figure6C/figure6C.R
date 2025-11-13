library(VennDiagram)

df <- read.table("figure6C.txt",header = T)
A <- df$'gMLP'
B <- df$'SVR'
C <- df$'SPLS'
D <- df$'XGBoost'
E <- df$'BayesA'

A_set <- na.omit(as.character(A))
B_set <- na.omit(as.character(B))
C_set <- na.omit(as.character(C))
D_set <- na.omit(as.character(D))
E_set <- na.omit(as.character(E))
venn.plot <- venn.diagram(
  x = list(
    'gMLP' = A_set,
    'SVR' = B_set,
    'SPLS' = C_set,
    'XGBoost' = D_set,
    'BayesA' = E_set         
  ),
  filename = "figure6C.tif",   
  height = 2000,  
  width = 2000,   
  resolution = 300,  
  margin = 0.08,     
  col = "transparent",    
  fill = c("#4DBBD5", "#E64B35", "#F39B7F", "#3C5488", "#00A087"),   
  alpha = 0.3,  
  cat.col =c("black","black","black","black","black"), 
  cat.cex = 1.8,  
  #cat.fontface = "bold",  
  cat.dist = c(0.18,0.2,0.18,0.2,0.2),   
  cat.pos = c(0, -30, -150, 150,20), 
  cex = 1.5,        
  fontfamily = "Arial",  
  cat.fontfamily = "Arial"  
)


dev.off()