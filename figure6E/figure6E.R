library(dplyr)
library(ggplot2)  
library(tidyverse)


BP = read.table("figure6E.txt", head = TRUE, sep = "\t", quote = "")  

p_threshold = 0.05  


bp_output_file = "figure6E.tif"


colnames(BP)[colnames(BP) == "X."] <- "Ratio"


BP = separate(BP,Term, sep="~",into=c("ID","Description"))


BP <- BP[BP$PValue < p_threshold, ]


wrap_width <- 47  
# BP
if (nrow(BP) > 0) {
  BP$Description <- str_wrap(BP$Description, width = wrap_width)
  BP$order <- factor(BP$Description, levels = rev(BP$Description), ordered = TRUE)
}


if (nrow(BP) > 0) {
  bp_plot <- ggplot(BP,
                    aes(x = Ratio, y = order, color = PValue)) +
    geom_point(aes(size = Count)) +
    labs(size="Counts",x="Gene_Ratio(%)",y="",title="GO_BP",color = expression(bolditalic(P))) +  
    scale_color_gradient(low="red",high ="blue") +   
    theme(plot.title = element_text(family = "Arial", size = 18,
                                    color = "black",
                                    hjust = 0.5),   
          legend.title = element_text(family = "Arial", size = 15,
                                      color = "black",
                                      face = "bold"),  
          legend.text = element_text(family = "Arial", size = 12, colour = "black"),  
          axis.title.x = element_text(family = "Arial", size = 15,
                                      color = "black",
                                      face = "bold",
                                      margin = margin(t = 5)  
          ),
          axis.text.x = element_text(family = "Arial", size = 15,   
                                     color = "black",
                                     face = "bold",  
                                     margin = margin(t = 5)   
          ),  
          axis.title.y = element_text(family = "Arial", size = 15,
                                      color = "black",
                                      face = "bold",
                                      margin = margin(r = 5)  
          ),
          axis.text.y = element_text(family = "Arial", size = 13,
                                     color = "black",
                                     face = "bold",
                                     margin = margin(r = 5)  
          ),
          plot.margin = margin(t = 10, r = 20, b = 10, l = 20) 
    ) +
    guides(
      size = guide_legend(order = 1),
      color = guide_colorbar(order = 2)
    )   
}


if (nrow(BP) > 0) {ggsave(bp_output_file, plot = bp_plot, width = 8.1, height = 5, dpi = 300)}