library(data.table)
library(ggplot2)
library(grid)
library(gridExtra)

grid_arrange_shared_legend <- function(...) {
  plots <- list(...)
  g <- ggplotGrob(plots[[1]] + theme(legend.position="bottom"))$grobs
  legend <- g[[which(sapply(g, function(x) x$name) == "guide-box")]]
  lheight <- sum(legend$height)
  grid.arrange(
    do.call(arrangeGrob, lapply(plots, function(x)
      x + theme(legend.position="none"))),
    legend,
    ncol = 1,
    heights = unit.c(unit(1, "npc") - lheight, lheight))
}

adadelta <- fread(file='data/onstage/adadelta.txt')$V1
adam <- fread(file='data/onstage/adam.txt')$V1
adagrad <- fread(file='data/onstage/adagrad.txt')$V1
amsgrad <- fread(file='data/onstage/amsgrad.txt')$V1

length(adadelta) <- max(length(adadelta), length(adam), length(amsgrad), length(adagrad))
length(adam) <- max(length(adadelta), length(adam), length(amsgrad), length(adagrad))
length(adagrad) <- max(length(adadelta), length(adam), length(amsgrad), length(adagrad))
length(amsgrad) <- max(length(adadelta), length(adam), length(amsgrad), length(adagrad))

onstage <- data.table(adadelta, adam, adagrad, amsgrad)
onstage.melt <- melt(onstage, variable.name = 'method', value.name = 'loss')
onstage.melt$iteration <- 1:nrow(onstage)

adadelta <- fread(file='data/saa/adadelta.txt')$V1
adam <- fread(file='data/saa/adam.txt')$V1
adagrad <- fread(file='data/saa/adagrad.txt')$V1
amsgrad <- fread(file='data/saa/amsgrad.txt')$V1

length(adadelta) <- max(length(adadelta), length(adam), length(amsgrad), length(adagrad))
length(adam) <- max(length(adadelta), length(adam), length(amsgrad), length(adagrad))
length(adagrad) <- max(length(adadelta), length(adam), length(amsgrad), length(adagrad))
length(amsgrad) <- max(length(adadelta), length(adam), length(amsgrad), length(adagrad))

saa <- data.table(adadelta, adam, adagrad, amsgrad)
saa.melt <- melt(saa, variable.name = 'method', value.name = 'loss')
saa.melt$iteration <- 1:nrow(saa)

onstage.plot <- ggplot(data=onstage.melt) +
  geom_line(aes(x=iteration, y=loss, color=method), size = 1) +
  ggtitle('Onstage') +
  ylim(c(0,0.05)) +
  ylab('loss') +
  theme_bw() +
  theme(axis.title=element_text(size=14), axis.text=element_text(size=12), legend.text=element_text(size=16), legend.title = element_blank(), legend.key.size = unit(2, 'lines'))

saa.plot <- ggplot(data=saa.melt) +
  geom_line(aes(x=iteration, y=loss, color=method), size = 1) +
  ggtitle('Amsterdam City Archives') +
  ylim(c(0,0.05)) +
  ylab('loss') +
  theme_bw() + 
  theme(axis.title=element_text(size=14), axis.text=element_text(size=12))

grid_arrange_shared_legend(onstage.plot, saa.plot, nrow = 1)
