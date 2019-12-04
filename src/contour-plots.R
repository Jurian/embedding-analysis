library(data.table)
library(ggplot2)

glove <- function(x,p) {
  x^(3/4) * (p - log(x) )^2
}

pglove <- function(x,p) {
  x * (p - log(x/(1-x)) )^2
}

x.resolution <- 0.01;
x <- seq(from = x.resolution, to = 1-x.resolution, by = x.resolution )
p <- seq(from = -10, to = 10, length.out = length(x))

grid <- data.table(expand.grid(x,p))
colnames(grid) <- c('x', 'p')

grid$j <- apply(grid, 1, function(t){glove(t[1],t[2])})
ggplot(grid, aes(x=x,y=p)) + geom_raster(aes(fill=j)) + geom_contour(aes(z=j), bins = 20, color = 'white') + scale_fill_gradientn(colours=c('blue','red'))


grid$j <- apply(grid, 1, function(t){pglove(t[1],t[2])})
ggplot(grid, aes(x=x,y=p)) + geom_raster(aes(fill=j)) + geom_contour(aes(z=j), bins = 20, color = 'white') + scale_fill_gradientn(colours=c('blue','red'))