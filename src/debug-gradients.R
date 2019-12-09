library(data.table)
library(ggplot2)
library(reshape2)

bias <- runif(100, -1e-3, 1e-3) + runif(100, -1e-3, 1e-3)

# GloVe partial derivative for u, assume xmax = 1 and alpha = 3/4
glove.u <- function(u, v, x) {
  if(x == 0) return(rep(0, length(u)))
  x^(3/4) * (bias + drop(u %*% v) - log(x)) * v
}
# GloVe partial derivative for v, assume xmax = 1 and alpha = 3/4
glove.v <- function(u, v, x) {
  if(x == 0) return(rep(0, length(u)))
  x^(3/4) * (bias + drop(u %*% v) - log(x)) * u
}
# pGloVe partial derivative for u
pglove.u <- function(u, v, x) {
  if(x == 0) return(rep(0, length(u)))
  if(x == 1) return(rep(NA, length(u)))
  x * (bias + drop(u %*% v) - log(x / (1 - x)) ) * v
}
# pGloVe partial derivative for v
pglove.v <- function(u, v, x) {
  if(x == 0) return(rep(0, length(u)))
  if(x == 1) return(rep(NA, length(u)))
  x * (bias + drop(u %*% v) - log(x / (1 - x)) ) * u
}

# Create vectors u and v around the origin
u <- rep(0,100)#rnorm(100, 0, .5)
v <- rep(0,100)#rnorm(100, 0, .1)
# How many x values are we using, will increase resolution in the final plots
resolution <- 1e-2
# Create some probability input in a range  (0, 1) (exclusive on both)
x <- seq(0, 1, resolution)

# Calculate the gradients for a range of possible x values
grad.glove.u <- melt(t(sapply(x, glove.u, u = u, v = v)), varnames = c('x','u'))
grad.pglove.u <- melt(t(sapply(x, pglove.u, u = u, v = v)), varnames = c('x','u'))
grad.glove.v <- melt(t(sapply(x, glove.v, u = u, v = v)), varnames = c('x','v'))
grad.pglove.v <- melt(t(sapply(x, pglove.v, u = u, v = v)), varnames = c('x','v'))

grad.glove.u$x <- x[grad.glove.u$x]
grad.pglove.u$x <- x[grad.pglove.u$x]
grad.glove.v$x <- x[grad.glove.v$x]
grad.pglove.v$x <- x[grad.pglove.v$x]

# Plot the outcomes for all 4 possibilities
ggplot(grad.glove.u) + 
  geom_tile(aes(x=x, y=u, fill=value)) +
  scale_fill_gradient2(low ='red', high = 'green', mid = 'white') +
  theme_classic() + xlab('x') + ylab('u') + ggtitle('GloVe delta u')
ggplot(grad.pglove.u) + 
  geom_tile(aes(x=x, y=u, fill=value)) +
  scale_fill_gradient2(low ='red', high = 'green', mid = 'white') +
  theme_classic() + xlab('x') + ylab('u') + ggtitle('pGloVe delta u')
ggplot(grad.glove.v) + 
  geom_tile(aes(x=x, y=v, fill=value)) +
  scale_fill_gradient2(low ='red', high = 'green', mid = 'white') +
  theme_classic() + xlab('x') + ylab('v') + ggtitle('GloVe delta v')
ggplot(grad.pglove.v) + 
  geom_tile(aes(x=x, y=v, fill=value)) +
  scale_fill_gradient2(low ='red', high = 'green', mid = 'white') +
  theme_classic() + xlab('x') + ylab('v') + ggtitle('pGloVe delta v')
