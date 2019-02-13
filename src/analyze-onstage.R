library(data.table)
library(ggplot2)
library(Rtsne)
library(mclust)

# Load in the data
vectors <- fread('data/onstage.AMSGrad.50.vectors.txt')
keys <- fread('data/onstage.AMSGrad.50.dict.txt')

# Only keep the records for URI's
uris <- keys$V2 == 0
vectors <- vectors[uris]
keys <- keys[uris]$V1

# Many URI's are pointing to outside sources, remove them
uris <- grepl('www.vondel.humanities.uva.nl', keys)
vectors <- vectors[uris]
keys <- keys[uris]

# Clean up
rm(uris)

# Use principal component analysis to reduce the number of dimensions
pca <- prcomp(vectors)

# We are fine with using the principal components that explain 90% of the variance
# Calculate variance
pca.var <- pca$sdev^2
# Take the cumulutive sum of proportional variance
pca.cum.var <- cumsum(pca.var/sum(pca.var))

# For fun, let's view a scree plot of this pca object
plot(pca.var/sum(pca.var), 
     xlab = 'Principal Component',
     ylab = 'Proportion of Variance Explained',
     type = 'b',
     ylim = c(0,1))

# Find the first component that meets the 90% mark
pca.min <- min(which(pca.cum.var >= 0.9))

# Transform our original vectors to pca space, taking only the components necessary to reach 90%
vectors.pc <- predict(pca, vectors)[,1:pca.min]

# Clean up some more
rm(pca.cum.var, pca.var)

# To visualize our data, perform tsne. 
# We set pca to false because we already did that ourselves
# and num_threads to 0, which means use all available cores
vis <- Rtsne(X = vectors.pc, dims = 2, pca = F, num_threads = 0)

# Label the keys to make sure the clusters we find are meaningful
labels <- data.table(
  shows = grepl('/shows/', keys),
  plays = grepl('/plays', keys),
  persons = grepl('/persons/', keys)
)
# Mark all other keys as 'other'
labels$other <- apply(labels, 1, function(x){!any(x)})
# Create a vector with character labels
labels <- apply(labels, 1, function(x){
  colnames(labels)[which(x)]
})
# We only have a few labels, so turn into factor
labels <- as.factor(labels)

# Plot the result
ggplot(data.table(vis$Y)) +
  geom_point(aes(x = V1, y = V2, col = labels))

