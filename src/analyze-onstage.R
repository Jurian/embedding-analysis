library(data.table)
library(ggplot2)
library(Rtsne)

bca.type <- 'vanilla'

# Load in the data
vectors <- fread(paste0('data/onstage.new.reverse.',bca.type,'.amsgrad.200.vectors.tsv'), sep = "\t")
keys <- fread(paste0('data/onstage.new.reverse.',bca.type,'.amsgrad.200.dict.tsv'), sep = "\t", quote = "")
metadata <- fread('data/onstage_labels.tsv', header = T, sep = "\t")

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

labels <- data.table(keys, labels)
colnames(labels) <- c("key","type")
labels$label <- sapply(keys, function(key) {
  idx <- which(key == metadata$URI)[1]
  if(is.na(idx))
    return('unknown')
  return(metadata[idx]$label)
})
fwrite(labels, file = 'output/onstage.new.metadata.tsv', sep = "\t", row.names = F)
fwrite(vectors, file = paste0('output/onstage.new.',bca.type,'.tsv'), sep = "\t", col.names = F, row.names = F, quote = F)

# Use principal component analysis to reduce the number of dimensions
pca <- prcomp(vectors)

# We are fine with using the principal components that explain min.var of the variance
min.var <- 0.9
# Calculate variance
pca.var <- pca$sdev^2
# Take the cumulutive sum of proportional variance
pca.cum.var <- cumsum(pca.var/sum(pca.var))

# For fun, let's view a scree-like plot of this pca object
plot(pca.var/sum(pca.var), 
     xlab = 'Principal Component',
     ylab = 'Proportion of Variance Explained',
     type = 'b')

# Find the first component that meets the min.var mark
pca.min <- min(which(pca.cum.var >= min.var))

# Transform our original vectors to pca space, taking only the components necessary to reach min.var
vectors.pc <- predict(pca, vectors)[,1:pca.min]

# Clean up some more
rm(pca.cum.var, pca.var, min.var)

# To visualize our data, perform tsne. 
# We set pca to false because we already did that ourselves
# and num_threads to 0, which means use all available cores
vis <- Rtsne(X = vectors.pc, dims = 2, pca = F, num_threads = 0)

# Plot the result
ggplot(data.table(vis$Y)) +
  geom_point(aes(x = V1, y = V2, col = labels))

# Find 10 example shows from each cluster, using the plot as a guide
#shows.1 <- keys[sample(which(vis$Y[,1] > -10 & vis$Y[,1] < 10  & vis$Y[,2] > 0   & vis$Y[,2] < 20 ), 10)]
#shows.2 <- keys[sample(which(vis$Y[,1] > 0   & vis$Y[,1] < 20  & vis$Y[,2] > -30 & vis$Y[,2] < -10), 10)]
#shows.3 <- keys[sample(which(vis$Y[,1] > -20 & vis$Y[,1] < -10 & vis$Y[,2] > -30 & vis$Y[,2] < -20), 10)]

# Find 10 example persons from each cluster, using the plot as a guide
#persons.1 <- keys[sample(which(vis$Y[,1] > -35 & vis$Y[,1] < -25 & vis$Y[,2] > -10 & vis$Y[,2] < 0 ), 10)]
#persons.2 <- keys[sample(which(vis$Y[,1] > -20 & vis$Y[,1] < -10 & vis$Y[,2] >  33 & vis$Y[,2] < 40), 10)]
#persons.3 <- keys[sample(which(vis$Y[,1] > -30 & vis$Y[,1] < -23 & vis$Y[,2] >  22 & vis$Y[,2] < 26), 10)]

# Which persons appear in the largest shows cluster?
#keys.persons <- keys[which(vis$Y[,1] > -10 & vis$Y[,1] < 10  & vis$Y[,2] > 0   & vis$Y[,2] < 20 )]
#keys.persons <- keys.persons[grepl('/persons/', keys.persons)]
