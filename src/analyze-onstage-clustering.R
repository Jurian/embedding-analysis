library(data.table)
library(ggplot2)
library(fpc)
library(pvclust)

bca.type <- 'vanilla'

# Load in the data
vectors <- fread(paste0('data/onstage.nt.',bca.type,'.amsgrad.50.vectors.txt'))
keys <- fread(paste0('data/onstage.nt.',bca.type,'.amsgrad.50.dict.txt'))

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


# Use principal component analysis to reduce the number of dimensions and speed up clustering
# while (hopefully) keeping clusters intact
pca <- prcomp(vectors)

# We are fine with using the principal components that explain min.var fraction of the variance
min.var <- 0.9
# Calculate variance
pca.var <- pca$sdev^2
# Take the cumulutive sum of proportional variance
pca.cum.var <- cumsum(pca.var/sum(pca.var))
# Find the first component that meets the min.var mark
pca.min <- min(which(pca.cum.var >= min.var))

# Transform our original vectors to pca space, taking only the components necessary to reach min.var
vectors.pc <- predict(pca, vectors)[,1:pca.min]

# Clean up some more
rm(pca.cum.var, pca.var, min.var)


k <- 8

# Perform clustering around mediods
pam <- fpc::pamk(vectors.pc, krange=(k-4):(k+4))

# Write clusters to disk
clusters.pam <- as.factor(pam$pamobject$clustering)
for(c in levels(clusters.pam)) {
  fwrite(as.list(keys[clusters.pam == c]), file = paste0('output/', 'onstage-cluster-pam-',c,'.txt'), sep = '\n', col.names = F, row.names = F)
}

# Ward Hierarchical Clustering with Bootstrapped p values
ward <- pvclust(vectors.pc, method.hclust="ward.D2", method.dist='euclidean', parallel = T)

# Write clusters to disk
clusters.ward <- as.factor(cutree(fit$hclust, k = k))
for(c in levels(clusters.ward)) {
  fwrite(as.list(keys[clusters.ward == c]), file = paste0('output/', 'onstage-cluster-ward-',c,'.txt'), sep = '\n', col.names = F, row.names = F)
}

fwrite(data.table(vectors.pc), file = "output/onstage.pca.tsv", sep = "\t", col.names = F, row.names = F, quote = F)
fwrite(data.table(keys, pam = as.factor(pam$pamobject$clustering)), file = 'output/onstage.pca.labels.tsv', sep = "\t", row.names = F)

