library(data.table)
library(ggplot2)
library(Rtsne)
library(pbapply)

bca.type <- 'vanilla'

# Load in the data
vectors <- fread(paste0('data/stcn.reverse.',bca.type,'.amsgrad.50.vectors.tsv'), sep = '\t')
keys <- fread(paste0('data/stcn.reverse.',bca.type,'.amsgrad.50.dict.tsv'), sep = '\t', quote = "")
metadata <- data.table::fread("tr -d \'\"\' < data/stcn_labels.tsv")
#metadata <- fread('data/stcn_labels.tsv', header = T, sep = '\t', quote="")
colnames(metadata) <- c('URI', 'label')

# Only keep the records for URI's
uris <- keys$V2 == 0
vectors <- vectors[uris]
keys <- keys[uris]$V1

# Many URI's are pointing to outside sources, remove them
uris <- grepl('http://data.bibliotheken.nl/id/nbt/', keys)
vectors <- vectors[uris]
keys <- keys[uris]

# Clean up
rm(uris)

labels <- data.table(keys)
colnames(labels) <- c("key")
labels$label <- pbapply::pbsapply(keys, function(key) {
  idx <- which(key == metadata$URI)[1]
  if(is.na(idx))
    return('unknown')
  return(metadata[idx]$label)
})

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

fwrite(labels, file = 'output/stcn.metadata.tsv', sep = "\t", row.names = F)
fwrite(data.table(vectors.pc), file = "output/stcn.pca.tsv", sep = "\t", col.names = F, row.names = F, quote = F)
