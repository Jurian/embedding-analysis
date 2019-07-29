library(data.table)
library(ggplot2)
library(Rtsne)
library(pbapply)


createFileName <- function(file, reverse, bca.type, glove.type, dimensions) {
  reverse <- if(reverse) 'reverse' else ''
  paste(file,reverse,bca.type,glove.type,dimensions, sep = '.')
}

createFileDir <- function(dir, filename) {
  paste0(dir, filename)
}


filename <- createFileName('onstage', T, 'semantic', 'amsgrad', 200)
inputFile <- createFileDir('../graph-embeddings/out/', filename)
outputFile <- createFileDir('output/', filename)

# Load in the data
vectors <- fread(paste0(inputFile, '.vectors.tsv'), sep = "\t")
keys <- fread(paste0(inputFile,'.dict.tsv'), sep = '\t', quote = "", header = T)
metadata <- fread('data/onstage_labels.tsv', header = T, sep = '\t')

# Only keep the records for URI's
uris <- keys$V2 == 0
vectors <- vectors[uris]
keys <- keys[uris]$V1

# Many URI's are pointing to outside sources, remove them
uris <- grepl('www.vondel.humanities.uva.nl/onstage/', keys)
vectors <- vectors[uris]
keys <- keys[uris]

# Clean up
rm(uris)

#vectors <- scale(vectors)

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
labels$label <- pbapply::pbsapply(keys, function(key) {
  idx <- which(key == metadata$URI)[1]
  if(is.na(idx))
    return('unknown')
  return(metadata[idx]$label)
})

# Use principal component analysis to reduce the number of dimensions
pca <- prcomp(vectors)

# We are fine with using the principal components that explain min.var of the variance
min.var <- 0.95
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

fwrite(labels, file = paste0(outputFile,'.metadata.tsv'), sep = "\t", row.names = F)
fwrite(data.table(vectors.pc), file = paste0(outputFile,'.pca.tsv'), sep = "\t", col.names = F, row.names = F, quote = F)
