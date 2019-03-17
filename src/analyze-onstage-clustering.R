library(data.table)
library(ggplot2)
library(mclust)

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


# Use principal component analysis to reduce the number of dimensions and speed up clustering
# while (hopefully) keeping clusters intact
pca <- prcomp(vectors)

# We are fine with using the principal components that explain min.var of the variance
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
rm(pca, pca.cum.var, pca.var, min.var, vectors, pca.min, bca.type)

shows.idx <- grepl('/shows/', keys)
plays.idx <- grepl('/plays', keys)
persons.idx <- grepl('/persons/', keys)

clst.shows <- mclust::Mclust(vectors.pc[shows.idx,], G = 1:16)
clst.plays <- mclust::Mclust(vectors.pc[plays.idx,], G = 1:24)
clst.persons <- mclust::Mclust(vectors.pc[persons.idx,], G = 1:24)

labels$cluster <- 1
labels$cluster[shows.idx] <- paste0('shows-cluster-', clst.shows$classification)
labels$cluster[plays.idx] <- paste0('plays-cluster-', clst.plays$classification)
labels$cluster[persons.idx] <- paste0('persons-cluster-', clst.persons$classification)
rm(shows.idx, plays.idx, persons.idx)

clst.combined <- mclust::Mclust(vectors.pc, G = 1:24)
labels$cluster2 <- clst.combined$classification

fwrite(data.table(vectors.pc), file = "output/onstage.pca.tsv", sep = "\t", col.names = F, row.names = F, quote = F)
fwrite(labels, file = 'output/onstage.pca.labels.tsv', sep = "\t", row.names = F)


shows <- labels[shows.idx]
shows$date <- as.Date(shows$label)
shows$date.numeric <- as.numeric(shows$label)

p <- ggplot(shows, aes(date, ..count..))
p <- p + geom_histogram(binwidth = 250, aes(color = cluster))



