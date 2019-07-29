library(data.table)
library(ggplot2)
library(mclust)
library(pbapply)
library(scales)

createFileName <- function(file, reverse, bca.type, glove.type, dimensions) {
  reverse <- if(reverse) 'reverse' else ''
  paste(file,reverse,bca.type,glove.type,dimensions, sep = '.')
}

createFileDir <- function(dir, filename) {
  paste0(dir, filename)
}


filename <- createFileName('onstage', T, 'vanilla', 'amsgrad', 200)
inputFile <- createFileDir('../graph-embeddings/out/', filename)
outputFile <- createFileDir('output/', filename)

# Load in the data
vectors <- fread(paste0(inputFile, '.vectors.tsv'), sep = "\t")
keys <- fread(paste0(inputFile,'.dict.tsv'), sep = '\t', quote = "", header = F)
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


# Use principal component analysis to reduce the number of dimensions and speed up clustering
# while (hopefully) keeping clusters intact
pca <- prcomp(vectors)

# We are fine with using the principal components that explain min.var of the variance
min.var <- 0.95
# Calculate variance
pca.var <- pca$sdev^2
# Take the cumulutive sum of proportional variance
pca.cum.var <- cumsum(pca.var/sum(pca.var))

# Find the first component that meets the min.var mark
pca.min <- min(which(pca.cum.var >= min.var))

# Transform our original vectors to pca space, taking only the components necessary to reach min.var
vectors.pc <- predict(pca, vectors)[,1:pca.min]

# Clean up some more
rm(pca, pca.cum.var, pca.var, min.var, vectors, pca.min)

shows.idx <- grepl('/shows/', keys)
plays.idx <- grepl('/plays', keys)
persons.idx <- grepl('/persons/', keys)

clst.shows <- mclust::Mclust(vectors.pc[shows.idx,], G = 1:4)
clst.plays <- mclust::Mclust(vectors.pc[plays.idx,], G = 1:24)
clst.persons <- mclust::Mclust(vectors.pc[persons.idx,], G = 1:24)

labels$cluster <- 1
labels$cluster[shows.idx] <- paste0('shows-cluster-', clst.shows$classification)
labels$cluster[plays.idx] <- paste0('plays-cluster-', clst.plays$classification)
labels$cluster[persons.idx] <- paste0('persons-cluster-', clst.persons$classification)
rm(shows.idx, plays.idx, persons.idx)

fwrite(data.table(vectors.pc), file = "output/onstage.pca.tsv", sep = "\t", col.names = F, row.names = F, quote = F)
fwrite(labels, file = 'output/onstage.pca.labels.tsv', sep = "\t", row.names = F)

labels.shows <- labels[labels$type == 'shows']
labels.shows$type = NULL
labels.shows$cluster <- as.factor(labels.shows$cluster)
labels.shows$label <- as.Date(labels.shows$label)
labels.shows$date.num <- as.numeric(labels.shows$label)
labels.shows$year <- year(labels.shows$label)
labels.shows <- labels.shows[!is.na(labels.shows$date.num)]


bin <- diff(range(year(labels.shows$label))) # used for aggregating the data and aligning the labels

year.sums = table(factor(labels.shows$year, levels=min(labels.shows$year):max(labels.shows$year)))
year.range <- min(labels.shows$year):max(labels.shows$year)
year.per.cluster <- data.table(t(sapply(levels(labels.shows$cluster), function(c){
  x <- table(factor(labels.shows[labels.shows$cluster == c]$year, levels=year.range)) / year.sums
  x[is.nan(x)] <- 0
  return(x)
})))
year.per.cluster <- data.table(cluster = seq_len(nrow(year.per.cluster)), year.per.cluster)
year.per.cluster.melt <- melt(year.per.cluster, id.vars = 'cluster')
year.per.cluster.melt$cluster <- as.factor(year.per.cluster.melt$cluster)
colnames(year.per.cluster.melt) <- c('cluster', 'year', 'fraction')

ggplot(year.per.cluster.melt, aes(x = as.Date(ISOdate(year, 1, 1)), y = fraction, fill = cluster)) + 
  geom_bar(stat = 'identity', colour='black') +
  ylab('Fraction') +
  theme_bw() + 
  scale_x_date (
    name = 'Year',
    breaks = seq(min(labels.shows$label), max(labels.shows$label), bin*10),
    labels = date_format("%Y"),
    limits = c( as.Date(min(labels.shows$label), origin="1970-01-01"), 
                as.Date(max(labels.shows$label), origin="1970-01-01"))
    )

