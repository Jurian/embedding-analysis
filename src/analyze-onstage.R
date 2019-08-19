library(data.table)
library(ggplot2)
library(Rtsne)
library(pbapply)
library(mclust)

createFileName <- function(file, reverse, bca.type, glove.type, dimensions) {
  reverse <- if(reverse) 'reverse' else ''
  paste(file,reverse,'0.1_1.0E-4',bca.type,glove.type,dimensions, sep = '.')
}

filename <- createFileName('onstage', T, 'directed_weighted_literal', 'amsgrad', 200)

# Load in the data
vectors <- fread(paste0('../graph-embeddings/out/', filename, '.vectors.tsv'), sep = "\t")
keys <- fread(paste0('../graph-embeddings/out/',filename,'.dict.tsv'), sep = '\t', quote = "")
metadata <- fread('data/onstage_play_metadata.csv', header = F, sep = '\t')
colnames(metadata) <- c('URI', 'lang', 'label')

# Only keep the records for URI's
uris <- keys$type == 0
vectors <- vectors[uris]
keys <- keys[uris]$key

#keys <- keys$key

# Remove shows
uris <- grepl('/shows/', keys)
vectors <- vectors[!uris]
keys <- keys[!uris] 
# Remove persons
uris <- grepl('/persons/', keys)
vectors <- vectors[!uris]
keys <- keys[!uris]

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

labels$lang <- pbapply::pbsapply(keys, function(key) {
  idx <- which(key == metadata$URI)[1]
  if(is.na(idx))
    return('unknown')
  return(metadata[idx]$lang)
})



#clust <- Mclust(vectors, G = length(unique(labels$lang)))
#train <- sample(1:nrow(vectors), 1500)
clust <- MclustDA(vectors, as.factor(labels$lang), modelType = "EDDA")
labels$cluster <- predict(clust, vectors)$classification

fwrite(labels, file = paste0('output/', filename,'.metadata.tsv'), sep = "\t", row.names = F)
fwrite(vectors, file = paste0('output/',filename,'.tsv'), sep = "\t", col.names = F, row.names = F, quote = F)

#clust <- hclust(dist(vectors), method = "complete")
#clust.cut <- cutree(clust, h = .5)
#labels$cut <- clust.cut
#barplot(table(labels$cut))

