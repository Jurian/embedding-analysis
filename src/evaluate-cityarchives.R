library(data.table)
library(ggplot2)
library(pbapply)

rm(list = ls())

createFileName <- function(file, reverse, bca.type, glove.type, dimensions) {
  reverse <- if(reverse) 'reverse' else ''
  paste(file,reverse,'1.0E-4_1.0E-8',bca.type,glove.type,dimensions, sep = '.')
}

filename <- createFileName('saa', T, 'directed_weighted', 'amsgrad', 200)

# Load in the data
vectors <- fread(paste0('../graph-embeddings/out/', filename, '.vectors.tsv'), sep = "\t")
keys <- fread(paste0('../graph-embeddings/out/',filename,'.dict.tsv'), sep = '\t', quote = "")

# Only keep the records for URI's
uris <- keys$type == 0
vectors <- vectors[uris]
keys <- keys[uris]$key

# Only keep the records for persons
uris <- 
  grepl('ecartico/marriage', keys) | 
  grepl('record/IndexOpDoopregister', keys) | 
  grepl('record/IndexOpOndertrouwregister', keys) | 
  grepl('record/IndexOpBegraafregistersVoor1811', keys)
vectors <- vectors[!uris]
keys <- keys[!uris]

colnames(vectors) <-  paste0('d', 1:ncol(vectors))

# Clean up
rm(uris)

# Load in ground truth cluster data
clusters <- fread('data/saa_clusters.tsv', header = T, sep = '\t', blank.lines.skip = T)
clusters <- clusters[!is.na(clusters$ILN_GROUP)]
clusters$CLUSTER_ID <- as.integer(as.factor(clusters$CLUSTER_ID))
clusters$CLUSTER_ID <- as.integer(paste0(clusters$CLUSTER_ID, clusters$ILN_GROUP))
clusters$RESOURCE_URI <- paste0(clusters$DATASET, clusters$RESOURCE_URI)
# Clear all unused columns
clusters$STATUS <- NULL
clusters$CLUSTER_COUNT <- NULL
clusters$DATASET <- NULL
clusters$CLUSTER_SIZE <- NULL
clusters$ILN_GROUP <- NULL

# Match cluster ID with vector index based on URI
cluster <- pbsapply(keys, function(key) {
  idx <- which(key == clusters$RESOURCE_URI)[1]
  if(is.na(idx))
    return(NA)
  return(clusters[idx]$CLUSTER_ID)
})


has_cluster <- !is.na(cluster)

cluster <- cluster[has_cluster]
vectors <- vectors[has_cluster]

rm(clusters, has_cluster)


vectors.dist <- data.table(as.matrix(dist(vectors, method = 'canberra')))

R.precision <- pbapply(vectors.dist, 1, function(distance) {
  
  vector.order <- cluster[base::order(distance)]

  cluster <- (vector.order == vector.order[1])
  cluster.idx <- which(cluster)
  cluster.size <- length(cluster.idx)
  
  # Nr of relevant documents R
  R <- cluster.size
  # Nr of results in top-R
  r <- sum(cluster.idx <= R)
  # R-precision
  r / R
})

sum(R.precision == 1 ) / nrow(vectors.dist)
hist(R.precision, breaks = 100)
 
ndcg <-  pbapply(vectors.dist, 1, function(distance) {
  
  vector.order <- cluster[base::order(distance)]
  
  # Discounted cumultive gain
  rel <- as.integer(vector.order == vector.order[1])
  p <- length(rel)
  dcg <- rel[1] + sum(rel[-1] / log2((2:p)+1))
  
  # Ideal discounted cumulutive gain
  p <- sum(rel == 1)
  rel <- rel[rel == 1]
  idcg <- sum( (2^rel - 1) /  log2( (1:p)+1 )  )
  
  # Normalized discounted cumulative gain
  dcg / idcg
  
})

sum(ndcg == 1 ) / nrow(vectors.dist)
hist(ndcg, breaks = 100, xlab = 'Normalized Discounted Cumulative Gain')

# We test using a subset
# First order by the cluster IDs so the first 100 rows contain clusters with all members present
order.cluster <- order(cluster)
labels.subset <- cluster[order.cluster][1:150]
vectors.subset <- vectors[order.cluster][1:150]

plot(hclust(dist(vectors.subset, method = 'canberra'), method = 'ward.D2'), labels = labels.subset)