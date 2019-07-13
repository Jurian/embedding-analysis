library(data.table)
library(ggplot2)
library(pbapply)

createFileName <- function(file, reverse, bca.type, glove.type, dimensions) {
  reverse <- if(reverse) 'reverse' else ''
  paste(file,reverse,'0.1_1.0E-4',bca.type,glove.type,dimensions, sep = '.')
}

filename <- createFileName('saa', T, 'undirected_weighted', 'amsgrad', 200)

# Load in the data
vectors <- fread(paste0('../graph-embeddings/out/', filename, '.vectors.tsv'), sep = "\t")
keys <- fread(paste0('../graph-embeddings/out/',filename,'.dict.tsv'), sep = '\t', quote = "")
metadata <- fread('data/saa_labels.tsv', header = T, sep = '\t')

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

# Add a column to see from which dataset the record came
labels <- data.table(
  ecartico = grepl('ecartico/persons', keys),
  baptism = grepl('person/IndexOpDoopregister', keys),
  marriage =  grepl('person/IndexOpOndertrouwregister', keys),
  burial = grepl('person/IndexOpBegraafregistersVoor1811', keys)
)
# Mark all other keys as 'other'
labels$other <- apply(labels, 1, function(x){!any(x)})
# Create a vector with character labels?
labels <- apply(labels, 1, function(x){
  colnames(labels)[which(x)]
})
# We only have a few labels, so turn into factor
labels <- as.factor(labels)

labels <- data.table(keys, labels)
colnames(labels) <- c("key","dataset")
# Give each URI a label, if available
labels$label <- pbapply::pbsapply(keys, function(key) {
  idx <- which(key == metadata$person)[1]
  if(is.na(idx))
    return('unknown')
  metadata[idx]$name
})
# Give each URI a cluster ID, if available 
labels$cluster <- pbapply::pbsapply(keys, function(key) {
  idx <- which(key == clusters$RESOURCE_URI)[1]
  if(is.na(idx))
    return(NA)
  clusters[idx]$CLUSTER_ID
})

# Clean up
rm(keys, metadata, clusters)
# Remove records with no cluster ID
has_cluster <- !is.na(labels$cluster)

labels <- labels[has_cluster]
vectors <- vectors[has_cluster]

fwrite(labels, file = paste0('output/', filename, '.metadata.tsv'), sep = "\t", row.names = F)
fwrite(vectors, file = paste0('output/', filename, '.tsv'), sep = "\t", col.names = F, row.names = F, quote = F)

rm(filename, has_cluster)


vectors.dist <- data.table(as.matrix(dist(vectors, method = 'canberra')))
R.precision <- pbapply::pbapply(vectors.dist, 1, function(distance) {
  
  vector.order <- labels[base::order(distance)]$cluster
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
hist(R.precision, breaks = length(unique(R.precision)))

ndcg <- pbapply::pbapply(vectors.dist, 1, function(distance) {
  
  # Order all cluster labels by distance
  cluster.ordered <- labels[base::order(distance)]$cluster
  
  # Discounted cumultive gain
  rel <- as.integer(cluster.ordered == cluster.ordered[1])
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
hist(ndcg, breaks = length(unique(R.precision))*10)

normalize <- function(vec) {
  vec / sqrt(sum(vec^2))
}

# We test using a subset
# First order by the cluster IDs so the first 100 rows contain clusters with all members present
order.cluster <- order(labels$cluster)
labels.subset <- labels[order.cluster][1:150]
vectors.subset <- vectors[order.cluster][1:150]


# The two closest points from each cluster
plot(hclust(dist(vectors.subset), method = 'single'), labels = labels.subset$cluster)

# The two closest points from each cluster
plot(hclust(dist(vectors.subset), method = 'complete'), labels = labels.subset$cluster)

# The inter-cluster mid-point
plot(hclust(dist(vectors.subset), method = 'centroid'), labels = labels.subset$cluster)

# The inter-cluster median point
plot(hclust(dist(vectors.subset), method = 'median'), labels = labels.subset$cluster)

# The average of the cluster’s distances is taken whilst compensating for the number of points in that cluster
# (UPGMA, Unweighted Pair Group Method with Arithmetic Mean)
plot(hclust(dist(vectors.subset), method = 'average'), labels = labels.subset$cluster)

# The average of the cluster’s distances is taken, not considering the number of points in that cluster
# (WPGMA, Weighted Pair Group Method with Arithmetic Mean)
plot(hclust(dist(vectors.subset), method = 'mcquitty'), labels = labels.subset$cluster)

# Calculates the increase in the error sum of squares (ESS) after fusing two clusters
# Successive clustering steps chosen so as to minimize the increase in ESS
plot(hclust(dist(vectors.subset, method = 'euclidean'), method = 'ward.D2'), labels = labels.subset$cluster)
plot(hclust(dist(vectors.subset, method = 'manhattan'), method = 'ward.D2'), labels = labels.subset$cluster)
plot(hclust(dist(vectors.subset, method = 'canberra'), method = 'ward.D2'), labels = labels.subset$cluster)
