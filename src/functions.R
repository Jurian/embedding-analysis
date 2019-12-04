
#' Create a file name from embedding options
#' @param file The file name without options
#' @param glove.type GloVe or pGloVe
#' @param reverse Was the graph walker also run in reverse
#' @param bca.type The graph walk strategy used
#' @param gd.type The gradient descent algorithm used
#' @param dimension The number of dimensions of the embedding
#' @return A vector with the values from M
createFileName <- function(file, glove.type, reverse, bca.type, gd.type, dimensions) {
  reverse <- if(reverse) 'reverse' else ''
  paste(file, glove.type, reverse,'0.1_1.0E-8', bca.type, gd.type, dimensions, sep = '.')
}

#' Calculate the normalized discounted cumulative gain for each vector that has a cluster label
#' based on the distance ranking of all other vectors
#' @param vectors A matrix containg the vectors
#' @param cluster A vector with cluster labels
#' @return A vector containing the ndcg values
ndcg <- function(vectors, cluster) {
  
  vectors <- as.matrix(vectors)
  
  pbsapply(which(cluster != -1), function(i) {
    
    distance <- m2v(euclideanDist(i-1, vectors))
    vector.order <- cluster[m2v(Rfast::Order(distance))]
    
    #vector.order <- cluster[base::order(distance)]
    
    # Discounted cumultive gain
    rel <- as.integer(vector.order == vector.order[1])
    p <- length(rel)
    dcg <- sum( (2^rel - 1) / log2((1:p) + 1) )
    
    # Ideal discounted cumulutive gain
    rel <- rel[rel == 1]
    p <- sum(rel)
    idcg <- sum( (2^rel - 1) / log2((1:p) + 1) )
    
    # Normalized discounted cumulative gain
    dcg / idcg
    
  })
}

#' Convert a matrix to a vector
#' Useful because rcpp outputs a matrix by default
#' @param M The matrix to convert
#' @return A vector with the values from M
m2v <- function(M) {
  dim(M) <- NULL
  return(M)
}

#' Generate a train set for the a classifier by taking pairs of vectors and calculating the hadamard product
#' All matching pairs are used and additional pairs are chosen based on closest distance, plus some random pairs
#' @param vectors A matrix containg the vectors
#' @param cluster A vector with cluster labels
#' @param negative.samples The number of negative samples to take
#' @return The hadamard product of each pair that was considered
generate.train.set <- function(vectors, cluster, negative.samples) {
  
  #vectors <- vectors.pglove
  #vectors.dist <- vectors.pglove.dist
  #cluster <- clusters$CLUSTER_ID
  #negative.samples <- mean.cluster.size*2
  
  vectors <- t(as.matrix(vectors))
  
  # We can only use labeled vectors
  labeled <- which(cluster != -1)
  
  hadamard <- pblapply(labeled, function(i) {
    
    # Find the indexes of other vectors with the same cluster ID
    o <- which(cluster == cluster[i])
    
    # Find the indexes of the k closest points based on how many points are in this cluster
    # Make sure positive examples in o are included if they are not close by
    k <- (length(o)-1) * 2
    
    c <- unique(c(o, m2v(kSmallestFast(vectors, i-1, k + 1))))
    
    # Avoid comparing a vector with itself
    c <- c[-which(c == i)]
    
    # Pad the remaining with random negative samples
    p <- (length(o)*3) - length(c)
    
    # Also add random pairs for a more generic solution
    set.seed(i)
    c <- c(c, sample(ncol(vectors), p + negative.samples))
    
    # Calculate Hadamard product for both in- and out-cluster pairs
    y <- hadamardVM(vectors[,i], vectors[,c])
    
    # Add labels
    y <- cbind(y, as.integer(cluster[c] == cluster[i]))
    
    return(y)
    
  })
  
  hadamard <- do.call('rbind', hadamard)
  
  
  # Update column names
  colnames(hadamard) <- paste0('d', 1:ncol(hadamard))
  colnames(hadamard)[ncol(hadamard)] <- 'y'
  
  return(hadamard)
}

#' Generate a sparse relational matrix R where element i,j = 1 when the supplied classifier outputs 1 for hadamard(i,j)
#' @param vectors A matrix containg the vectors
#' @param classifier The classifier used to consider pairs of points
#' @param k Classify the k closest number of vectors
#' @return A sparse relational matrix R
generate.relational.matrix <- function(vectors, classifier, k) {
  
  vectors <- as.matrix(vectors)

  R <- pblapply(1:nrow(vectors), function(i) {
    
    neighbors <- m2v(kSmallestFast(vectors, i-1, k))
    #neighbors <- m2v(Ksmallest(m2v(euclideanDist(i-1, vectors)), k + 1))
    neighbors <- neighbors[-which(neighbors == i)]
    
    hadamard <- hadamardVM(vectors[i,], vectors[neighbors,])
    prediction <- predict(classifier, hadamard)
    
    # We don't use a logic sparse vector because rcpparmadillo does not support those
    return(Matrix::sparseVector(x = 1, i = neighbors[prediction == 'yes'], length = nrow(vectors)))
  })
  
  R <- sameSizeVectorList2Matrix(R)
  R <- makeReflexive(R)
  R <- makeSymmetric(R)
  R <- makeTransitive(R)
  
  return(R)
}


sameSizeVectorList2Matrix <- function(vectorList){  
  sm_i<-NULL
  sm_j<-NULL
  sm_x<-NULL
  for (k in 1:length(vectorList)) {
    sm_i <- c(sm_i,rep(k,length(vectorList[[k]]@i)))
    sm_j <- c(sm_j,vectorList[[k]]@i)
    sm_x <- c(sm_x,vectorList[[k]]@x)
  }
  return (sparseMatrix(i=sm_i,j=sm_j,x=sm_x,dims=c(length(vectorList),vectorList[[1]]@length)))
}