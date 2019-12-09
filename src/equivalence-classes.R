is.trans <- function(M) {
  !any((M%*%M != 0) != (M != 0))
}

make.transitive <- function(M) {
  M <- M + ((M%*%M != 0) != (M != 0)) * 1
  return(M)
}

make.reflexive <- function(M) {
  diag(M) <- 1
  return(M)
}

make.symmetric <- function(M) {
  M[lower.tri(M)] <- t(M)[lower.tri(M)]
  return(M)
}

find.eq.class <- function(M) {
  
  captains <- 1:nrow(M)
  
  for(i in 1:nrow(M)) {
    subrow <- M[i,1:(i-1)]
    if(any(subrow != 0)) {
      # Demote this captain to existing equivalence class
      captains[i] <- which.max(subrow)
    }
  }
  
  # Make class labels sequential
  nr.of.classes <- length(unique(captains))
  unique.classes <- unique(captains)
  class.labels <- 1:nr.of.classes
  
  for(i in 1:nr.of.classes) {
    captains[which(captains == unique.classes[i])] <- class.labels[i]
  }
  
  return(as.factor(captains))
}

# This R is not transitive, there is no relation between 4 and 5, and 8 and 9
R <- Matrix(data = c(
  0,0,1,0,0,1,0,0,0, 
  0,0,0,1,1,0,0,0,0, 
  0,0,0,0,0,1,0,0,0, 
  0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0, 
  0,0,0,0,0,0,0,1,1,
  0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0), ncol = 9, byrow = T, sparse = T)

# Make sure R is reflexive
R <- makeReflexive(R)
# Make sure R is symmetric
R <- makeSymmetric(R)

