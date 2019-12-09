#include <RcppArmadillo.h>
#include <progress.hpp>
#include <progress_bar.hpp>
#include "eta_progress_bar.hpp"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppProgress)]]

using namespace Rcpp;
using namespace arma;

//' Compute the hadamard product between matrices A and B
//' @param A A m x n matrix
//' @param B A m x n matrix
//' @return A m x n matrix
// [[Rcpp::export]]
mat hadamardMM(const mat& A, const mat& B) { 
  return(A % B);
}

//' Compute the hadamard product between vector a and matrix B
//' @param a A column vector of length n
//' @param B A m x n matrix 
//' @return A m x n matrix
// [[Rcpp::export]]
mat hadamardVM(const vec& a, const mat& B) {
  mat A(size(B));
  for(uword i = 0; i < B.n_rows; i++) {
    A.row(i) = a.t() % B.row(i);
  }
  return A;
}

//' Compute the Hadamard product between two column vectors
//' @param a First vector
//' @param b Second vector
//' @return Hadamard product
vec hadamardVV(const vec& a, const vec& b) {
  return a % b;
}

//' Force the sparse matrix M to be reflexive
//' @param M A sparse matrix
//' @return The modified sparse matrix M
// [[Rcpp::export]]
sp_mat makeReflexive(sp_mat& M) {
  M.diag().ones();
  return M;
}

//' Force the sparse matrix M to be symmetric
//' @param M A sparse matrix
//' @return The modified sparse matrix M
// [[Rcpp::export]]
sp_mat makeSymmetric(sp_mat& M) {
  M = symmatu(M);
  return M;
}



void print(const uvec& v) {
  for(uword i = 0; i < v.n_elem; i++){
    Rcout << v[i];
    if(i < v.n_elem-1) Rcout << ", ";
  }
  Rcout << endl;
}

//' Find the indices of the k smallest elements in vector v
//' @param v The vector in which to find k smallest elements
//' @param k The number of smallest elements to find
//' @return A vector containing the unordered indices of the k smallest elements
// [[Rcpp::export]]
uvec Ksmallest(const dvec& v, const uword& k) {
  
  dvec curValues = vec(k).zeros();
  uvec curIndexes = uvec(k).zeros();
  uword curLargestIndex = 0;
  double curLargestValue = v[curLargestIndex];
  
  for( uword i = 0; i < v.size(); i++ ) {
    
    if(i < k) {
      curValues[i] = v[i];
      curIndexes[i] = i;
      
      if(v[i] > curLargestValue) {
        curLargestValue = v[i];
        curLargestIndex = i;
      }
      
    } else if(v[i] < curLargestValue) {
      curValues[curLargestIndex] = v[i];
      curIndexes[curLargestIndex] = i;
      curLargestIndex = curValues.index_max();
      curLargestValue = curValues[curLargestIndex];
    }
  }
  
  for( uword i = 0; i < k; i++ ) {
    curIndexes[i] = curIndexes[i] + 1;
  }
  
  return curIndexes ;
}

//' Compute the euclidean distance between two column vectors
//' @param a First vector
//' @param b Second vector
//' @return The euclidean distance
double euclideanDistVV(const vec& a, const vec& b) {
  return sqrt(sum(square(a - b)));
}

//' Compute the index of the k smallest euclidean distances between
//' the column vector (indexed by j) and all other column vectors in M
//' @param M Input embedding
//' @param j Index of the column to compare other columns to
//' @param k The number of smallest values to find
//' @return A vector of length k
// [[Rcpp::export]]
uvec kSmallestFast(const mat& M, const uword& j, const uword& k) {
  
  dvec curValues = vec(k).zeros();
  uvec curIndexes = uvec(k).zeros();
  uword curLargestIndex = 0;
  double curLargestValue = euclideanDistVV(M.col(j), M.col(0));
  double curDist;
  
  for( uword i = 0; i < M.n_cols; i++ ) {
    
    if(i == j) continue;
    
    curDist = euclideanDistVV(M.col(j), M.col(i));
    
    if(i < k) {
      curValues[i] = curDist;
      curIndexes[i] = i;
      
      if(curDist > curLargestValue) {
        curLargestValue = curDist;
        curLargestIndex = i;
      }
      
    } else if(curDist < curLargestValue) {
      curValues[curLargestIndex] = curDist;
      curIndexes[curLargestIndex] = i;
      curLargestIndex = curValues.index_max();
      curLargestValue = curValues[curLargestIndex];
    }
  }
  
  return curIndexes ;
}

//' Compute the euclidean distance between the column vector indexed by k
//' and all other column vectors in the matrix M
//' @param k Index of column vector
//' @param M An embedding to compute eucildean distances over
//' @return A vector with all euclidean distances
// [[Rcpp::export]]
vec euclideanDist(const uword& k, const mat& M) {
  
  vec dist = vec(M.n_rows).zeros();
  
  for(uword i = 0; i < M.n_rows; i++) {
    dist[i] = sqrt(sum(square(M.row(k) - M.row(i))));
  }
  
  return dist;
}

//' Compute a relational matrix using the supplied classifier. The hadamard product is computed between pairs of vectors.
//' As there are too many pairs to compute, we limit the number of pairs by sampling the nearest neighbors for a given vector.
//' @param M The embedding to compute nearest neighbors with
//' @param k The number of nearest neighbors to sample
//' @param classifier The classifier from R to use
//' @param classFunc The predict function in R that works for the supplied classifier
//' @return A sparse matrix containing the confidence of the classifier if for each element if it is large enough, zero otherwise
// [[Rcpp::export]]
sp_mat createRelationalMatrix(const mat& M, const uword& k, const RObject& classifier, const Function& classFunc) {
  
  sp_mat R(M.n_cols, M.n_cols);
  uvec neighbors = uvec(k);
  uvec classify = uvec(k);
  int* classification;
  mat hadamard(M.n_rows, k);
  //ETAProgressBar pb;
  Progress p(M.n_cols, true);
  
  for( uword i = 0; i < M.n_cols; i++ ) {
    
    neighbors = kSmallestFast(M, i, k);
    hadamard.resize(M.n_rows, k);
    uword s = 0;
    
    for( uword j = 0; j < k; j++) {
      
      // Only do upper triangle
      if(neighbors[j] > i) {
        hadamard.col(s) = hadamardVV(M.col(i), M.col(neighbors[j]));
        classify[s] = neighbors[j];
        s++;
      }
    }
    
    // In case we skip all of the neighbors
    if(s > 0) {
    
      // We may have skipped some neighbors we have already classified
      // So remove any unused trailing columns
      hadamard.resize(M.n_rows, s);
      
      classification = INTEGER(classFunc(classifier, hadamard.t()));
      
      for( uword j = 0; j < k; j++) {
        if(classification[j] - 1 == 1) R(i, classify[j]) = 1;
      }
      
    }
    
    p.increment();
    
  }
  
  p.cleanup();
  //pb.end_display();
  
  // We only filled in the upper triangle, so reflect it to the lower
  R = symmatu(R);
  
  // Make R reflexive
  R.diag().ones();
  
  return R;
}

//' Compute a relational matrix using the supplied classifier. The hadamard product is computed between pairs of vectors.
//' As there are too many pairs to compute, we limit the number of pairs by sampling the nearest neighbors for a given vector.
//' @param M The embedding to compute nearest neighbors with
//' @param k The number of nearest neighbors to sample
//' @param c The batch size base, total batch will be of maximum size k*c
//' @param classifier The classifier from R to use
//' @param classFunc The predict function in R that works for the supplied classifier
//' @return A sparse matrix containing the confidence of the classifier if for each element if it is large enough, zero otherwise
// [[Rcpp::export]]
sp_mat createRelationalMatrixFast(const mat& M, const uword& k, const uword& c, const RObject& classifier, const Function& classFunc) {
  
  sp_mat R(M.n_cols, M.n_cols);
  int* classification;
  uword sTotal = 0;
  mat hadamardCache(M.n_rows, k*c);
  uvec neighborColCache(k*c);
  uvec neighborRowCache(k*c);
  //ETAProgressBar pb;
  Progress p(M.n_cols, true);
  
  for( uword i = 0; i < M.n_cols; i++ ) {
    
    uvec neighbors = kSmallestFast(M, i, k);
    uword s = 0;
    
    for( uword j = 0; j < k; j++) {
      
      // Only fill upper triangle
      uword row = std::min(i, neighbors[j]);
      uword col = std::max(i, neighbors[j]);
      
      // Don't classify something twice
      if(R(row, col) == 0) {
        hadamardCache.col(s + sTotal) = hadamardVV(M.col(i), M.col(neighbors[j]));
        neighborRowCache[s + sTotal] = row;
        neighborColCache[s + sTotal] = col;
        s++;
      }

    }
    
    sTotal += s;
    
    if(i > 0 && ((i+1) % c == 0 || i == M.n_cols-1)) {
      
      // We may have skipped some neighbors we have already classified
      // So remove any unused trailing columns
      hadamardCache.resize(M.n_rows, sTotal);
      
      classification = INTEGER(classFunc(classifier, hadamardCache.t()));
      
      for( uword j = 0; j < sTotal; j++) {
        if(classification[j] - 1 == 1) R(neighborRowCache[j], neighborColCache[j]) = 1;
      }
      
      hadamardCache.resize(M.n_rows, k*c);
      sTotal = 0;
      
    }
    
    p.increment();
    
  }
  
  p.cleanup();
  //pb.end_display();
  
  // We only filled in the upper triangle, so reflect it to the lower
  R = symmatu(R);
  
  // Make R reflexive
  R.diag().ones();
  
  return R;
}

//' Compute a relational matrix using the supplied classifier. The hadamard product is computed between pairs of vectors.
//' As there are too many pairs to compute, we limit the number of pairs by sampling the nearest neighbors for a given vector.
//' @param M The embedding to compute nearest neighbors with
//' @param k The number of nearest neighbors to sample
//' @param classifier The classifier from R to use
//' @param classFunc The predict function in R that works for the supplied classifier
//' @return A sparse matrix containing the probability of a match for a given pair, zero if the pair is not classified
// [[Rcpp::export]]
sp_mat createRelationalPMatrix(const mat& M, const uword& k, const RObject& classifier, const Function& classFunc) {
  
  sp_mat R(M.n_cols, M.n_cols);
  
  uvec neighbors;
  uvec classify = uvec(k);
  double* classification;
  mat hadamard(M.n_rows, k);
  //ETAProgressBar pb;
  //Progress p(M.n_cols, true, pb);
  
  for( uword i = 0; i < M.n_cols; i++ ) {
    
    neighbors = kSmallestFast(M, i, k);
    hadamard.resize(M.n_rows, k);
    uword s = 0;
    
    for( uword j = 0; j < k; j++) {

      // Only do upper triangle
      if(neighbors[j] > i) {
        hadamard.col(s) = hadamardVV(M.col(i), M.col(neighbors[j]));
        classify[s] = neighbors[j];
        s++;
      }
    }
    
    // In case we skip all of the neighbors
    if(s > 0) {
      
      // We may have skipped some neighbors we have already classified
      // So remove any unused trailing columns
      hadamard.resize(M.n_rows, s);
      
      classification = REAL(classFunc(classifier, hadamard.t(), "probabilities"));
      
      // Take the probability of a 'yes' classification
      // which is the second half of the vector
      for( uword j = 0; j < s; j++) {
        R(i, classify[j]) = classification[j + k];
      }
      
    }

    //p.increment();
  }
  
  //p.cleanup();
  //pb.end_display();
  
  // We only filled in the upper triangle, so reflect it to the lower
  R = symmatu(R);
  
  // Make R reflexive
  R.diag().ones();
  
  return R;
}

//' Compute the ideal discounted cumulative gain, when all relevant
//' results are on top of the ranking
//' @param rel A relevance vector
//' @return The ideal DCG
double idcg(const uvec& rel) {
  double sum = 0;
  const uword k = accu(rel);
  
  for(uword i = 1; i <= k; i++) {
    sum += 1 / log2(i + 1);
  }
  
  return sum;
}

//' Compute the (unnormalized) discounted cumulative gain
//' @param rel A relevance vector
//' @return The discounted cumulative gain
double dcg(const uvec& rel) {
  double sum = 0;
  
  for(uword i = 1; i <= rel.size(); i++) {
    sum += (pow(2, rel[i-1]) - 1) / log2(i + 1);
  }
  
  return sum;
}

//' Compute the normalized discounted cumulative gain
//' @param rel A relevance vector
//' @return The normalized discounted cumulative gain
double ndcg(const uvec& rel) {
  return dcg(rel) / idcg(rel);
}

//' Compute the normalized discounted cumulative gain for each labeled vector
//' @param M An embedding
//' @param cluster A vector with labels, a value of -1 is considered unlabeled
//' @return A vector with the NDCG for each labeled vector
// [[Rcpp::export]]
vec calculateNDCG(const mat& M, const vec& cluster) {
  
  uword idx;
  uvec dist_ordered;
  uvec rel = uvec(cluster.size());
  uvec idxV = find(cluster != -1);
  vec out = vec(idxV.size());
  
  ETAProgressBar pb;
  Progress p(idxV.size(), true, pb);
  
  for(uword i = 0; i < idxV.size(); i++) {
    idx = idxV[i];
    dist_ordered = sort_index(euclideanDist(idx, M));
    
    for(uword j = 0; j < cluster.size(); j++) {
      rel[j] = cluster[dist_ordered[j]] == cluster[idx] ? 1 : 0;
    }
    
    out[i] = ndcg(rel);
    
    p.increment();
  }
  return out;
}
