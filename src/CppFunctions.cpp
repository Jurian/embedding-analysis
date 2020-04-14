#include <progress.hpp>
#include <progress_bar.hpp>
#include <set>

// [[Rcpp::depends(RcppProgress)]]

using namespace std;
using namespace Rcpp;

// [[Rcpp::export]]
DataFrame createHadamardPairs(const DataFrame& vect, const NumericVector& clust, const size_t minK = 10, const bool display_progress = false) {
  
  // Note that every column is a embedding-vector in M
  // We use the squared euclidean distance, which is faster to calculate
  
  DataFrame pairs = DataFrame::create();
  set<size_t> pairIndex;
  
  const size_t N = clust.size();
  Progress p(vect.size(), display_progress);
  
  for(size_t i = 0; i < N; i++) {
    
    R_CheckUserInterrupt();
    
    const int currentClust = clust[i];
    
    // We skip all vectors with a cluster id of -1
    if(currentClust == -1) {
      p.increment();
      continue;
    }
    
    for(size_t j = 0; j < N; j++) {
      if(clust[j] == currentClust) pairIndex.insert(j);
    }

    // We choose a minimum of minK pairs, or thrice the size of the current cluster
    const size_t k = max(minK, pairIndex.size() * 3);
    
    const NumericVector currentVec = vect[i];
    NumericVector otherVec = vect[0];
    
    // Find the indices of the nearest vectors in a linear fashion
    NumericVector curValues(k);
    IntegerVector curIndexes(k);
    int curLargestIndex = 0;
    double curLargestValue = sum(pow(currentVec - otherVec, 2.0));
    double curDist = 0;
    
    for(size_t j = 0; j < N; j++) {
      
      if(i == j) continue; // don't campare with itself
      
      otherVec = vect[j];
      curDist = sum(pow(currentVec - otherVec, 2.0));
      
      if(j < k) {
        curValues[j] = curDist;
        curIndexes[j] = j;
        
        if(curDist > curLargestValue) {
          curLargestValue = curDist;
          curLargestIndex = j;
        }
        
      } else if(curDist < curLargestValue) {
        curValues[curLargestIndex] = curDist;
        curIndexes[curLargestIndex] = j;
        curLargestIndex = which_max(curValues) - 1;
        curLargestValue = curValues[curLargestIndex];
      }
    }
    
    // Add the indices of the nearest vectors
    for(size_t j = 0; j < k; j++) {
      pairIndex.insert(curIndexes[j]);
    }
    
    // Calculate the hadamard product of all pairs we found
    for(set<size_t>::iterator it = pairIndex.begin(); it != pairIndex.end(); ++it) {
      
      const size_t otherIdx = (*it);
      const NumericVector otherVec = vect[otherIdx];
      NumericVector hadamard = currentVec * otherVec;
      
      hadamard.push_back((currentClust == clust[otherIdx]) ? 1 : 0);
      pairs.push_back(hadamard);
    }
    
    pairIndex.clear();
    p.increment();
  }
  p.cleanup();
  return pairs;
}

