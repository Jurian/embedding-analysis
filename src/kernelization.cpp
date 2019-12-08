#include <RcppArmadillo.h>
#include <progress.hpp>
#include <progress_bar.hpp>
#include "eta_progress_bar.hpp"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppProgress)]]

using namespace Rcpp;
using namespace arma;

// Define constants for readability
static const sword PERMANENT = 1;
static const sword FORBIDDEN = -1;

void print(const uvec& v) {
  for(uword i = 0; i < v.n_elem; i++){
    Rcout << v[i];
    if(i < v.n_elem-1) Rcout << ", ";
  }
  Rcout << endl;
}

bool isClique(const sp_mat& M, const uvec& idx) {

  if(idx.n_elem == 1) return true;
  
  uword sum = 0;
  for(uword i = 0; i < idx.n_elem; i++) {
    for(uword j = i+1; j < idx.n_elem; j++) {
      if(M(idx[i],idx[j]) != 0) sum++;
    }
  }
  
  return sum == (0.5 * idx.n_elem * ( idx.n_elem - 1 ));
}


//' Calculates the number of common and non-common neighbors for vertices i and j
//' @param M Sparse adjacency matrix
//' @param i Index of vertex i
//' @param j Index of vertex j
//' @return An array with in positions 0 and 1 common and non-common neighbors respectively
uword* neighborInfo(const sp_mat& M, const uword& i, const uword& j) {
  uword* sums = new uword[2] {0,0};
  for(uword a = 0; a < M.n_rows; a++) {
    if(M(a, i) != M(a, j)) sums[1]++;
    else if(M(a, i) != 0 && M(a, j) != 0) sums[0]++;
  }
  return sums;
}

// [[Rcpp::export]]
mat permAndForbEdges(const sp_mat& M, const uvec& idx, const uword& k) {
  
  if(idx.n_elem == 0) {
    throw 0;
    //std::stringstream ss;
    //ss << "Cannot kernalize empty component";
    //stop(ss.str());
  }
  
  // Marks permanent (1) and forbidden (-1) edges
  mat edgeInfo(idx.n_elem, idx.n_elem, fill::zeros);
  
  // Stop right away if there is only one vertex
  if(idx.n_elem == 1) return edgeInfo;
  
  // Apply rule 1
  for(uword i = 0; i < idx.n_elem; i++) {
    for(uword j = i+1; j < idx.n_elem; j++) {
      
      const uword *n = neighborInfo(M, idx[i], idx[j]);
      
      // Conflict state, there is no solution
      if(n[0] > k && n[1] > k) {
        throw 1;
        //std::stringstream ss;
        // Don't forget to convert indexes to R standard
        //ss << "Cannot mark (" <<(i+1)<<", "<<(j+1)<< "), nr of common and uncommon neighbors exceeds " << k;
        //stop(ss.str());
      }
      
      // The number of common neighbors exceeds k, mark as permanent
      else if(n[0] > k) {
        edgeInfo(i,j) = PERMANENT;
        edgeInfo(j,i) = PERMANENT;
      }
      
      // The number of uncommon neighbors exceeds k, mark as forbidden
      else if(n[1] > k) {
        edgeInfo(i,j) = FORBIDDEN;
        edgeInfo(j,i) = FORBIDDEN;
      }
    }
  }
  
  // Apply rule 2
  for(uword i = 0; i < idx.n_elem; i++) {
    for(uword j = i+1; j < idx.n_elem; j++) {

      // Go through all other nodes...
      for(uword l = 0; l < idx.n_elem; l++) {
        // ...except i and j
        if(l == i || l == j) continue;
        
        // If we previously marked 2 edges of this triangle as forbidden and permanent, 
        // the third edge must be forbidden
        if((edgeInfo(i,l) == FORBIDDEN && edgeInfo(j,l) == PERMANENT) ||
           (edgeInfo(j,l) == FORBIDDEN && edgeInfo(i,l) == PERMANENT)) {
          
          // conflict state, we already marked this edge as permanent
          if(edgeInfo(i,j) == PERMANENT) {
            throw 2;
            //std::stringstream ss;
            // Don't forget to convert indexes to R standard
            //ss << "Cannot mark (" <<(i+1)<<", "<<(j+1)<< ") as forbidden, already marked as permanent";
            //stop(ss.str());
          }
          
          edgeInfo(i, j) = FORBIDDEN;
          edgeInfo(j, i) = FORBIDDEN;
        } 
        // If we previously marked 2 edges in this triangle as permanent,
        // the third edge must be permanent
        else if(edgeInfo(i,l) == PERMANENT && edgeInfo(j,l) == PERMANENT) {
          
          // Conflict state, we already marked this edge as forbidden
          if(edgeInfo(i,j) == FORBIDDEN) {
            throw 2;
            //std::stringstream ss;
            // Don't forget to convert indexes to R standard
            //ss << "Cannot mark (" <<(i+1)<<", "<<(j+1)<< ") as permanent, already marked as forbidden";
            //stop(ss.str());
          }
          
          edgeInfo(i, j) = PERMANENT;
          edgeInfo(j, i) = PERMANENT;
        }
      }
    }
  }

  return edgeInfo;
}

//' Find all equivalence classes in the sparse matrix M
//' @param M A sparse matrix, assumed to be symmetric, reflexive and transitive
//' @return A vector with all equivalence class labels
uvec findEquivalenceClasses(const sp_mat& M) {
  
  uvec captains = linspace<uvec>(0, M.n_cols-1, M.n_cols);
  uword currentCol = 0;
  const sp_mat::const_iterator end = M.end();
  
  // Go through all non-zero positions in M
  for(sp_mat::const_iterator it = M.begin() ; it != end; ++it ) {
    // Only use the first element of each column and only cover top triangle
    if(currentCol < it.col() && it.col() > it.row()) {
      captains[it.col()] = it.row();
      // Skip over the other elements in this column
      currentCol = it.col();
    } else {
      continue;
    }
  }
  
  return captains;
}

bool cmpSparseMatrices(const sp_mat& A, const sp_mat& B) {
  
  sp_mat::const_iterator it_A = A.begin();
  sp_mat::const_iterator it_B = B.begin();
  
  const sp_mat::const_iterator A_end = A.end();
  const sp_mat::const_iterator B_end = B.end();
  
  for(;it_A != A_end && it_B != B_end; ++it_A, ++it_B) {
    if(it_A.row() != it_B.row()) return false;
    if(it_A.col() != it_B.col()) return false;
  }
  
  return true;
}

uvec findConnectedComponents(const sp_mat& M) {
  
  sp_mat P = M, R = M, T;
  
  // Keep taking exponents of S until
  while(true) {
    P = P * M;
    T = R + P;
    
    if(!cmpSparseMatrices(R, T)) {
      R = T;
    } else {
      break;
    }
  }
  
  // At this point R is symmetric, reflexive and transitive
  return findEquivalenceClasses(R);
  
}

bool commonNeighbors(const sp_mat& M, const uword& i, const uword& j) {
  if(M.col(i).n_nonzero != M.col(j).n_nonzero) return false;
  
  sp_mat::const_col_iterator it_a = M.begin_col(i);
  sp_mat::const_col_iterator it_b = M.begin_col(j);
  const sp_mat::const_col_iterator a_end = M.end_col(i);
  
  for(; it_a != a_end; ++it_a, ++it_b) {
    if(it_a.row() != it_b.row()) return false;
  }
  
  return true;
}

uvec findCriticalClique(const sp_mat& M, const uword& idx) {
  
  uvec clique(M.col(idx).n_nonzero);
  uword j = 0;
  
  sp_mat::const_col_iterator it = M.begin_col(idx);
  const sp_mat::const_col_iterator it_end = M.end_col(idx);
  
  for(; it != it_end; ++it) {
    if(it.row() == idx) clique[j++] = idx;
    else if(commonNeighbors(M, idx, it.row())) clique[j++] = it.row();
  }
  
  clique.resize(j);
  
  return clique;
}

// [[Rcpp::export]]
uvec criticalCliques(const sp_mat& M) {
  
  uvec cliques(M.n_cols);
  bool processed[M.n_cols] {};
  
  for(uword i = 0; i < M.n_cols; i++) {
    if(processed[i]) continue;
    
    const uvec crit = findCriticalClique(M, i);
    
    for(uword j = 0; j < crit.n_elem; j++) {
      cliques[crit[j]] = i;
      processed[crit[j]] = true;
    }
  }
  
  return cliques;
}

// [[Rcpp::export]]
umat extractEquivalenceClass(const sp_mat& M, const uvec& idx) {
  umat out(idx.size(), idx.size());
  
  for(uword i = 0; i < idx.n_elem; i++) {
    for(uword j = 0; j < idx.n_elem; j++) {
      out(i,j) = M(idx[i],idx[j]);
    }
  }
  
  return out;
}

// [[Rcpp::export]]
double subgraphConnectedness(const sp_mat& M, const uvec& idx) {
  
  double n_elem = 0.5 * idx.n_elem * ( idx.n_elem - 1 );
  uword sum = 0;
  
  for(uword i = 0; i < idx.n_elem; i++) {
    for(uword j = i+1; j < idx.n_elem; j++) {
      if(M(idx[i],idx[j]) != 0) sum++;
    }
  }
  return sum/n_elem;
}

// [[Rcpp::export]]
sp_mat editClusters(const sp_mat& M, const uword& kMin, const uword& kMax, const bool debug = false) {
  
  sp_mat _M = M;
  const uvec components = findConnectedComponents(M);
  // We sort the components once so that we can quickly find all members without searching all the vertices
  const uvec comp_srt_ind = sort_index(components);
  
  bool processed[M.n_cols] {};
  //bool cliques[M.n_cols] {};
  
  Progress p(M.n_cols, true);
  uword maxComponentSize = 1;
  
  // First we find permanent and forbidden edges and edit _M accordingly
  // For each vertex...
  for(uword i = 0; i < M.n_cols;) {
    
    if(debug) Rcout << i;
    // We already processed the component this vertex belongs to
    if(processed[comp_srt_ind[i]]) {
      if(debug) Rcout << " already processed" << endl;
      p.increment();
      i++;
      continue;
    }
    
    // Find the connnected component this vertex belongs to
    std::vector<uword> c;
    c.reserve(maxComponentSize);
    for( uword j = i; components[comp_srt_ind[j]] == components[comp_srt_ind[i]]; j++) {
      c.push_back(comp_srt_ind[j]);
    }
    i += c.size();

      
    // Contains the indexes of vertices
    const uvec component(c);
    
    // Make sure we reserve enough space to at least accommodate the largest component so far
    maxComponentSize = std::max(maxComponentSize, component.n_elem);
    
    if(debug) Rcout << " of size " << component.n_elem;
    // If this component is already a clique in M, we mark it and skip to the next vertex
    if(isClique(M, component)) {
      for(uword j = 0; j < component.n_elem; j++) {
        //cliques[component[j]] = true;
        processed[component[j]] = true;
      }
      if(debug) Rcout << " is clique" << endl;
      p.increment();
    } else {
      
      uword k = kMin;
  
      // Find the smallest k that will yield a valid cluster edit
      while(k <= kMax) {
        try {
          if(debug) Rcout << endl << " trying k = " << k;
          const mat edgeInfo = permAndForbEdges(M, component, k);
          if(debug) Rcout << "... valid k" << endl;
          // Edit this component in _M
          for(uword a = 0; a < component.n_elem; a++) {
            for(uword b = a+1; b < component.n_elem; b++) {
              if(edgeInfo(a,b) == PERMANENT) {
                _M(component[a], component[b]) = 1;
                _M(component[b], component[a]) = 1;
              }
              else if(edgeInfo(a,b) == FORBIDDEN) {
                _M(component[a], component[b]) = 0;
                _M(component[b], component[a]) = 0;
              }
            }
          }
          
          // We may have turned this component into a clique in _M
          if(isClique(_M, component)) {
            for(uword j = 0; j < component.n_elem; j++) {
              //cliques[component[j]] = true;
              processed[component[j]] = true;
            }
          } else {
            // Now that the connected component has been edited, we can do the branching
            
            for(uword j = 0; j < component.n_elem; j++) {
              processed[component[j]] = true;
            }
          }
          
          p.increment();
          break;
        } catch( int e) {
          if(debug) Rcout << "... invalid k";
          k++;
          if(debug && k > kMax) Rcout << endl;
        }
      }
    }
  }
  p.cleanup();
  
  uword sum = 0;
  for(uword i = 0; i < M.n_cols; i++) {
    if(!processed[i]) sum++;
  }
  Rcout << "unprocessed: " << sum << endl;
  

  return _M;
}