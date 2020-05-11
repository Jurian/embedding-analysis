#include <RcppArmadillo.h>
#include <progress.hpp>
#include <progress_bar.hpp>
#include "eta_progress_bar.hpp"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppProgress)]]

using namespace std;
using namespace Rcpp;
using namespace arma;

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

arma::sp_mat upperTriangle(arma::sp_mat& M) {
  M = arma::trimatu(M);
  M.diag().zeros();
  return M;
}

arma::sp_mat makeReflexive(arma::sp_mat& M) {
  M.diag().ones();
  return M;
}

arma::sp_mat makeSymmetric(arma::sp_mat& M) {
  M = symmatu(M);
  return M;
}

arma::sp_mat makeTransitive(const arma::sp_mat& M) {
  
  arma::sp_mat P = M, R = M, T;
  
  // Keep taking exponents of M until there is no change
  while(true) {
    P = P * M;
    T = R + P;
    
    if(!cmpSparseMatrices(R, T)) {
      
      // Continue with taking exponents
      R = T;
      
    } else {
      
      // Set set all non zero values back to 1
      sp_mat::iterator it = R.begin();
      sp_mat::const_iterator it_end = R.end();
      for(; it != it_end; ++it)  *it = 1;
      
      return R;
    }
  }
}

// [[Rcpp::export]]
double cpp_CosineSimilarity(const arma::vec& a, const arma::vec& b) {
  const double d = arma::dot(a, b);
  
  const arma::vec p1 = arma::pow(a, 2);
  const arma::vec p2 = arma::pow(b, 2);
  
  const double s1 = arma::sum(p1);
  const double s2 = arma::sum(p2);
  
  const double norm1 = sqrt(s1);
  const double norm2 = sqrt(s2);
  
  return d / (norm1 * norm2);
}

// [[Rcpp::export]]
arma::sp_mat cpp_RelationalMatrix(const arma::mat& embedding, const arma::vec& coeff) {
  
  const size_t N = embedding.n_cols;
  arma::sp_mat R(N, N);
  
  Progress p(N*N, true);
  
  for(size_t i = 0; i < N; i++) {
    
    if(i % 10 == 0) R_CheckUserInterrupt();
    
    const arma::vec a = embedding.col(i); 

    for(size_t j = i + 1; j < N; j++) {
      
      const arma::vec b = embedding.col(j); 
      
      if(coeff[0] + coeff[1] * cpp_CosineSimilarity(a,b) < 0) R(i, j) = 1;
      
      p.increment();
    }
  }
  
  p.cleanup();
  
  //R = makeReflexive(R);
  R = makeSymmetric(R);
  //R = makeTransitive(R);
  //R = upperTriangle(R);
  
  return R;
}

// [[Rcpp::export]]
Rcpp::NumericVector cpp_Performance(const arma::uvec pred, const arma::uvec obs, const arma::uword posValue) {
  
  const arma::uword N = pred.n_elem;
  
  double TP = 0;
  double FP = 0;
  double TN = 0;
  double FN = 0;
  
  for(arma::uword i = 0; i < N; i++) {
    
    if(pred[i] == posValue && obs[i] == posValue) {
      TP++;
    }
    else if(pred[i] == posValue && obs[i] != posValue) {
      FP++;
    }
    else if(pred[i] != posValue && obs[i] != posValue) {
      TN++;
    }
    else if(pred[i] != posValue && obs[i] == posValue) {
      FN++;
    }
  }
  
  double a = (TP + TN)/(TP + TN + FP + FN);
  double p = TP/(TP + FP);
  double r = TP/(TP + FN);
  double f05 = (1 + std::pow(0.5, 2)) * ((p*r) / ((std::pow(0.5, 2) * p) + r));
  double f1 = (1 + std::pow(1, 2)) * ((p*r) / ((std::pow(1, 2) * p) + r));
  double f2 = (1 + std::pow(2, 2)) * ((p*r) / ((std::pow(2, 2) * p) + r));

  Rcpp::NumericVector perf = 
    NumericVector::create(
      _["accuracy"] = a, 
      _["precision"] = p, 
      _["recall"] = r,
      _["F0.5"] = f05,
      _["F1"] = f1,
      _["F2"] = f2
  );
  
  return perf;
}

// [[Rcpp::export]]
arma::uvec cpp_CriticalCliques(const arma::sp_mat& M) {
  
  arma::uvec cliques = linspace<uvec>(0, M.n_cols-1, M.n_cols);

  
  bool processed[M.n_cols] {};

  for(arma::uword i = 0; i < M.n_cols; i++) {
    
    if(processed[i]) continue;

    sp_mat::const_col_iterator it_i = M.begin_col(i);
    sp_mat::const_col_iterator it_i_end = M.end_col(i);

    for(; it_i != it_i_end; ++it_i) {
      
      bool commonNeighbors = true;
      const uword j = it_i.row();

      if(M.col(i).n_nonzero != M.col(j).n_nonzero) continue;
      
      sp_mat::const_col_iterator N_i = M.begin_col(i);
      sp_mat::const_col_iterator N_i_end = M.end_col(i);
      
      for(; N_i != N_i_end; ++N_i) {
        
        const uword iRow = N_i.row();
        
        if(iRow == i) continue;
        if(iRow == j) continue;
        
        if(M.at(iRow,j) == 0) {
          //Rcout << iRow << endl;
          commonNeighbors = false;
          break;
        }
      }
      
      if(commonNeighbors) {
        processed[j] = true;
        cliques[j] = i;
      } 
    }
    
    processed[i] = true;
  }
  
  return cliques;
}


