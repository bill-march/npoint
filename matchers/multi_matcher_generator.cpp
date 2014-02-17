/*
 *  multi_matcher_generation.cpp
 *  
 *
 *  Created by William March on 6/21/11.
 *  Copyright 2011 Georgia Institute of Technology. All rights reserved.
 *
 */

#include "multi_matcher_generator.hpp"

size_t npoint_mlpack::MultiMatcherGenerator::FindWhichMatcher_(size_t i, size_t j) {
  
  if (i > j) {
    std::swap(i, j);
  }
  
  assert(i != j);
  
  size_t res = 0;
  
  if (i > 0) {
    for (size_t k = 0; k < i; k++) {
      res += (tuple_size_ - k - 1);
    }
  }
  
  res += (j - i - 1);
  
  return res;
  
} 

void npoint_mlpack::MultiMatcherGenerator::FillInMatchers_(std::vector<size_t>& matcher_ind, 
                                                           size_t k) {

  // we're filling in the kth spot in matcher ind
  
  std::vector<size_t>& matcher_ind_copy(matcher_ind);
  
  for (int i = 0; i < num_bands_[k]; i++) {
    
    // Do I need to copy it again here?
    
    matcher_ind_copy[k] = i;
  
    if (k == matcher_ind.size() - 1) {
      // we've completed a tuple, make the matcher and add to the list
      
      arma::mat matcher_lower(tuple_size_, tuple_size_);
      arma::mat matcher_upper(tuple_size_, tuple_size_);
      
      for (int m = 0; m < tuple_size_; m++) {
        
        matcher_lower(m,m) = 0.0;
        matcher_upper(m,m) = 0.0;
        
        for (int n = m+1; n < tuple_size_; n++) {
          
          size_t which_matcher = FindWhichMatcher_(m, n);
          
          matcher_lower(m,n) = min_bands_[which_matcher] 
                              + (double)matcher_ind_copy[which_matcher] 
                                * band_steps_[which_matcher];
          matcher_upper(m,n) = min_bands_[which_matcher] 
                              + (double)(matcher_ind_copy[which_matcher] + 1)
                                  * band_steps_[which_matcher];
          
          matcher_lower(n,m) = matcher_lower(m,n);
          matcher_upper(n,m) = matcher_upper(m,n);
          
        } // for n
        
      } // for m
      
      //matcher.print("Matcher (in generator):");
      
      //matcher_lower.print("Lower bounds (in generator):");
      //matcher_upper.print("Upper bounds (in generator):");
      
      //matchers_.push_back(matcher);
      matcher_lower_bounds_.push_back(matcher_lower);
      matcher_upper_bounds_.push_back(matcher_upper);
      
    } // finished this matcher
    else {
      
      FillInMatchers_(matcher_ind_copy, k+1);
      
    } // keep recursing
    
  } // for i
  
} // FillInMatchers_

npoint_mlpack::MultiMatcherGenerator::MultiMatcherGenerator()
{

  // don't want to do anything in this case, not going to use it
  
}

void npoint_mlpack::MultiMatcherGenerator::Init(std::vector<double>& min_bands,
                                                            std::vector<double>& max_bands,
                                                            std::vector<int>& num_bands, 
                                                            int tuple_size)
{
  tuple_size_ = tuple_size;
  
  band_steps_.resize(num_bands.size());
  matcher_dists_.resize(num_bands.size());
  
  min_bands_ = min_bands;
  max_bands_ = max_bands;
  num_bands_ = num_bands;
  
  for (size_t i = 0; i < num_bands_.size(); i++) {
    
    double band_step = (max_bands_[i] - min_bands_[i]) / (double)num_bands_[i];
    
    band_steps_[i] = band_step;
    
    matcher_dists_[i].resize(num_bands_[i]);
    
    if (num_bands_[i] > 1) {
      for (int j = 0; j < num_bands_[i]; j++) {
        
        matcher_dists_[i][j] = min_bands_[i] + (double)j * band_step;
        
      } // for j
    } // if more than one band
    else {
      
      matcher_dists_[i][0] = min_bands_[i];
      
    } // only one band
    
  } // for i
  
  std::vector<size_t> matcher_ind(num_bands_.size());
  
  FillInMatchers_(matcher_ind, 0);
  
}

arma::mat& npoint_mlpack::MultiMatcherGenerator::lower_matcher(size_t i) {
  return matcher_lower_bounds_[i];
}

arma::mat& npoint_mlpack::MultiMatcherGenerator::upper_matcher(size_t i) {
  return matcher_upper_bounds_[i];
}

int npoint_mlpack::MultiMatcherGenerator::num_matchers() {
  return matcher_lower_bounds_.size();
}

void npoint_mlpack::MultiMatcherGenerator::Print() {
  
  for (size_t i = 0; i < matcher_lower_bounds_.size(); i++) {
 
    matcher_lower_bounds_[i].print("Lower bound: ");
    matcher_upper_bounds_[i].print("Upper bound: ");
    std::cout << "\n";
    
  }
  
}





