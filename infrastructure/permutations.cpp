/*
 *  permutations.cpp
 *  
 *
 *  Created by William March on 2/7/11.
 *  Copyright 2011 Georgia Institute of Technology. All rights reserved.
 *
 */

#include "permutations.hpp"

// The constructor needs to fill in permuation_indices_
npoint_mlpack::Permutations::Permutations(size_t n)
:
tuple_size_(n),
num_perms_(1),
permutation_indices_(tuple_size_, num_perms_)
{
  
  // compute the factorial
  for (int i = 2; i <= tuple_size_; i++) {
    num_perms_ = num_perms_ * i;
  } // for i
  
  // make the trial permutation
  arma::Col<int> trial_perm(tuple_size_);
  trial_perm.fill(-1);
  
  // allocate the matrix
  permutation_indices_.set_size(tuple_size_, num_perms_);
  permutation_indices_.fill(-1);
  
  int perm_index = 0;

  GeneratePermutations_(0, &perm_index, trial_perm);
  
} // constructor


void npoint_mlpack::Permutations::GeneratePermutations_(int k, int* perm_index,
                                              arma::Col<int>& trial_perm) {

  // simple bounds check
  if (*perm_index >= num_perms_) {
    return;
  } 

  // Iterate over all points (i.e. everything that might be in the permutation)
  for (int i = 0; i < tuple_size_; i++) {
    
    bool perm_ok = true;
    
    // Iterate over everything already in trial_perm
    for (int j = 0; perm_ok && j < k; j++) {
      
      // Did we already use j in this one?  Then don't use it again.
      if (trial_perm[j] == i) {
        perm_ok = false;
      }
      
    } // for j
    
    // go to the next i if this one didn't work, otherwise proceed
    if (perm_ok) {
      
      // add i to the trial permutation
      trial_perm(k) = i;
    
      // if the whole permutation is filled, put it in the matrix
      if (k == tuple_size_ - 1) {
        
        permutation_indices_.col(*perm_index) = trial_perm;
        (*perm_index)++;
        
      } // is the permutation filled?
      else {
        // move on to the next spot in the permutation
        GeneratePermutations_(k+1, perm_index, trial_perm);
        
      } // permutation not filled
        
    } // if perm_ok
    
  } // for i
  
  
} // GeneratePermutations_

int npoint_mlpack::Permutations::num_permutations() const {
  return num_perms_;
}

// just accesses elements of permutation_indices_
int npoint_mlpack::Permutations::GetPermutation(size_t perm_index, size_t point_index) const {
  
  // note that these are backward from how they're input
  // the old code does it this way and I don't want to get confused
  return permutation_indices_(point_index, perm_index);
  
} // GetPermutation()


// for debugging purposes
void npoint_mlpack::Permutations::Print() {
  
  permutation_indices_.print("Permutation Indices:");
  
}



