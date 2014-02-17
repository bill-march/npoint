/*
 *  permutations.hpp
 *  
 *
 *  Created by William March on 2/7/11.
 *  Copyright 2011 Georgia Institute of Technology. All rights reserved.
 *
 *  Stores all the permutations of n elements.  Used in the standard 
 *  multi-tree algorithm.
 *
 */

#ifndef __MLPACK_METHODS_NPOINT_INFRASTRUCTURE_PERMUTATIONS_HPP
#define __MLPACK_METHODS_NPOINT_INFRASTRUCTURE_PERMUTATIONS_HPP

#include <mlpack/core.hpp>

namespace npoint_mlpack {

  class Permutations {
    
  private:
    
    ///////////// member variables ///////////////////////
    
    // the value of n being considered
    int tuple_size_;
    
    // tuple_size_!
    int num_perms_;
    
    // permutation_indices_(i, j) is the location of point i in permutation j
    arma::Mat<int> permutation_indices_;
    
    
    
    //////////////// functions //////////////////
    
    // helper function at startup, actually forms all the permutations
    void GeneratePermutations_(int k, int* perm_index,
                               arma::Col<int>& trial_perm);
    
    
  public:
    
    // used to have dummy constructor here, might still need it?
    
    // The constructor needs to fill in permuation_indices_
    Permutations(size_t n);
    
    int num_permutations() const;
    
    // just accesses elements of permutation_indices_
    int GetPermutation(size_t perm_index, size_t point_index) const;
    
    // for debugging purposes
    void Print();
    
    
  }; // class
  
} // namespace

#endif

