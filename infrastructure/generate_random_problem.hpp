/*
 *  generate_random_problem.hpp
 *  
 *
 *  Created by William March on 10/31/11.
 *  Copyright 2011 Georgia Institute of Technology. All rights reserved.
 *
 */

#ifndef __MLPACK_METHODS_NPOINT_INFRASTRUCTURE_GENERATE_RANDOM_PROBLEM_HPP
#define __MLPACK_METHODS_NPOINT_INFRASTRUCTURE_GENERATE_RANDOM_PROBLEM_HPP

#include <mlpack/core.hpp>

#include "resampling_helper.hpp"


namespace npoint_mlpack {

  class GenerateRandomProblem {
    
  private:
    
    double matcher_dist_lo_;
    double matcher_dist_hi_;
    
    double matcher_thick_lo_;
    double matcher_thick_hi_;
    
    int num_data_lo_;
    int num_data_hi_;
    
    bool use_helper_;
    
    ResamplingHelper helper_;
    
  public:
    
    void GenerateRandomSet(arma::mat& data);
    
    // fills the matcher with distances and returns the thickness
    double GenerateRandomMatcher(arma::mat& matcher);
    
    void GenerateRandomMatcher(arma::mat& lower_bounds, 
                               arma::mat& upper_bounds);

    
    GenerateRandomProblem(double matcher_dist_lo, double matcher_dist_hi,
                          double matcher_thick_lo, double matcher_thick_hi,
                          int num_data_lo, int num_data_hi);
    
    GenerateRandomProblem(double matcher_dist_lo, double matcher_dist_hi,
                          double matcher_thick_lo, double matcher_thick_hi,
                          int num_data_lo, int num_data_hi,
                          ResamplingHelper& helper);
    
    
    
  }; // class
  
} 


#endif