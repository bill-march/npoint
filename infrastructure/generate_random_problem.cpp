/*
 *  generate_random_problem.cpp
 *  
 *
 *  Created by William March on 10/31/11.
 *  Copyright 2011 Georgia Institute of Technology. All rights reserved.
 *
 */

#include "generate_random_problem.hpp"


npoint_mlpack::GenerateRandomProblem::GenerateRandomProblem(double matcher_dist_lo, double matcher_dist_hi,
                      double matcher_thick_lo, double matcher_thick_hi,
                      int num_data_lo, int num_data_hi) :
matcher_dist_lo_(matcher_dist_lo), matcher_dist_hi_(matcher_dist_hi),
matcher_thick_lo_(matcher_thick_lo), matcher_thick_hi_(matcher_thick_hi),
num_data_lo_(num_data_lo), num_data_hi_(num_data_hi),
use_helper_(false)
{
  //printf("constructed test class\n");
}

npoint_mlpack::GenerateRandomProblem::GenerateRandomProblem(double matcher_dist_lo, double matcher_dist_hi,
                                                            double matcher_thick_lo, double matcher_thick_hi,
                                                            int num_data_lo, int num_data_hi,
                                                            ResamplingHelper& helper) :
matcher_dist_lo_(matcher_dist_lo), matcher_dist_hi_(matcher_dist_hi),
matcher_thick_lo_(matcher_thick_lo), matcher_thick_hi_(matcher_thick_hi),
num_data_lo_(num_data_lo), num_data_hi_(num_data_hi),
use_helper_(true),
helper_(helper)
{}


void npoint_mlpack::GenerateRandomProblem::GenerateRandomSet(arma::mat& data) {
  
  for (unsigned int row_ind = 0; row_ind < data.n_rows; row_ind++) {
    
    double row_add;
    double row_scale;
    
    if (use_helper_) {
      if (row_ind == 0) {
        row_scale = helper_.x_size();
        row_add = helper_.x_min();
      }
      else if (row_ind == 1) {
        row_scale = helper_.y_size();
        row_add = helper_.y_min();
      }
      else { // row_ind == 2
        row_scale = helper_.z_size();
        row_add = helper_.z_min();
      }
    }
    else {
      row_add = 0.0;
      row_scale = 1.0;
    }
    
    for (unsigned int col_ind = 0; col_ind < data.n_cols; col_ind++) {
      
      data(row_ind,col_ind) = (mlpack::math::Random() * row_scale) + row_add;
      
    }
    
  }
  
} // GenerateRandomSet

// fills the matcher with distances and returns the thickness
double npoint_mlpack::GenerateRandomProblem::GenerateRandomMatcher(arma::mat& matcher) {
  
  for (unsigned int i = 0; i < matcher.n_rows; i++) {
    
    matcher(i,i) = 0.0;
    
    for (unsigned int j = i+1; j < matcher.n_cols; j++) {
      
      //matcher(i,j) = matcher_dist_gen();
      //matcher(j,i) = matcher(i,j);
      
      matcher(i,j) = mlpack::math::Random(matcher_dist_lo_, matcher_dist_hi_);
      matcher(j,i) = matcher(i,j);
      
    }
    
  }
  
  double matcher_thick = mlpack::math::Random(matcher_thick_lo_, 
                                              matcher_thick_hi_);
  
  return matcher_thick;
  
} 

void npoint_mlpack::GenerateRandomProblem::GenerateRandomMatcher(arma::mat& lower_bounds, 
                                                       arma::mat& upper_bounds) {
  
  assert(lower_bounds.n_cols == lower_bounds.n_rows);
  assert(lower_bounds.n_cols == upper_bounds.n_rows);
  assert(upper_bounds.n_rows == upper_bounds.n_cols);
  
  for (unsigned int i = 0; i < lower_bounds.n_rows; i++) {
    
    lower_bounds(i,i) = 0.0;
    upper_bounds(i,i) = 0.0;
    
    for (unsigned int j = i+1; j < lower_bounds.n_cols; j++) {
      
      double val1 = mlpack::math::Random(matcher_dist_lo_, matcher_dist_hi_);
      double val2 = mlpack::math::Random(matcher_dist_lo_, matcher_dist_hi_);
      
      upper_bounds(i,j) = std::max(val1, val2);
      upper_bounds(j,i) = upper_bounds(i,j);
      
      lower_bounds(i,j) = std::min(val1, val2);
      lower_bounds(j,i) = lower_bounds(i,j);
      
    }
    
  }
  
} // GenerateRandomMatcher (upper and lower)
