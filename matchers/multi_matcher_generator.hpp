/*
 *  matcher_generation.hpp
 *  
 *
 *  Created by William March on 6/21/11.
 *  Copyright 2011 Georgia Institute of Technology. All rights reserved.
 *
 */

#ifndef _MLPACK_NPOINT_MLPACK_MATCHERS_MULTI_MATCHER_GENERATOR_HPP_
#define _MLPACK_NPOINT_MLPACK_MATCHERS_MULTI_MATCHER_GENERATOR_HPP_

#include <mlpack/core.hpp>

namespace npoint_mlpack {
 
  class MultiMatcherGenerator {
    
  private:
    
    int tuple_size_;
    
    std::vector<double> min_bands_;
    std::vector<double> max_bands_;
    std::vector<int> num_bands_;
    
    std::vector<double> band_steps_;
    
    std::vector<arma::mat> matcher_lower_bounds_;
    std::vector<arma::mat> matcher_upper_bounds_;
    
    std::vector<std::vector<double> > matcher_dists_;
    
    void FillInMatchers_(std::vector<size_t>& matcher_ind, size_t k);
    
    size_t FindWhichMatcher_(size_t i, size_t j);
    
  public: 
    
    // empty constructor for when we aren't going to use the generator
    MultiMatcherGenerator();
    
    //MultiMatcherGenerator(std::vector<double>& min_bands,
    //                      std::vector<double>& max_bands,
    //                      std::vector<int>& num_bands, 
    //                      int tuple_size);
    
    void Init(std::vector<double>& min_bands,
                          std::vector<double>& max_bands,
                          std::vector<int>& num_bands, 
                          int tuple_size);
    
    arma::mat& lower_matcher(size_t i);

    arma::mat& upper_matcher(size_t i);
    
    int num_matchers();
     
    void Print();
    
  }; // class
  
} //namespace




#endif

