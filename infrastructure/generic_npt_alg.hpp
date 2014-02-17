/*
 *  generic_npt_alg.hpp
 *  
 *
 *  Created by William March on 8/24/11.
 *  Copyright 2011 Georgia Institute of Technology. All rights reserved.
 *
 */


/*
 *  This class handles the multi-tree recursion using NodeTuples.  It is 
 *  templated by a matcher class, which has access to the data, checks for 
 *  prunes and stores the results. 
 *
 *  Basically, it does very little while the matcher and NodeTuple do most
 *  of the work.
 */

#ifndef __MLPACK_METHODS_NPOINT_INFRASTRUCTURE_GENERIC_NPT_ALG_HPP
#define __MLPACK_METHODS_NPOINT_INFRASTRUCTURE_GENERIC_NPT_ALG_HPP


#include "node_tuple.hpp"


namespace npoint_mlpack {
  
  template <class TMatcher>
  class GenericNptAlg {
  
  private:
    
    // Matcher owns the data
    TMatcher& matcher_;
    
    // a list of tree roots
    // NOTE: trees_.size() is the number of distinct sets in this computation
    std::vector<NptNode*> trees_;
    
    // how many times should each tree appear in the tuple?
    //std::vector<int> multiplicities_;
    
    int tuple_size_;
    
    int num_prunes_;
    int num_base_cases_;
    // for now, this is an estimate of the number of pairwise distances computed
    long long int num_point_tuples_considered_;
    long long int num_node_tuples_considered_;
    long long int num_pairwise_distances_computed_;

    /////////// functions //////////////////
    
    bool CanPrune_(NodeTuple& nodes);
    
    void BaseCase_(NodeTuple& nodes);
    
    void DepthFirstRecursion_(NodeTuple& nodes);
    
    
    
  public:
    
    GenericNptAlg(std::vector<NptNode*>& trees_in, 
                  TMatcher& matcher_in);
    
    int num_prunes() const;
    
    int num_base_cases() const;
    
    long long int num_point_tuples_considered() const;
    
    long long int num_node_tuples_considered() const;
    
    long long int num_pairwise_distances_computed() const;
    
    // Ensures that the matcher now contains the correct results
    void Compute();
    
    void PrintStats();
    
  }; // class
  
  
} // namespace


#include "generic_npt_alg_impl.hpp"

#endif 

