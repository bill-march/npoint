/*
 *  single_matcher.hPP
 *  
 *
 *  Created by William March on 2/7/11.
 *  Copyright 2011 Georgia Institute of Technology. All rights reserved.
 *
 *
 *  A single matcher.  This will use the same matrix formulation as before.
 *
 */

/*
 *  Takes in a single matcher, specified either by upper and lower bound
 *  matrices or a matrix of distances and a bandwidth
 *  Called by the generic algorithm to do prune checks 
 *  and base cases.
 *
 *  Needs access to the data.  Right now, I'm using a reference to a 
 *  vector of references to matrices.  I don't think this works.
 *  Other options: 
 *  - aliasing (like the neighbor search code) 
 *  - pointers to memory that's held by the resampling class.
 */


#ifndef __MLPACK_METHODS_NPOINT_MATCHERS_SINGLE_MATCHER_HPP
#define __MLPACK_METHODS_NPOINT_MATCHERS_SINGLE_MATCHER_HPP

#include "../infrastructure/permutations.hpp"
#include "../infrastructure/node_tuple.hpp"
#include "matcher_arguments.hpp"

namespace npoint_mlpack {
  
  class SingleMatcher {
    
  public:
    
    class PartialResult {
      
      long long int results_;
      
    public:
      
      PartialResult(const std::vector<int>& result_size);
        
      ~PartialResult();
      
      void Reset();
      
      long long int results() const;
      
      void IncrementResult();
      
      PartialResult& operator=(const PartialResult& other);
      
      PartialResult& operator+=(const PartialResult& other);
      
      const PartialResult operator+(const PartialResult &other) const;
      
    }; // partial result
    
  private:
    
    
    // the data and weights
    std::vector<arma::mat*> data_mat_list_;
    std::vector<arma::colvec*> data_weights_list_;
    
    // n
    int tuple_size_;
    
    
    // stores the permutations, make sure to always reuse it instead of making
    // more
    Permutations perms_;
    
    // The upper and lower bounds for the matcher, stored in an upper triangular
    // matrix.  They're squared to avoid dealing with square roots of distancess
    arma::mat lower_bounds_sqr_;
    arma::mat upper_bounds_sqr_;
    
    int results_;
    double weighted_results_;
    
    int num_base_cases_;
    
    // n!
    int num_permutations_;
    
    // the metric class (with one static method)
    // maybe I can update this to handle projected coordinates
    // this should get default constructed, which is fine
    mlpack::metric::SquaredEuclideanDistance metric_;
    
    long long int num_pairs_considered_;
    
    // this is the index of the matcher in a multi-matcher scheme
    // negative if it doesn't apply
    int matcher_ind_;
    
    std::vector<int> results_size_;
    
    /////////////////// functions ////////////////////////
    
    /**
     * Just accesses the Permutations class
     */
    int GetPermIndex_(int perm_index, int pt_index);
    
    
    /**
     * Helper function for checking points or bounds.
     */
    bool CheckDistances_(double dist_sq, int ind1, int ind2);
    
    
    bool TestPointPair_(double dist_sq, int tuple_ind_1, int tuple_ind_2,
                        std::vector<char>& permutation_ok);
    
    bool TestHrectPair_(double min_dist_sq,
                        double max_dist_sq,
                        int tuple_ind_1, int tuple_ind_2,
                        std::vector<char>& permutation_ok);
    
    void BaseCaseHelper_(NodeTuple& nodes,
                         std::vector<char>& permutation_ok,
                         std::vector<int>& points_in_tuple,
                         int k,
                         PartialResult& result);
    
    
  public:
    
    // constructor
    SingleMatcher(const std::vector<arma::mat*>& data_in, 
                  const std::vector<arma::colvec*>& weights_in,
                  const MatcherArguments& matcher_args);
    
    // copy constructor
    // if is_copy is true, then copy the results from other
    // otherwise, set results to 0
    //SingleMatcher(const SingleMatcher& other, bool is_copy);
    
    //void SumResults(const SingleMatcher& left_matcher, 
    //                const SingleMatcher& right_matcher);
    
    bool TestNodeTuple(NodeTuple& nodes);
    
    // Important: these are kept in the same order, since they have to match
    // up
    bool TestNodeTuple(std::vector<double>& min_dists_sqr_,
                       std::vector<double>& max_dists_sqr_);
    
    void ComputeBaseCase(NodeTuple& nodes);
    
    void ComputeBaseCase(NodeTuple& nodes, PartialResult& result);
    
    // IMPORTANT: this doesn't really use the partial result 
    // i.e. it directly counts the tuples and stores the result in the matcher
    void ComputeBaseCase(NptNode* nodeA,
                         NptNode* nodeB,
                         std::vector<NptNode*>& nodeC_list,
                         PartialResult& result);
    
    void AddResult(PartialResult& result);
    
    //void ComputeBaseCase(NodeTuple& nodes, EfficientCpuPartialResult& result);
    
    
    int results() const;

    double weighted_results() const;
    
    int tuple_size() const;
    
    int num_permutations() const;
    
    const std::vector<arma::colvec*>& data_weights_list() const;
    
    const Permutations& perms() const;
    
    const arma::mat& lower_bounds_sqr() const;
    
    const arma::mat& upper_bounds_sqr() const;
    
    int num_base_cases() const;
    
    long long int num_pairs_considered() const;
    
    double min_dist_sq() const;
    
    double max_dist_sq() const;
    
    int matcher_ind() const;
    
    void set_matcher_ind(int new_ind);
    
    const std::vector<int>& results_size() const;
    
    void OutputResults();
    
  }; // class
  
} // namespace

#endif
