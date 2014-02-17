//
//  3_point_single_matcher.hpp
//  contrib_march
//
//  Created by William March on 7/15/12.
//  Copyright (c) 2012 Georgia Institute of Technology. All rights reserved.
//

#ifndef _MLPACK_NPOINT_MATCHERS_3_POINT_SINGLE_MATCHER_HPP_
#define _MLPACK_NPOINT_MATCHERS_3_POINT_SINGLE_MATCHER_HPP_

#include "../infrastructure/node_tuple.hpp"
#include "../infrastructure/permutations.hpp"
#include "matcher_arguments.hpp"

namespace npoint_mlpack {

  class ThreePointSingleMatcher {
    
  public:
    
    class PartialResult {
      
      long long int results_;
      
    public:
      
      PartialResult(const std::vector<int>& results_size);
      
      ~PartialResult();
      
      void Reset();
      
      long long int results() const;
      
      void Increment();
      
      PartialResult& operator=(const PartialResult& other);
      
      PartialResult& operator+=(const PartialResult& other);
      
      const PartialResult operator+(const PartialResult &other) const;
      
    }; // partial result

  private:
    
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
    
    long long int results_;
    double weighted_results_;
    
    
    int num_base_cases_;
    
    // n!
    int num_permutations_;
    
    // the metric class (with one static method)
    // maybe I can update this to handle projected coordinates
    // this should get default constructed, which is fine
    mlpack::metric::SquaredEuclideanDistance metric_;
    
    long long int num_pairs_considered_;

    int matcher_ind_;
    
    std::vector<int> results_size_;
    
    ////////////// functions ///////////////////
    
    int GetPermIndex_(int perm_index, int pt_index);
    
    bool TestHrectPair_(double min_dist_sq,
                        double max_dist_sq,
                        int tuple_ind_1, int tuple_ind_2,
                        std::vector<char>& permutation_ok);
    
    bool TestPermutation_(size_t perm_ind,
                          double dist01,
                          double dist02,
                          double dist12);
    
  public:
    
    ThreePointSingleMatcher(const std::vector<arma::mat*>& data_in, 
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

    void ComputeBaseCase(NptNode* nodeA,
                         NptNode* nodeB,
                         std::vector<NptNode*>& nodeC_list,
                         PartialResult& result);
    
    void AddResult(PartialResult& result);
    
    const std::vector<int>& results_size() const;
    
    long long int results() const;
    
    double weighted_results() const;
    
    int tuple_size() const;
    
    int num_permutations() const;
    
    //const std::vector<arma::mat*>& data_mat_list() const;
    
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
    
    void OutputResults();

    
    
    
  }; // class

} // namespace



#endif
