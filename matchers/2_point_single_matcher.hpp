//
//  2_point_single_matcher.hpp
//  contrib_march
//
//  Created by William March on 6/25/13.
//
//

#ifndef __contrib_march____point_single_matcher__
#define __contrib_march____point_single_matcher__


#include "../infrastructure/node_tuple.hpp"
#include "../infrastructure/permutations.hpp"
#include "matcher_arguments.hpp"


namespace npoint_mlpack {
  
  class TwoPointSingleMatcher {
    
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
    
    double min_matcher_dist_sqr_, max_matcher_dist_sqr_;
    
    long long int results_;
    double weighted_results_;
    
    
    long long int num_base_cases_;
    
    // n!
    int num_permutations_;
    
    // the metric class (with one static method)
    // maybe I can update this to handle projected coordinates
    // this should get default constructed, which is fine
    mlpack::metric::SquaredEuclideanDistance metric_;
    
    long long int num_pairs_considered_;
    
    int matcher_ind_;
    
    std::vector<int> results_size_;
  
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

    
    TwoPointSingleMatcher(const std::vector<arma::mat*>& data_in,
                          const std::vector<arma::colvec*>& weights_in,
                          const MatcherArguments& matcher_args);
    
    bool TestNodeTuple(NodeTuple& nodes);
    
    // Ordering doesn't matter here, there's only one distance in each
    bool TestNodeTuple(std::vector<double>& min_dists_sqr_,
                       std::vector<double>& max_dists_sqr_);
    
    void ComputeBaseCase(NodeTuple& nodes);
    
    void ComputeBaseCase(NodeTuple& nodes, PartialResult& result);
    
    /*
    void ComputeBaseCase(NptNode* nodeA,
                         NptNode* nodeB,
                         std::vector<NptNode*>& nodeC_list,
                         PartialResult& result);
    */
    
    void AddResult(PartialResult& result);
    
    const std::vector<int>& results_size() const;
    
    long long int results() const;
    
    double weighted_results() const;
    
    int tuple_size() const;
    
    int num_permutations() const;
    
    const std::vector<arma::colvec*>& data_weights_list() const;
    
    const Permutations& perms() const;
    
    const arma::mat& lower_bounds_sqr() const;
    
    const arma::mat& upper_bounds_sqr() const;
    
    long long int num_base_cases() const;
    
    long long int num_pairs_considered() const;
    
    double min_dist_sq() const;
    
    double max_dist_sq() const;
    
    int matcher_ind() const;
    
    void set_matcher_ind(int new_ind);
    
    void OutputResults();
        
  }; // class
  
  
} // namespace


#endif /* defined(__contrib_march____point_single_matcher__) */
