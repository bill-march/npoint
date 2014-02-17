//
//  unordered_efficient_multi_matcher.hpp
//  contrib_march
//
//  Created by William March on 10/19/12.
//
//

#ifndef _NPOINT_MLPACK_MATCHERS_UNORDERED_EFFICIENT_MULTI_MATCHER_HPP_
#define _NPOINT_MLPACK_MATCHERS_UNORDERED_EFFICIENT_MULTI_MATCHER_HPP_

#include "../infrastructure/node_tuple.hpp"
#include "matcher_arguments.hpp"
#include "../efficient_kernels/kernel_2pt_cpu.hpp"

namespace npoint_mlpack {
  
  /**
   * A multi-matcher that just handles a list of matcher matrices with no
   * assumed structure.
   */
  class UnorderedEfficientMultiMatcher {
    
  public:
    
    class PartialResult {
      
      int num_results_;
      
      std::vector<long long int> results_;
      
    public:
      
      PartialResult(const std::vector<int>& results_size);
      
      ~PartialResult();
      
      void Reset();
      
      int num_results() const;
      
      const std::vector<long long int>& results() const;
      
      std::vector<long long int>& results();
      
      PartialResult& operator=(const PartialResult& other);
      
      PartialResult& operator+=(const PartialResult& other);
      
      const PartialResult operator+(const PartialResult &other) const;
      
    }; // partial result
    
  private:
    
    //! The data and weights for use in base cases
    std::vector<arma::mat*> data_mat_list_;
    std::vector<arma::colvec*> data_weights_list_;
    
    //! n
    int tuple_size_;
    int tuple_size_choose_2_;
    
    //! Lower and upper bounds
    std::vector<arma::mat> lower_matcher_list_;
    std::vector<arma::mat> upper_matcher_list_;
    
    //! The total number of matchers to compute
    int total_matchers_;
    
    //! The counts and weighted counts for each matcher
    //! Indexed by index in the list of matchers
    std::vector<long long int> results_;
    std::vector<double> weighted_results_;
    
    //! The upper and lower bounds (squared) of the various matchers
    //! entry i,j is the lower (upper) bound of the jth matcher value in
    //! dimension i
    std::vector<std::vector<double> > matcher_lower_bounds_sq_;
    std::vector<std::vector<double> > matcher_upper_bounds_sq_;
    
    //! Sorted upper and lower bounds used for pruning
    std::vector<double> sorted_upper_bounds_sq_;
    std::vector<double> sorted_lower_bounds_sq_;
    
    double** lower_bounds_ptr_;
    double** upper_bounds_ptr_;
    
    //! metric object
    mlpack::metric::SquaredEuclideanDistance metric_;
    
    //! Needs to know its arguments in order to generate single matchers
    MatcherArguments matcher_args_;
    
    //! Matcher ind used for some drivers.  Won't matter with multi matcher
    int matcher_ind_;
    
    std::vector<int> results_size_;
    
    std::vector<unsigned char> satisfiability;
    
  public:
    
    UnorderedEfficientMultiMatcher(const std::vector<arma::mat*>& data_in,
                          const std::vector<arma::colvec*>& weights_in,
                          MatcherArguments& args);
    
    ~UnorderedEfficientMultiMatcher();
    
    void ComputeBaseCase(NodeTuple& nodes);
    
    // IMPORTANT: this doesn't use the partial result either
    // this means it's not thread safe!!!
    void ComputeBaseCase(NodeTuple& nodes, PartialResult& result);
    
    // IMPORTANT: this doesn't really use the partial result
    // i.e. it directly counts the tuples and stores the result in the matcher
    void ComputeBaseCase(NptNode* nodeA,
                         NptNode* nodeB,
                         std::vector<NptNode*>& nodeC_list,
                         PartialResult& result);
    
    bool TestNodeTuple(NodeTuple& nodes);
    
    bool TestNodeTuple(std::vector<double>& min_dists_sqr_,
                       std::vector<double>& max_dists_sqr_);
    
    void AddResult(PartialResult& result);
    
    std::vector<long long int>& results();
    
    std::vector<double>& weighted_results();
    
    int num_permutations();
    
    double matcher_dists(size_t i, size_t j);
    
    double min_dist_sq() const;
    
    double max_dist_sq() const;
    
    void OutputResults();
    
    int tuple_size() const;
    
    int num_permutations() const;
    
    int num_base_cases() const;
    
    long long int num_pairs_considered() const;
    
    int matcher_ind() const;
    
    void set_matcher_ind(int new_ind);
    
    std::vector<int>& results_size();
    
  }; // class
  
} // namespace

#endif /* defined(__contrib_march__efficient_multi_matcher__) */
