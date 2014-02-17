//
//  efficient_multi_matcher.hpp
//  contrib_march
//
//  Created by William March on 10/19/12.
//
//

#ifndef _NPOINT_MLPACK_MATCHERS_EFFICIENT_MULTI_MATCHER_HPP_
#define _NPOINT_MLPACK_MATCHERS_EFFICIENT_MULTI_MATCHER_HPP_

// Idea: for each of the (n choose 2) distances in the matcher, the user will
// specify a range and number of distances to compute for


// IMPORTANT: assuming that all dimensions have the same thickness
// assuming that matcher values +- band don't overlap within a dimension

#include "../infrastructure/node_tuple.hpp"
#include "../infrastructure/permutations.hpp"
#include "matcher_arguments.hpp"
#include "multi_matcher_generator.hpp"
#include "../efficient_kernels/kernels_npt_cpu.hpp"

namespace npoint_mlpack {
  
  /**
   * A matcher that simultaneously handles multiple matchers, laid out in a
   * regular pattern.  See March et al. in KDD 2012 for more details.
   * Uses the efficient 3-point kernel.
   *
   * For each of the (n choose 2) distances in the matcher, the inputs will
   * specify a lower and upper bound and a number of bins.  The matchers then
   * consist of all combinations of bins.
   *
   * IMPORTANT: we assume that all dimensions use the same thickness parameter,
   * although this shouldn't be too hard to extend if necessary.
   *
   * MORE IMPORTANT: we assume that no bins overlap within a single dimension.
   * The current input method won't even allow the user to specify this.  If
   * we need to change this, it will take a lot of work.
   */
  class EfficientMultiMatcher {
    
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
    
    //! The total number of matchers to compute
    int total_matchers_;
    
    //! n
    int tuple_size_;
    
    //! (n choose 2)
    int tuple_size_choose_2_;
    
    //! The counts and weighted counts for each matcher
    //! indexed by: matcher_ind_0 + num_bands[0]*matcher_ind_1 + . . .
    std::vector<long long int> results_;
    std::vector<double> weighted_results_;
    
    //! The minimum matcher value in each dimension
    std::vector<double> min_bands_;
    
    //! The maximum matcher value in each dimension
    std::vector<double> max_bands_;
    //! The number of bins in each dimension
    std::vector<int> num_bands_;
    
    //! The step size in between matcher dists in each of the (n choose 2)
    //! dimensions
    //! This is just (max_bands_ - min_bands_) / num_bands_
    std::vector<double> band_steps_;
    
    //! The upper and lower bounds (squared) of the various matchers
    //! entry i,j is the lower (upper) bound of the jth matcher value in
    //! dimension i
    std::vector<std::vector<double> > matcher_lower_bounds_sq_;
    std::vector<std::vector<double> > matcher_upper_bounds_sq_;
    
    //! Sorted upper and lower bounds used for pruning
    std::vector<double> sorted_upper_bounds_sq_;
    std::vector<double> sorted_lower_bounds_sq_;
    
    //! Permutations class
    Permutations perms_;
    //! n!
    size_t num_permutations_;
    
    //! metric object
    mlpack::metric::SquaredEuclideanDistance metric_;
    
    //! Matcher ind used for some drivers.  Won't matter with multi matcher
    int matcher_ind_;
    
    std::vector<int> results_size_;
    
    double** lower_bounds_sqr_ptr_;
    double** upper_bounds_sqr_ptr_;
    
    std::vector<unsigned char> satisfiability;
    
    bool do_off_diagonal_;
    
  public:
    
    EfficientMultiMatcher(const std::vector<arma::mat*>& data_in,
                 const std::vector<arma::colvec*>& weights_in,
                 MatcherArguments& args);
    
    ~EfficientMultiMatcher();
    
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
