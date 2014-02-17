/**
 * @file efficient_4pt_matcher.hpp
 * @author Bill March (march@gatech.edu)
 *
 * Header for 4-point correlation matcher that uses the efficient CPU kernels
 * from the supercomputing paper.
 */

#ifndef _NPOINT_MLPACK_MATCHERS_EFFICIENT_4PT_MATCHER_HPP_
#define _NPOINT_MLPACK_MATCHERS_EFFICIENT_4PT_MATCHER_HPP_

#include "../infrastructure/permutations.hpp"
#include "../infrastructure/node_tuple.hpp"
#include "../efficient_kernels/kernel_4pt_cpu.hpp"
#include "matcher_arguments.hpp"

#include <boost/multi_array.hpp>

namespace npoint_mlpack {
  
  class Efficient4ptMatcher {
    
  public:
    
    class PartialResult {
      
      long long int results_;
      
    public:
      
      PartialResult(std::vector<int>& results_size);
      
      ~PartialResult();
      
      void Reset();
      
      long long int results() const;
      
      PartialResult& operator=(const PartialResult& other);
      
      PartialResult& operator+=(const PartialResult& other);
      
      const PartialResult operator+(const PartialResult &other) const;
      
      void AddResult(long long int);
      
    }; // partial result
    
  private:
    
    //! contains pointers to the data/weights for this computation
    //! same as in the other matchers
    std::vector<arma::mat*> data_mat_list_;
    std::vector<arma::colvec*> data_weights_list_;
    
    //! The upper and lower bounds, stored in an upper triangular matrix.
    //! They're squared to avoid dealing with square roots of distances
    arma::mat lower_bounds_sqr_;
    arma::mat upper_bounds_sqr_;
    
    //! We store the upper and lower bounds as a linear array of doubles to
    //! pass to the efficient kernel code
    double lower_bounds_sqr_ptr_[6];
    double upper_bounds_sqr_ptr_[6];
    
    double min_matcher_dist_sq_, max_matcher_dist_sq_;
    
    //! The results
    long long int results_;
    double weighted_results_;
    
    //! n
    int tuple_size_;
    
    //! The number of times the base case computation is called.
    int num_base_cases_;
    
    //! this is the index of the matcher in a multi-matcher scheme
    //! negative if it doesn't apply
    int matcher_ind_;
    
    std::vector<int> results_size_;
    
    std::vector<unsigned char> satisfiability;
    
    std::vector<std::vector<int> > node_sets_;
    
    int OvercountingFactor_(int begin0, int begin1, int begin2, int begin3);
    
  public:
    
    /**
     * Constructor: takes same arguments as the other matchers.
     *
     * @param data_in The list of pointers to the data matrices.
     * @param weights_in The list of pointers to the data weights.  NOT YET
     * IMPLEMENTED.
     * @param matcher_args The MatcherArguments class for this computation.
     */
    Efficient4ptMatcher(const std::vector<arma::mat*>& data_in,
                        const std::vector<arma::colvec*>& weights_in,
                        const MatcherArguments& matcher_args);
    
    /**
     * Shouldn't need to free anything, since it's all static or owned by
     * another class.
     */
    ~Efficient4ptMatcher();
    
    /**
     * A (sort-of) copy constructor used for the TBB parallel version.
     *
     * @param other The EfficientCpuMatcher to be copied.
     * @param is_copy Indicates if we should copy the result of other, or
     * start with a zero result.
     */
    Efficient4ptMatcher(const Efficient4ptMatcher& other, bool is_copy = false);
    
    
    /**
     * In the TBB parallel version, we need to be able to combine the results
     * of subproblems.
     *
     * @param left_matcher One of the matchers to sum together.
     * @param right_matcher The other matcher to sum.
     */
    void SumResults(const Efficient4ptMatcher& left_matcher,
                    const Efficient4ptMatcher& right_matcher);
    
    /**
     * Tests if any tuples from the set of nodes could possibly satisfy the
     * matcher.
     *
     * Returns true if the tuple might satisfy the matcher (i.e. CANNOT be
     * pruned.
     *
     * @param nodes The NodeTuple to be tested.
     */
    bool TestNodeTuple(NodeTuple& nodes);
    
    /**
     * Test if any tuples from nodes with the given pairwise distance bounds
     * can possibly satisfy the matcher.
     *
     * Returns true if the tuple might satisfy the matcher (i.e. CANNOT be
     * pruned.
     *
     * @param min_dists_sqr_ The lower bound distances squared.
     * @param max_dists_sqr_ The upper bounds squared.  The ordering isn't
     * important, but must be the same for the two vectors.
     */
    bool TestNodeTuple(std::vector<double>& min_dists_sqr_,
                       std::vector<double>& max_dists_sqr_);
    
    /**
     * Counts the number of tuples in the NodeTuple that satisfy the matcher.
     * Stores the result in the private results variable.
     *
     * @param nodes The NodeTuple to be computed.
     */
    void ComputeBaseCase(NodeTuple& nodes);
    
    /**
     * Counts the number of tuples in the NodeTuple that satisfy the matcher.
     * Stores the result in the PartialResult and doesn't touch the private
     * results variable.
     *
     * @param nodes The NodeTuple to be computed.
     * @param result The PartialResult to store the results.
     */
    void ComputeBaseCase(NodeTuple& nodes,
                         PartialResult& result);
    
    /**
     * Counts the number of tuples in the node tuple (nodeA, nodeB, C), where
     * C is the union of all nodes in nodeC_list.  Used in pairwise traversals.
     * Stores the result in the PartialResult and doesn't touch the private
     * results variable.
     *
     * @param nodeA Pointer to the first NptNode in the tuple.
     * @param nodeB Pointer to the second NptNode in the tuple.
     * @param nodeC_list List of pointers to the third NptNode in each tuple.
     * @param result The PartialResult to store the results.
     */
    void ComputeBaseCase(NptNode* nodeA,
                         NptNode* nodeB,
                         std::vector<NptNode*>& nodeC_list,
                         PartialResult& result);
    
    
    /**
     * Counts the number of tuples in the node tuple (nodeA, nodeB, C), where
     * C is the union of all nodes in nodeC_list.  Used in pairwise traversals.
     * Stores the result in the private results_ variable.
     *
     * @param nodeA Pointer to the first NptNode in the tuple.
     * @param nodeB Pointer to the second NptNode in the tuple.
     * @param nodeC_list List of pointers to the third NptNode in each tuple.
     */
    void ComputeBaseCase(NptNode* nodeA,
                         NptNode* nodeB,
                         std::vector<NptNode*>& nodeC_list);
    
    /**
     * Adds the PartialResult to the private results variable.
     *
     * @param result The PartialResult to store the results.
     */
    void AddResult(PartialResult& result);
    
    long long int results() const;
    
    double weighted_results() const;
    
    int tuple_size() const;
    
    int num_permutations() const;
    
    const std::vector<arma::mat*>& data_mat_list() const;
    
    const std::vector<arma::colvec*>& data_weights_list() const;
    
    const arma::mat& lower_bounds_sqr() const;
    
    const arma::mat& upper_bounds_sqr() const;
    
    long long int num_base_cases() const;
    
    long long int num_pairs_considered() const;
    
    double min_dist_sq() const;
    
    double max_dist_sq() const;
    
    int matcher_ind() const;
    
    void set_matcher_ind(int new_ind);
    
    const std::vector<std::vector<int> >& node_sets() const;
    
    std::vector<int>& results_size();
    
    /**
     * Prints the results to the command line along with some performance
     * statistics.
     */
    void OutputResults();
    
  }; // class
  
} // namespace


#endif
