//
//  efficient_angle_matcher.h
//  contrib_march
//
//  Created by William March on 10/12/12.
//
//

#ifndef _NPOINT_MLPACK_MATCHERS_EFFICIENT_ANGLE_MATCHER_HPP_
#define _NPOINT_MLPACK_MATCHERS_EFFICIENT_ANGLE_MATCHER_HPP_

#include "../infrastructure/node_tuple.hpp"
#include "../efficient_kernels/kernels_npt_cpu.hpp"
#include "matcher_arguments.hpp"

#include <boost/multi_array.hpp>

namespace npoint_mlpack {
  
  class EfficientAngleMatcher
  {
    
  public:
    
    class PartialResult {
      
    private:
      
      int num_r1_;
      int num_theta_;
      
      boost::multi_array<long long int, 2> results_;
      
    public:
      
      PartialResult(std::vector<int>& sizes);
      
      ~PartialResult();
      
      void Reset();
      
      int num_r1() const;
      
      int num_theta() const;
      
      const boost::multi_array<long long int, 2>& results() const;

      boost::multi_array<long long int, 2>& results();

      PartialResult& operator=(const PartialResult& other);
      
      PartialResult& operator+=(const PartialResult& other);
      
      const PartialResult operator+(const PartialResult &other) const;
      
    }; // partial result

  private:
    
    std::vector<arma::mat*> data_mat_list_;
    std::vector<arma::colvec*> data_weights_list_;
    
    
    std::vector<double> short_sides_;
    // the long side is this times the short side
    // should I include more than one of these?
    double long_side_multiplier_;
    
    // This is 0.25 in the thesis (or maybe 0.1?)
    // Note that the bin thickness is this times the scale
    // Does this mean in each dimension?
    double bin_thickness_factor_;
    
    std::vector<double> long_sides_;
    
    // these are in radians
    std::vector<double> thetas_;
    
    // indexed by [value of r1][value of theta]
    //std::vector<std::vector<double> > r3_sides_;
    boost::multi_array<double, 2> r3_sides_;
    
    // upper and lower bound arrays
    // these include the half bandwidth added or subtracted
    std::vector<double> r1_lower_sqr_;
    std::vector<double> r1_upper_sqr_;
    
    std::vector<double> r2_lower_sqr_;
    std::vector<double> r2_upper_sqr_;
    
    // these are indexed by r1 value, then by angle/r3
    //std::vector<std::vector<double> > r3_lower_sqr_;
    //std::vector<std::vector<double> > r3_upper_sqr_;
    
    boost::multi_array<double, 2> r3_lower_sqr_;
    boost::multi_array<double, 2> r3_upper_sqr_;
    
    int tuple_size_;
    int total_num_matchers_;
    
    double** lower_bounds_sqr_ptr_;
    double** upper_bounds_sqr_ptr_;
    
    int num_base_cases_;
    long long int num_pairs_considered_;
    
    int matcher_ind_;
    
    std::vector<int> results_size_;
    
    double longest_possible_side_sqr_;
    double shortest_possible_side_sqr_;
    
    // indexed by [r1][theta]
    boost::multi_array<long long int, 2> results_;
    boost::multi_array<double, 2> weighted_results_;
    
    
    //! This is the lookup table used by the efficient CPU code
    std::vector<unsigned char> satisfiability;
    
    ////////////////////////////
    
    double ComputeR3_(double r1, double r2, double theta);
    
    int TestNodeTupleCarefully_(NptNode* node1,
                                NptNode* node2,
                                NptNode* node3,
                                std::vector<std::vector<int> >& valid_indices);

    
    // The metric class that we'll need
    // In the future, this can be templatized to handle projected coordinates?
    // Also, I think this should get default constructed, which should be fine
    mlpack::metric::SquaredEuclideanDistance metric_;
    
    
  public:
    
    EfficientAngleMatcher(const std::vector<arma::mat*>& data_in,
                          const std::vector<arma::colvec*>& weights_in,
                          MatcherArguments& args);
    
    ~EfficientAngleMatcher();
    
    /////////// multi matcher versions /////////////
    void ComputeBaseCase(NodeTuple& nodes);
    
    // IMPORTANT: this doesn't use the partial result either
    // this means it's not thread safe!!!
    void ComputeBaseCase(NodeTuple& nodes, PartialResult& result);
    
    // IMPORTANT: this doesn't really use the partial result
    // i.e. it directly counts the tuples and stores the result in the matcher
    // So, not thread safe!
    void ComputeBaseCase(NptNode* nodeA,
                         NptNode* nodeB,
                         std::vector<NptNode*>& nodeC_list,
                         PartialResult& result);
    
    bool TestNodeTuple(NodeTuple& nodes);
    
    bool TestNodeTuple(std::vector<double>& min_dists_sqr_,
                       std::vector<double>& max_dists_sqr_);
    
    void AddResult(PartialResult& result);
    
    boost::multi_array<long long int, 2>& results();
    
    boost::multi_array<double, 2>& weighted_results();
    
    double min_dist_sq() const;
    
    double max_dist_sq() const;
    
    void OutputResults();
    
    int tuple_size() const;
    
    int num_base_cases() const;
    
    long long int num_pairs_considered() const;
    
    int matcher_ind() const;
    
    void set_matcher_ind(int new_ind);
    
    std::vector<int>& results_size();
    
  }; // class
  
}


#endif
