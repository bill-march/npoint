/*
 *  angle_matcher.hpp
 *  
 *  Computes three point correlations for a matcher specified by a range of 
 *  values of r1, a factor c such that r2 = c*r1, and a range of angles between 
 *  the sides theta.
 *
 *  Created by William March on 7/26/11.
 *  Copyright 2011 Georgia Institute of Technology. All rights reserved.
 *
 */

#ifndef _NPOINT_MLPACK_MATCHERS_ANGLE_MATCHER_HPP_
#define _NPOINT_MLPACK_MATCHERS_ANGLE_MATCHER_HPP_

#include "boost/multi_array.hpp"
#include "../infrastructure/node_tuple.hpp"
#include "matcher_arguments.hpp"

// Assumptions (for now):
//
// bins might overlap (especially at large values of theta)
// Values of r1 are spaced far enough apart such that a tuple of points will 
// only satisfy one

// IMPORTANT: I think I'm assuming that r2 is enough larger than r1 that there
// isn't any overlap - NOT true any more

namespace npoint_mlpack {

  class AngleMatcher {
    
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
      
      void Increment(int r1_ind, int theta_ind);
      
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
    
    std::vector<int> results_size_;
    
    // indexed by [r1][theta]
    boost::multi_array<long long int, 2> results_;
    boost::multi_array<double, 2> weighted_results_;
    
    
    int tuple_size_;
    int num_base_cases_;
    long long int num_pairs_considered_;
    
    int matcher_ind_;
    
    // This is 0.25 in the thesis (or maybe 0.1?)
    // Note that the bin thickness is this times the scale
    // Does this mean in each dimension?
    double bin_thickness_factor_;
    
    double longest_possible_side_sqr_;
    double shortest_possible_side_sqr_;
    
    ////////////////////////////
    
    double ComputeR3_(double r1, double r2, double theta);
    
    void TestPointTuple_(arma::colvec& vec1, arma::colvec& vec2,
                         arma::colvec& vec3,
                         std::vector<std::vector<int> >& valid_indices);
    
    // The metric class that we'll need
    // In the future, this can be templatized to handle projected coordinates?
    // Also, I think this should get default constructed, which should be fine
    mlpack::metric::SquaredEuclideanDistance metric_;
    
    
  public:
    
    AngleMatcher(const std::vector<arma::mat*>& data_in,
                 const std::vector<arma::colvec*>& weights_in,
                 MatcherArguments& args);
    
    /////////// multi matcher versions /////////////
    void ComputeBaseCase(NodeTuple& nodes);
    
    void ComputeBaseCase(NodeTuple& nodes, PartialResult& result);
    
    void ComputeBaseCase(NptNode* nodeA,
                         NptNode* nodeB,
                         std::vector<NptNode*>& nodeC_list,
                         PartialResult& result);
    
    bool TestNodeTuple(NodeTuple& nodes);
    
    bool TestNodeTuple(std::vector<double>& min_dists_sqr_,
                       std::vector<double>& max_dists_sqr_);
    
    void AddResult(PartialResult& result);
    
    std::vector<int>& results_size();

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

    
  }; // AngleMatcher

} // namespace


#endif