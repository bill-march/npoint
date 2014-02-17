/*
 *  single_results.hpp
 *  
 *
 *  Created by William March on 9/7/11.
 *  Copyright 2011 Georgia Institute of Technology. All rights reserved.
 *
 */

#ifndef __MLPACK_METHODS_NPOINT_RESULTS_SINGLE_RESULTS_HPP
#define __MLPACK_METHODS_NPOINT_RESULTS_SINGLE_RESULTS_HPP

#include "../matchers/single_matcher.hpp"
#include "../matchers/efficient_cpu_matcher.hpp"
#include "../matchers/efficient_2pt_matcher.hpp"
#include "../matchers/efficient_4pt_matcher.hpp"
#include "../matchers/3_point_single_matcher.hpp"
#include "boost/multi_array.hpp"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

/*
 *  Knows the structure of the results and processes the intermediate 
 *  results from the matcher
 *  This is where a result from n tree nodes gets processed into the 
 *  correct jackknife results
 * 
 *  This takes the matcher and info on which computation(s) were run and 
 *  puts the results in the right place.
 *
 */

namespace npoint_mlpack {
  
  class SingleResults {

    // for serialization
    friend class boost::serialization::access;
    
  private:
    
    int tuple_size_;
    
    arma::mat lower_bounds_, upper_bounds_;
    
    // indexed by [resampling_region][num_random]
    boost::multi_array<long long int, 2> results_;
    boost::multi_array<double, 2> weighted_results_;
    
    long long int RRR_result_;
    double RRR_weighted_result_;
    
    int num_regions_;

    void AddResult_(int region_ind, int num_random, int result);
    
    void AddResult_(int region_ind, int num_random, long long int result);
    
    void AddRandomResult_(int result);

    void AddRandomResult_(long long int result);

    
    
  public:
    
    // Need this one for boost mpi stuff
    SingleResults();
    
    SingleResults(MatcherArguments& args, int num_regions);
    
    // copy constructor
    SingleResults(const SingleResults& other);
    
    SingleResults& operator=(const SingleResults& other);
    
    template<class Archive>
    void serialize(Archive & ar, const unsigned int /*version*/)
    {
      
      ar & tuple_size_;
      ar & lower_bounds_;
      ar & upper_bounds_;
      ar & results_;
      ar & weighted_results_;
      ar & RRR_result_;
      ar & RRR_weighted_result_;
      ar & num_regions_;
      
    } // serialization for boost mpi stuff

    void ProcessResults(std::vector<int>& region_ids, int num_random,
                        bool is_efficient,
                        long long int this_result);
    
    // I need all of these for now, can probably get rid of them with more
    // templates . . . sigh
    void ProcessResults(std::vector<int>& region_ids, int num_random,
                        bool is_efficient,
                        SingleMatcher& matcher);
    
    void ProcessResults(std::vector<int>& region_ids, int num_random,
                        bool is_efficient,
                        EfficientCpuMatcher& matcher);
    
    void ProcessResults(std::vector<int>& region_ids, int num_random,
                        bool is_efficient,
                        ThreePointSingleMatcher& matcher);
   
    void ProcessResults(std::vector<int>& region_ids, int num_random,
                        bool is_efficient,
                        Efficient2ptMatcher& matcher);
    
    void ProcessResults(std::vector<int>& region_ids, int num_random,
                        bool is_efficient,
                        Efficient4ptMatcher& matcher);
    
    void PrintResults(std::ostream& stream);
    
    void PrintResults();
    
    boost::multi_array<long long int, 2>& results();
    
    boost::multi_array<double, 2>& weighted_results();
    
    boost::multi_array<long long int, 2> results() const;
    
    boost::multi_array<double, 2> weighted_results() const;
    
    long long int RRR_result() const;

    long long int RRR_weighted_result() const;

    // Adds another result to this one
    void AddResults(const SingleResults& other);
    
    
    SingleResults operator+(const SingleResults& other) const;
    
    
    int tuple_size() const;
    
    arma::mat& lower_bounds();
    
    arma::mat& upper_bounds();
    
    arma::mat lower_bounds() const;
    
    arma::mat upper_bounds() const;
    
    int num_regions() const;
    
    bool operator==(const SingleResults &other) const;
    
    bool operator!=(const SingleResults &other) const;
    
  }; // class
  
} // namespace


#endif