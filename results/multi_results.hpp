//
//  multi_results.hpp
//  
//
//  Created by William March on 1/18/12.
//  Copyright (c) 2012 Georgia Institute of Technology. All rights reserved.
//

#ifndef _MLPACK_NPOINT_MLPACK_RESULTS_MULTI_RESULTS_HPP_
#define _MLPACK_NPOINT_MLPACK_RESULTS_MULTI_RESULTS_HPP_

#include "../matchers/multi_matcher.hpp"
#include "../matchers/single_matcher.hpp"
#include "../matchers/efficient_cpu_matcher.hpp"
#include "../matchers/3_point_single_matcher.hpp"
#include "../matchers/efficient_multi_matcher.hpp"
#include "../matchers/unordered_multi_matcher.hpp"
#include "../matchers/unordered_efficient_multi_matcher.hpp"
#include "../matchers/matcher_arguments.hpp"
#include <boost/multi_array.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

namespace npoint_mlpack {
  
  class MultiResults {
    
    // for serialization
    friend class boost::serialization::access;
    
  private:
    
    int num_resampling_regions_;
    int tuple_size_;
    int total_num_matchers_;
    
    // first index: resampling region
    // second index: num_random 
    // third index: to access the matcher (i,j,k), go to 
    // i + num_bands[0]*j + . . . 
    
    // what I really need is an array of dimension 2 + (n choose 2)
    // the first two are for region and num_random, the rest to indicate which
    // matcher we're talking about
    
    // previously, I was using a single array and indexing into it
    // this could still work, but how to process partial results from the matcher?
    boost::multi_array<long long int, 3> results_;
    boost::multi_array<double, 3> weighted_results_;
    
    std::vector<long long int> RRR_result_;
    std::vector<double> weighted_RRR_result_;
    
    // given a vector with the indices into the matcher, find the right 
    // index for the third dimension of results_
    // perm_locations needs to have length (n choose 2)
    //size_t FindResultsInd_(const std::vector<size_t>& perm_locations);
    
    //void FindMatcherInd_(size_t loc, std::vector<size_t>& result);

    void AddRandomResult_(std::vector<long long int>& results);
    
    void AddResult_(int region_id, int num_random, 
                    std::vector<long long int>& results);
    
    void ProcessSingleResult_(std::vector<int>& region_ids, 
                              int num_random,
                              bool is_efficient, 
                              long long int result,
                              int matcher_ind);
    
  public:
    
    MultiResults();
    
    MultiResults(MatcherArguments& args, int num_regions);
    
    // copy constructor
    MultiResults(const MultiResults& other);
    
    MultiResults& operator=(const MultiResults& other);
    
    template<class Archive>
    void serialize(Archive & ar, const unsigned int /*version*/)
    {
      
      ar & num_resampling_regions_;
      ar & tuple_size_;
      ar & total_num_matchers_;
      ar & results_;
      ar & weighted_results_;
      ar & RRR_result_;
      ar & weighted_RRR_result_;
      
    } // serialization for boost mpi stuff
    
    void ProcessResults(std::vector<int>& region_ids, 
                        int num_random,
                        bool is_efficient, 
                        MultiMatcher& matcher);
    
    void ProcessResults(std::vector<int>& region_ids,
                        int num_random,
                        bool is_efficient,
                        UnorderedMultiMatcher& matcher);
    
    void ProcessResults(std::vector<int>& region_ids,
                        int num_random,
                        bool is_efficient,
                        UnorderedEfficientMultiMatcher& matcher);
    
    // need to just overload Process Results here for the drivers
    void ProcessResults(std::vector<int>& region_ids,
                        int num_random,
                        bool is_efficient, 
                        SingleMatcher& matcher);
    
    void ProcessResults(std::vector<int>& region_ids,
                        int num_random,
                        bool is_efficient, 
                        EfficientCpuMatcher& matcher);
    
    void ProcessResults(std::vector<int>& region_ids,
                        int num_random,
                        bool is_efficient, 
                        ThreePointSingleMatcher& matcher);
    
    void ProcessResults(std::vector<int>& region_ids,
                        int num_random,
                        bool is_efficient,
                        EfficientMultiMatcher& matcher);
    
    void PrintResults();
    
    void PrintResults(std::ostream& stream);
    
    int num_resampling_regions() const;

    int tuple_size() const;
    
    int total_num_matchers() const;
    
    
    boost::multi_array<long long int, 3>& results();
    
    boost::multi_array<double, 3>& weighted_results();
    
    std::vector<long long int>& RRR_result();
    
    std::vector<double>& weighted_RRR_result();
    
    boost::multi_array<long long int, 3> results() const;
    
    boost::multi_array<double, 3> weighted_results() const;
    
    std::vector<long long int> RRR_result() const;
    
    std::vector<double> weighted_RRR_result() const;
    
    bool operator==(const MultiResults &other) const;

    bool operator!=(const MultiResults &other) const;
    
    // Adds another result to this one
    void AddResults(const MultiResults& other);
    
    MultiResults operator+(const MultiResults& other) const;
    
  }; // class

}// namespace

#endif
