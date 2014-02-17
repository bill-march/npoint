/*
 *  angle_results.hpp
 *  
 *
 *  Created by William March on 7/27/11.
 *  Copyright 2011 Georgia Institute of Technology. All rights reserved.
 *
 */

#ifndef _MLPACK_NPOINT_RESULTS_ANGLE_RESULTS_HPP_
#define _MLPACK_NPOINT_RESULTS_ANGLE_RESULTS_HPP_

#include "../matchers/angle_matcher.hpp"
#include "../matchers/single_matcher.hpp"
#include "../matchers/efficient_cpu_matcher.hpp"
#include "../matchers/3_point_single_matcher.hpp"
#include "../matchers/efficient_angle_matcher.hpp"
#include "../matchers/matcher_arguments.hpp"
#include <boost/multi_array.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

/*
 *  Knows the structure of the results and processes the intermediate 
 *  results from the matcher
 *  This is where a result from n tree nodes gets processed into the 
 *  correct jackknife results
 * 
 *  The generic resampling class handles giving the right stuff to the matcher
 *  and running the algorithm (through the generic algorithm).
 *  This takes the matcher and info on which computation(s) were run and 
 *  puts the results in the right place.
 */

namespace npoint_mlpack {

  class AngleResults {
    
    friend class boost::serialization::access;
    
  private:
    
    const static int tuple_size_ = 3;
    
    int num_regions_;
    int num_r1_;
    int num_theta_;
    
    std::vector<double> r1_vec_;
    std::vector<double> theta_vec_;
    
    // indexed by [resampling_region][num_random][r1][theta]
    boost::multi_array<long long int, 4> results_;
    boost::multi_array<double, 4> weighted_results_;
    
    // indexed by [r1][theta]
    boost::multi_array<long long int, 2> RRR_result_;
    boost::multi_array<double, 2> RRR_weighted_result_;
    
    
    ///////////////////////////////
    
    
    void AddResult_(int region_id, int num_random, 
                    boost::multi_array<long long int, 2>& partial_result);
    
    
    void AddRandomResult_(boost::multi_array<long long int, 2>& partial_result);
    
  public:
    
    //needed for boost mpi
    AngleResults();
    
    AngleResults(MatcherArguments& args, int num_regions);
    
    // copy constructor
    AngleResults(const AngleResults& other);
    
    AngleResults& operator=(const AngleResults& other);
    
    template<class Archive>
    void serialize(Archive & ar, const unsigned int /*version*/)
    {
      
      ar & num_regions_;
      ar & num_r1_;
      ar & num_theta_;
      ar & r1_vec_;
      ar & theta_vec_;
      //ar & tuple_size_;
      ar & results_;
      ar & weighted_results_;
      ar & RRR_result_;
      ar & RRR_weighted_result_;
      
    } // serialization for boost mpi stuff
    
    int num_regions();
    int num_r1();
    int num_theta();
    
    int num_regions() const;
    int num_r1() const;
    int num_theta() const;
    
    std::vector<double>& r1_vec();
    std::vector<double>& theta_vec();
    
    std::vector<double> r1_vec() const;
    std::vector<double> theta_vec() const;
    
    // indexed by [resampling_region][num_random][r1][theta]
    boost::multi_array<long long int, 4> results();
    boost::multi_array<double, 4> weighted_results();
    
    // indexed by [r1][theta]
    boost::multi_array<long long int, 2> RRR_result();
    boost::multi_array<double, 2> RRR_weighted_result();

    // indexed by [resampling_region][num_random][r1][theta]
    boost::multi_array<long long int, 4> results() const;
    boost::multi_array<double, 4> weighted_results() const;
    
    // indexed by [r1][theta]
    boost::multi_array<long long int, 2> RRR_result() const;
    boost::multi_array<double, 2> RRR_weighted_result() const;

    
    // takes in a (variable-sized) list of regions used in the computation
    // along with the number of randoms involved
    // gets the result out of the matcher, and adds it into results_
    // in the correct place
    // note that region_ids.size() + num_random = tuple_size
    void ProcessResults(std::vector<int>& region_ids, int num_random,
                        bool is_efficient, AngleMatcher& matcher);
    
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
                        EfficientAngleMatcher& matcher);
    
    AngleResults operator+(const AngleResults& other) const;

    void AddResults(const AngleResults& other);
    
    void PrintResults(std::ostream& stream);

    void PrintResults();
    
    bool operator==(const AngleResults &other) const;
    
    bool operator!=(const AngleResults &other) const;
    
  }; // class

} //namespace


#endif