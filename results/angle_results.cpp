/*
 *  angle_results.cpp
 *  
 *
 *  Created by William March on 7/27/11.
 *  Copyright 2011 Georgia Institute of Technology. All rights reserved.
 *
 */

#include "angle_results.hpp"

npoint_mlpack::AngleResults::AngleResults()
:
num_regions_(0),
num_r1_(0),
num_theta_(0),
r1_vec_(),
theta_vec_(),
results_(),
weighted_results_(),
RRR_result_(),
RRR_weighted_result_()
{
  
}

npoint_mlpack::AngleResults::AngleResults(MatcherArguments& args,
                                          int num_regions)
:
num_regions_(num_regions),
num_r1_(args.short_sides().size()),
num_theta_(args.thetas().size()),
r1_vec_(args.short_sides()),
theta_vec_(args.thetas()),
results_(boost::extents[num_regions][tuple_size_]
         [r1_vec_.size()][theta_vec_.size()]),
weighted_results_(boost::extents[num_regions][tuple_size_]
                  [r1_vec_.size()][theta_vec_.size()]),
RRR_result_(boost::extents[r1_vec_.size()][theta_vec_.size()]),
RRR_weighted_result_(boost::extents[r1_vec_.size()][theta_vec_.size()])
{
  
  std::fill(results_.origin(), results_.origin() + results_.size(), 0);
  std::fill(weighted_results_.origin(),
            weighted_results_.origin() + weighted_results_.size(), 0.0);
  
  std::fill(RRR_result_.origin(), RRR_result_.origin() + RRR_result_.size(), 0);
  std::fill(RRR_weighted_result_.origin(),
            RRR_weighted_result_.origin() + RRR_weighted_result_.size(), 0.0);
  
}

// copy constructor
npoint_mlpack::AngleResults::AngleResults(const AngleResults& other)
:
num_regions_(other.num_regions()),
num_r1_(other.num_r1()),
num_theta_(other.num_theta()),
r1_vec_(other.r1_vec()),
theta_vec_(other.theta_vec()),
results_(other.results()),
weighted_results_(other.weighted_results()),
RRR_result_(other.RRR_result()),
RRR_weighted_result_(other.RRR_weighted_result())
{
  
  // results should also be copied over already
  
}

npoint_mlpack::AngleResults&
npoint_mlpack::AngleResults::operator=(const AngleResults& other)
{

  // don't self-assign
  if (this != &other)
  {
    
    num_regions_ = other.num_regions();
    num_r1_ = other.num_r1();
    num_theta_ = other.num_theta();
    
    r1_vec_ = other.r1_vec();
    theta_vec_ = other.theta_vec();
    
    results_.resize(boost::extents[num_regions_][tuple_size_][num_r1_][num_theta_]);
    results_ = other.results();
    weighted_results_.resize(boost::extents[num_regions_][tuple_size_][num_r1_][num_theta_]);
    weighted_results_ = other.weighted_results();
    
    RRR_result_.resize(boost::extents[num_r1_][num_theta_]);
    RRR_result_ = other.RRR_result();
    RRR_weighted_result_.resize(boost::extents[num_r1_][num_theta_]);
    RRR_weighted_result_ = other.RRR_weighted_result();
    
  }
  
  return *this;

}


void npoint_mlpack::AngleResults::AddResult_(int region_id, int num_random,
                                   boost::multi_array<long long int, 2>& partial_result) {
  
  for (int r1_ind = 0; r1_ind < num_r1_; r1_ind++) {
    
    for (int theta_ind = 0; theta_ind < num_theta_; theta_ind++) {
      
      results_[region_id][num_random][r1_ind][theta_ind] 
              += partial_result[r1_ind][theta_ind];
      
    } // for theta
    
  } // for r1
  
} // AddResult_

void npoint_mlpack::AngleResults::AddRandomResult_(boost::multi_array<long long int, 2>& partial_result) {
  
  for (int r1_ind = 0; r1_ind < num_r1_; r1_ind++) {
    
    for (int theta_ind = 0; theta_ind < num_theta_; theta_ind++) {
      
      RRR_result_[r1_ind][theta_ind] += partial_result[r1_ind][theta_ind];
      
    } // for theta
    
  } // for r1
  
} // AddRandomResult


void npoint_mlpack::AngleResults::ProcessResults(std::vector<int>& region_ids, 
                                       int num_random,
                                       bool is_efficient,
                                       AngleMatcher& matcher) {
  
  if (num_random == tuple_size_) {
    
    AddRandomResult_(matcher.results());
    
  }
  // corner case for only one resampling region and efficient resampling
  else if (num_regions_ == 1) {
    AddResult_(0, num_random, matcher.results());
  }
  else if (is_efficient) {

    for (int i = 0; i < num_regions_; i++) {

      bool skip_me = false;
      
      for (unsigned int j = 0; j < region_ids.size(); j++) {
        if (i == region_ids[j]) {
          skip_me = true;
          break;
        }
      } // check the invalid region ids
      
      if (!skip_me) {
        AddResult_(i, num_random, matcher.results());
      }
      
    } // for i
    
  } // efficient results
  else {
    // naive results
    // in this case, the region_ids should all be the region that we excluded
    AddResult_(region_ids[0], num_random, matcher.results());
  
  }
    
} // Process Results

void npoint_mlpack::AngleResults::ProcessResults(std::vector<int>& region_ids,
                                                 int num_random,
                                                 bool is_efficient,
                                                 EfficientAngleMatcher& matcher) {
  
  if (num_random == tuple_size_) {
    
    AddRandomResult_(matcher.results());
    
  }
  // corner case for only one resampling region and efficient resampling
  else if (num_regions_ == 1) {
    AddResult_(0, num_random, matcher.results());
  }
  else if (is_efficient) {
    
    for (int i = 0; i < num_regions_; i++) {
      
      bool skip_me = false;
      
      for (unsigned int j = 0; j < region_ids.size(); j++) {
        if (i == region_ids[j]) {
          skip_me = true;
          break;
        }
      } // check the invalid region ids
      
      if (!skip_me) {
        AddResult_(i, num_random, matcher.results());
      }
      
    } // for i
    
  } // efficient results
  else {
    // naive results
    // in this case, the region_ids should all be the region that we excluded
    AddResult_(region_ids[0], num_random, matcher.results());
    
  }
  
} // Process Results


void npoint_mlpack::AngleResults::ProcessResults(std::vector<int>& region_ids,
                                                 int num_random,
                                                 bool is_efficient,
                                                 SingleMatcher& matcher) 
{
 
  int matcher_ind = matcher.matcher_ind();
  int r1_ind = matcher_ind / num_theta_;
  int theta_ind = matcher_ind % num_theta_;
  
  
  if (num_random == tuple_size_) {
    RRR_result_[r1_ind][theta_ind] += matcher.results();
  }
  else if (num_regions_ == 1) {
    // everything goes in the first region
    results_[0][num_random][r1_ind][theta_ind] += matcher.results();
  }
  else if (is_efficient) {
    
    for (int i = 0; i < num_regions_; i++) {
      
      bool skip_me = false;
      
      for (unsigned int j = 0; j < region_ids.size(); j++) {
        if (i == region_ids[j]) {
          skip_me = true;
          break;
        }
      } // check the invalid region ids
      
      if (!skip_me) {
        results_[i][num_random][r1_ind][theta_ind] += matcher.results();
      }

    } // for num_regions_
  
  } // is_efficient
  else {

    results_[region_ids[0]][num_random][r1_ind][theta_ind] += matcher.results();
    
  }
  
} // process results (SingleMatcher)

void npoint_mlpack::AngleResults::ProcessResults(std::vector<int>& region_ids,
                                                 int num_random,
                                                 bool is_efficient,
                                                 EfficientCpuMatcher& matcher)
{
  
  int matcher_ind = matcher.matcher_ind();
  int r1_ind = matcher_ind / num_theta_;
  int theta_ind = matcher_ind % num_theta_;
  
  
  if (num_random == tuple_size_) {
    RRR_result_[r1_ind][theta_ind] += matcher.results();
  }
  else if (num_regions_ == 1) {
    // everything goes in the first region
    results_[0][num_random][r1_ind][theta_ind] += matcher.results();
  }
  else if (is_efficient) {
    
    for (int i = 0; i < num_regions_; i++) {
      
      bool skip_me = false;
      
      for (unsigned int j = 0; j < region_ids.size(); j++) {
        if (i == region_ids[j]) {
          skip_me = true;
          break;
        }
      } // check the invalid region ids
      
      if (!skip_me) {
        results_[i][num_random][r1_ind][theta_ind] += matcher.results();
      }
      
    } // for num_regions_
    
  } // is_efficient
  else {
    
    results_[region_ids[0]][num_random][r1_ind][theta_ind] += matcher.results();
    
  }
  
} // process results (EfficientCpuMatcher)

void npoint_mlpack::AngleResults::ProcessResults(std::vector<int>& region_ids,
                                                 int num_random,
                                                 bool is_efficient,
                                                 ThreePointSingleMatcher& matcher)
{
  
  int matcher_ind = matcher.matcher_ind();
  int r1_ind = matcher_ind / num_theta_;
  int theta_ind = matcher_ind % num_theta_;
  
  
  if (num_random == tuple_size_) {
    RRR_result_[r1_ind][theta_ind] += matcher.results();
  }
  else if (num_regions_ == 1) {
    // everything goes in the first region
    results_[0][num_random][r1_ind][theta_ind] += matcher.results();
  }
  else if (is_efficient) {
    
    for (int i = 0; i < num_regions_; i++) {
      
      bool skip_me = false;
      
      for (unsigned int j = 0; j < region_ids.size(); j++) {
        if (i == region_ids[j]) {
          skip_me = true;
          break;
        }
      } // check the invalid region ids
      
      if (!skip_me) {
        results_[i][num_random][r1_ind][theta_ind] += matcher.results();
      }
      
    } // for num_regions_
    
  } // is_efficient
  else {
    
    results_[region_ids[0]][num_random][r1_ind][theta_ind] += matcher.results();
    
  }
  
} // process results (ThreePointSingleMatcher)


void npoint_mlpack::AngleResults::PrintResults(std::ostream& stream) 
{
  
  std::string d_string(tuple_size_, 'D');
  std::string r_string(tuple_size_, 'R');
  std::string label_string;
  label_string+=d_string;
  label_string+=r_string;
  
  stream << "Multi-angle Resampling Results: \n\n";
  
  for (int region_ind = 0; region_ind < num_regions_; region_ind++) {
    
    stream << "Resampling region " << region_ind << "\n";
    
    for (int num_random = 0; num_random < tuple_size_; num_random++) {
      
      std::string this_string(label_string, num_random, tuple_size_);
      stream << this_string << ": \n";
      
      for (int r1_ind = 0; r1_ind < num_r1_; r1_ind++) {
        
        for (int theta_ind = 0; theta_ind < num_theta_; theta_ind++) {
          
          stream << "r1: " << r1_vec_[r1_ind] << ", theta: ";
          stream << theta_vec_[theta_ind] << ": ";
          stream << results_[region_ind][num_random][r1_ind][theta_ind];
          stream << "\n";
          
        } // for theta
        
      } // for r1_ind
      
      stream << "\n";
      
    } // for num_random
    
  } // for region_ind
  
  stream << "\nRRR results: \n";
  
  for (int r1_ind = 0; r1_ind < num_r1_; r1_ind++) {
    
    for (int theta_ind = 0; theta_ind < num_theta_; theta_ind++) {
      
      stream << "r1: " << r1_vec_[r1_ind] << ", theta: ";
      stream << theta_vec_[theta_ind] << ": ";
      stream << RRR_result_[r1_ind][theta_ind];
      stream << "\n";
      
    } // for theta
    
  } // for r1_ind
  
  stream << "\n";
  
  
} // PrintResults

void npoint_mlpack::AngleResults::PrintResults() 
{
  PrintResults(std::cout);
}

int npoint_mlpack::AngleResults::num_regions() 
{
  return num_regions_;
}

int npoint_mlpack::AngleResults::num_r1() 
{
  return num_r1_;
}

int npoint_mlpack::AngleResults::num_theta() 
{
  return num_theta_;
}

int npoint_mlpack::AngleResults::num_regions() const
{
  return num_regions_;
}

int npoint_mlpack::AngleResults::num_r1() const
{
  return num_r1_;
}

int npoint_mlpack::AngleResults::num_theta() const
{
  return num_theta_;
}

std::vector<double>& npoint_mlpack::AngleResults::r1_vec()
{
  return r1_vec_;
}

std::vector<double>& npoint_mlpack::AngleResults::theta_vec()
{
  return theta_vec_;
}

std::vector<double> npoint_mlpack::AngleResults::r1_vec() const
{
  return r1_vec_;
}

std::vector<double> npoint_mlpack::AngleResults::theta_vec() const
{
  return theta_vec_;
}

// indexed by [resampling_region][num_random][r1][theta]
boost::multi_array<long long int, 4> npoint_mlpack::AngleResults::results()
{
  return results_;
}

boost::multi_array<double, 4> npoint_mlpack::AngleResults::weighted_results()
{
  return weighted_results_;
}

// indexed by [r1][theta]
boost::multi_array<long long int, 2> npoint_mlpack::AngleResults::RRR_result()
{
  return RRR_result_;
}

boost::multi_array<double, 2> npoint_mlpack::AngleResults::RRR_weighted_result()
{
  return RRR_weighted_result_;
}

// indexed by [resampling_region][num_random][r1][theta]
boost::multi_array<long long int, 4>
npoint_mlpack::AngleResults::results() const
{
  return results_;
}

boost::multi_array<double, 4>
npoint_mlpack::AngleResults::weighted_results() const
{
  return weighted_results_;
}

// indexed by [r1][theta]
boost::multi_array<long long int, 2>
npoint_mlpack::AngleResults::RRR_result() const
{
  return RRR_result_;
}

boost::multi_array<double, 2>
npoint_mlpack::AngleResults::RRR_weighted_result() const
{
  return RRR_weighted_result_;
}

void npoint_mlpack::AngleResults::AddResults(const AngleResults& other)
{
  
  for (int i = 0; i < num_regions_; i++)
  {
    for (int j = 0; j < tuple_size_; j++)
    {
      
      for (int k = 0; k < num_r1_; k++)
      {

        for (int l = 0; l < num_theta_; l++)
        {

          results_[i][j][k][l] += other.results()[i][j][k][l];
          weighted_results_[i][j][k][l] += other.weighted_results()[i][j][k][l];
      
        } // for l
      } // for k
    } // for j
  } // for i
  
  for (int k = 0; k < num_r1_; k++)
  {
    
    for (int l = 0; l < num_theta_; l++)
    {
      
      RRR_result_[k][l] += other.RRR_result()[k][l];
      RRR_weighted_result_[k][l] += other.RRR_weighted_result()[k][l];
      
    } // for l
  } // for k
    
} // AddResults

npoint_mlpack::AngleResults
npoint_mlpack::AngleResults::operator+(const AngleResults& other) const
{

  // copy other
  AngleResults results_out(other);
  
  // now, loop through and add
  
  results_out.AddResults(*this);
  
  return results_out;

}

bool npoint_mlpack::AngleResults::operator==(const AngleResults &other) const
{

  bool are_equal = true;
  
  if (num_r1_ != other.num_r1()
      || num_theta_ != other.num_theta()
      || num_regions_ != other.num_regions())
  {
    are_equal = false;
  }

  for (int r1_ind = 0; r1_ind < num_r1_ && are_equal; r1_ind++)
  {
    
    for (int theta_ind = 0; theta_ind < num_theta_ && are_equal; theta_ind++)
    {

      for (int region_ind = 0; region_ind < num_regions_ && are_equal;
           region_ind++)
      {
        
      
        for (int num_random = 0; num_random < tuple_size_ && are_equal;
             num_random++)
        {
          
          // always use zero for index of resampling region since we're only
          // testing one
          are_equal = (results_[region_ind][num_random][r1_ind][theta_ind] ==
                       other.results()[region_ind][num_random][r1_ind][theta_ind]);
          
        } // loop over num random
        
      } // loop over resampling regions
      
      // now check the random result
      are_equal = (RRR_result_[r1_ind][theta_ind] ==
                   other.RRR_result()[r1_ind][theta_ind]);
      
    } // loop over thetas
    
  } // loop over r1
  
  return are_equal;
  
} // operator ==

bool npoint_mlpack::AngleResults::operator!=(const AngleResults &other) const
{
  return !(*this == other);
}

