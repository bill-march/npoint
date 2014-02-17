//
//  multi_results.cpp
//  
//
//  Created by William March on 1/18/12.
//  Copyright (c) 2012 Georgia Institute of Technology. All rights reserved.
//

#include "multi_results.hpp"

npoint_mlpack::MultiResults::MultiResults()
{}

npoint_mlpack::MultiResults::MultiResults(MatcherArguments& args,
                                          int num_regions) 
:
num_resampling_regions_(num_regions),
tuple_size_(args.tuple_size()),
total_num_matchers_(args.total_matchers()),
results_(boost::extents[num_resampling_regions_][tuple_size_][total_num_matchers_]),
weighted_results_(boost::extents[num_resampling_regions_][tuple_size_][total_num_matchers_]),
RRR_result_(total_num_matchers_, 0),
weighted_RRR_result_(total_num_matchers_, 0.0)
{
  
  std::fill(results_.origin(), results_.origin() + results_.size(), 0);
  std::fill(weighted_results_.origin(),
            weighted_results_.origin() + weighted_results_.size(), 0.0);
 
}

// copy constructor
npoint_mlpack::MultiResults::MultiResults(const MultiResults& other)
:
num_resampling_regions_(other.num_resampling_regions()),
tuple_size_(other.tuple_size()),
total_num_matchers_(other.total_num_matchers()),
results_(other.results()),
weighted_results_(other.weighted_results()),
RRR_result_(other.RRR_result()),
weighted_RRR_result_(other.weighted_RRR_result())
{}


npoint_mlpack::MultiResults&
npoint_mlpack::MultiResults::operator=(const MultiResults& other)
{
 
  // don't self-assign
  if (this != &other)
  {
    
    num_resampling_regions_ = other.num_resampling_regions();
    tuple_size_ = other.tuple_size();
    total_num_matchers_ = other.total_num_matchers();
    
    // These are vectors, but the assignment operator should work
    RRR_result_ = other.RRR_result();
    weighted_RRR_result_ = other.weighted_RRR_result();

    results_.resize(boost::extents[num_resampling_regions_][tuple_size_][total_num_matchers_]);
    results_ = other.results();
    weighted_results_.resize(boost::extents[num_resampling_regions_][tuple_size_][total_num_matchers_]);
    weighted_results_ = other.weighted_results();
    
  }
  
  return *this;
  
}



// We're assuming that the vector of results is laid out in the same order as
// the boost multi_array of results stored in this class
void npoint_mlpack::MultiResults::AddRandomResult_(std::vector<long long int>& results) 
{
 
  // TODO: look at using std::transform for this
  // not sure how it will work with the multi_arrays
  for (int i = 0; i < total_num_matchers_; i++) {
    
    RRR_result_[i] += results[i];
    
  }
  
}

void npoint_mlpack::MultiResults::AddResult_(int region_id, int num_random, 
                                   std::vector<long long int>& partial_result)
{
 
  // TODO: look at using std::transform for this
  // not sure how it will work with the multi_arrays
  for (int i = 0; i < total_num_matchers_; i++) {
    
    //mlpack::Log::Info << "before: " << results_[region_id][num_random][i] << "\n";
    //mlpack::Log::Info << "result: " << partial_result[i] << "\n";
    results_[region_id][num_random][i] += partial_result[i];
    //mlpack::Log::Info << "after: " << results_[region_id][num_random][i] << "\n";
    
  }
  
}

void npoint_mlpack::MultiResults::ProcessResults(std::vector<int>& region_ids, 
                                       int num_random,
                                       bool is_efficient, 
                                       MultiMatcher& matcher)
{
  
  // adding in the RR...R result
  if (num_random == tuple_size_) {
    AddRandomResult_(matcher.results());
  }
  // corner case for only one resampling region
  else if (num_resampling_regions_ == 1) {
    AddResult_(0, num_random, matcher.results());
  }
  // adding an efficient resampling result
  else if (is_efficient) {
    
    for (int i = 0; i < num_resampling_regions_; i++) {
      
      bool skip_me = false;
      
      for (unsigned int j = 0; j < region_ids.size(); j++) {
        // are we looking at a region for which these results shouldn't count?
        if (i == region_ids[j]) {
          skip_me = true;
          break;
        }
        
      }// loop over regions (j)
      
      if (!skip_me) {
        AddResult_(i, num_random, matcher.results());
      }
      
    } // for i (over resampling regions)
    
  } // efficient
  // adding a naive resampling result
  else {
    
    // we only excluded the one region
    AddResult_(region_ids[0], num_random, matcher.results());
    
  }
  
}

void npoint_mlpack::MultiResults::ProcessResults(std::vector<int>& region_ids,
                                                 int num_random,
                                                 bool is_efficient,
                                                 UnorderedMultiMatcher& matcher)
{
  
  // adding in the RR...R result
  if (num_random == tuple_size_) {
    AddRandomResult_(matcher.results());
  }
  // corner case for only one resampling region
  else if (num_resampling_regions_ == 1) {
    AddResult_(0, num_random, matcher.results());
  }
  // adding an efficient resampling result
  else if (is_efficient) {
    
    for (int i = 0; i < num_resampling_regions_; i++) {
      
      bool skip_me = false;
      
      for (unsigned int j = 0; j < region_ids.size(); j++) {
        // are we looking at a region for which these results shouldn't count?
        if (i == region_ids[j]) {
          skip_me = true;
          break;
        }
        
      }// loop over regions (j)
      
      if (!skip_me) {
        AddResult_(i, num_random, matcher.results());
      }
      
    } // for i (over resampling regions)
    
  } // efficient
    // adding a naive resampling result
  else {
    
    // we only excluded the one region
    AddResult_(region_ids[0], num_random, matcher.results());
    
  }
  
}

void npoint_mlpack::MultiResults::ProcessResults(std::vector<int>& region_ids,
                                                 int num_random,
                                                 bool is_efficient,
                                                 UnorderedEfficientMultiMatcher& matcher)
{
  
  // adding in the RR...R result
  if (num_random == tuple_size_) {
    AddRandomResult_(matcher.results());
  }
  // corner case for only one resampling region
  else if (num_resampling_regions_ == 1) {
    AddResult_(0, num_random, matcher.results());
  }
  // adding an efficient resampling result
  else if (is_efficient) {
    
    for (int i = 0; i < num_resampling_regions_; i++) {
      
      bool skip_me = false;
      
      for (unsigned int j = 0; j < region_ids.size(); j++) {
        // are we looking at a region for which these results shouldn't count?
        if (i == region_ids[j]) {
          skip_me = true;
          break;
        }
        
      }// loop over regions (j)
      
      if (!skip_me) {
        AddResult_(i, num_random, matcher.results());
      }
      
    } // for i (over resampling regions)
    
  } // efficient
    // adding a naive resampling result
  else {
    
    // we only excluded the one region
    AddResult_(region_ids[0], num_random, matcher.results());
    
  }
  
}


void npoint_mlpack::MultiResults::ProcessResults(std::vector<int>& region_ids,
                                                 int num_random,
                                                 bool is_efficient,
                                                 EfficientMultiMatcher& matcher)
{
  
  // adding in the RR...R result
  if (num_random == tuple_size_) {
    AddRandomResult_(matcher.results());
  }
  // corner case for only one resampling region
  else if (num_resampling_regions_ == 1) {
    AddResult_(0, num_random, matcher.results());
  }
  // adding an efficient resampling result
  else if (is_efficient) {
    
    for (int i = 0; i < num_resampling_regions_; i++) {
      
      bool skip_me = false;
      
      for (unsigned int j = 0; j < region_ids.size(); j++) {
        // are we looking at a region for which these results shouldn't count?
        if (i == region_ids[j]) {
          skip_me = true;
          break;
        }
        
      }// loop over regions (j)
      
      if (!skip_me) {
        AddResult_(i, num_random, matcher.results());
      }
      
    } // for i (over resampling regions)
    
  } // efficient
    // adding a naive resampling result
  else {
    
    // we only excluded the one region
    AddResult_(region_ids[0], num_random, matcher.results());
    
  }
  
}

void npoint_mlpack::MultiResults::ProcessSingleResult_(std::vector<int>& region_ids,
                                                       int num_random,
                                                       bool is_efficient, 
                                                       long long int result,
                                                       int matcher_ind)
{
  
  // adding random result
  if (num_random == tuple_size_) {
    RRR_result_[matcher_ind] += result;
  }
  // corner case for one resampling region
  else if (num_resampling_regions_ == 1) {
    results_[0][num_random][matcher_ind] += result;
  }
  // efficient resampling
  else if (is_efficient) {
    
    for (int i = 0; i < num_resampling_regions_; i++) {
      
      bool skip_me = false;
      
      for (unsigned int j = 0; j < region_ids.size(); j++) {
        if (i == region_ids[j]) {
          skip_me = true;
          break;
        }
      } // check the invalid region ids
      
      if (!skip_me) {
        results_[i][num_random][matcher_ind] += result;
      }
      
    } // for i
    
  }
  // naive resampling
  else {
    results_[region_ids[0]][num_random][matcher_ind] += result;
  }

}

void npoint_mlpack::MultiResults::ProcessResults(std::vector<int>& region_ids,
                                            int num_random,
                                            bool is_efficient, 
                                            SingleMatcher& matcher)
{
  
  ProcessSingleResult_(region_ids,
                       num_random,
                       is_efficient,
                       matcher.results(),
                       matcher.matcher_ind());
  
    
} // SingleMatcher

void npoint_mlpack::MultiResults::ProcessResults(std::vector<int>& region_ids,
                                                 int num_random,
                                                 bool is_efficient, 
                                                 EfficientCpuMatcher& matcher)
{
  
  ProcessSingleResult_(region_ids,
                       num_random,
                       is_efficient,
                       matcher.results(),
                       matcher.matcher_ind());
    
} // EfficientCpuMatcher

void npoint_mlpack::MultiResults::ProcessResults(std::vector<int>& region_ids,
                                                 int num_random,
                                                 bool is_efficient, 
                                                 ThreePointSingleMatcher& matcher)
{
  
  ProcessSingleResult_(region_ids,
                       num_random,
                       is_efficient,
                       matcher.results(),
                       matcher.matcher_ind());
  
} // ThreePointSingleMatcher


void npoint_mlpack::MultiResults::PrintResults(std::ostream& stream)
{
  
  std::string d_string(tuple_size_, 'D');
  std::string r_string(tuple_size_, 'R');
  std::string label_string;
  label_string+=d_string;
  label_string+=r_string;
  
  //printf("printing results\n");
  stream << "Multi-matcher Results: \n\n";
  
  for (int region_ind = 0; region_ind < num_resampling_regions_; region_ind++) {
    
    stream << "Resampling region " << region_ind << "\n";
    
    for (int num_random = 0; num_random < tuple_size_; num_random++) {
      
      std::string this_string(label_string, num_random, tuple_size_);
      stream << this_string << ": \n";
      
      for (int i = 0; i < total_num_matchers_; i++) {
        
        stream << "matcher[" << i << "]: ";
        stream << results_[region_ind][num_random][i] << "\n";
        
      } // loop over matchers
      
      mlpack::Log::Info << "\n";
      
    } // for num_random
    
  } // for region_ind
  
  stream << "\nRR...R results: \n";
  
  for (int i = 0; i < total_num_matchers_; i++) {
    
    stream << "matcher[" << i << "]: ";
    stream << RRR_result_[i] << "\n";
    
  }
  
  stream << "\n";
  
}

void npoint_mlpack::MultiResults::PrintResults()
{
  PrintResults(std::cout);
}


boost::multi_array<long long int, 3>& npoint_mlpack::MultiResults::results()  
{
  return results_;
}

boost::multi_array<double, 3>& npoint_mlpack::MultiResults::weighted_results() 
{
  return weighted_results_;
}

std::vector<long long int>& npoint_mlpack::MultiResults::RRR_result()  
{
  return RRR_result_;
}

std::vector<double>& npoint_mlpack::MultiResults::weighted_RRR_result() 
{
  return weighted_RRR_result_;
}

boost::multi_array<long long int, 3> npoint_mlpack::MultiResults::results() const
{
  return results_;
}

boost::multi_array<double, 3> npoint_mlpack::MultiResults::weighted_results() const
{
  return weighted_results_;
}

std::vector<long long int> npoint_mlpack::MultiResults::RRR_result() const
{
  return RRR_result_;
}

std::vector<double> npoint_mlpack::MultiResults::weighted_RRR_result() const
{
  return weighted_RRR_result_;
}



int npoint_mlpack::MultiResults::num_resampling_regions() const
{
  return num_resampling_regions_;
}

int npoint_mlpack::MultiResults::tuple_size() const 
{
  return tuple_size_;
}

int npoint_mlpack::MultiResults::total_num_matchers() const 
{
  return total_num_matchers_;
}


void npoint_mlpack::MultiResults::AddResults(const MultiResults& other)
{
  
  for (int i = 0; i < num_resampling_regions_; i++)
  {
    for (int j = 0; j < tuple_size_; j++)
    {
      for (int k = 0; k < total_num_matchers_; k++)
      {
        results_[i][j][k] += other.results()[i][j][k];
        weighted_results_[i][j][k] += other.weighted_results()[i][j][k];
      } // for k
    } // for j    
  } // for i
  
  for (int i = 0; i < total_num_matchers_; i++)
  {
    
    RRR_result_[i] += other.RRR_result()[i];
    weighted_RRR_result_[i] += other.weighted_RRR_result()[i];
    
  }
  
}


npoint_mlpack::MultiResults npoint_mlpack::MultiResults::
operator+(const MultiResults& other) const
{
  
  // copy other
  MultiResults results_out(other);
  
  // now, loop through and add
  
  results_out.AddResults(*this);
    
  return results_out;
  
}

bool npoint_mlpack::MultiResults::operator==(const MultiResults &other) const
{
  
  bool are_equal = true;
  
  if (total_num_matchers_ != other.total_num_matchers()
      || tuple_size_ != other.tuple_size()
      || num_resampling_regions_ != other.num_resampling_regions())
  {
    are_equal = false;
  }
  
  for (int matcher_ind = 0; matcher_ind < total_num_matchers_ && are_equal;
       matcher_ind++)
  {
    
    for (int region_ind = 0; region_ind < num_resampling_regions_ && are_equal;
         region_ind++)
    {
      
      
      for (int num_random = 0; num_random < tuple_size_ && are_equal;
           num_random++)
      {
        
        // always use zero for index of resampling region since we're only
        // testing one
        are_equal = (results_[region_ind][num_random][matcher_ind] ==
                     other.results()[region_ind][num_random][matcher_ind]);
        
      } // loop over num random
      
    } // loop over resampling regions
    
    // now check the random result
    are_equal = (RRR_result_[matcher_ind] ==
                 other.RRR_result()[matcher_ind]);
    
  
  } // loop over r1
  
  return are_equal;
  
} // operator ==

bool npoint_mlpack::MultiResults::operator!=(const MultiResults &other) const
{
  return !(*this == other);
}



