/*
 *  single_results.cpp
 *  
 *
 *  Created by William March on 9/7/11.
 *  Copyright 2011 Georgia Institute of Technology. All rights reserved.
 *
 */

#include "single_results.hpp"


npoint_mlpack::SingleResults::SingleResults()
:
tuple_size_(0),
lower_bounds_(),
upper_bounds_(),
results_(),
weighted_results_(),
RRR_result_(0),
RRR_weighted_result_(0.0),
num_regions_(0)
{
  
}
 

npoint_mlpack::SingleResults::SingleResults(MatcherArguments& args,
                                            int num_regions)
:
tuple_size_(args.lower_matcher().n_cols),
lower_bounds_(args.lower_matcher()), 
upper_bounds_(args.upper_matcher()),
results_(boost::extents[num_regions][tuple_size_]),
weighted_results_(boost::extents[num_regions][tuple_size_]),
RRR_result_(0),
RRR_weighted_result_(0.0),
num_regions_(num_regions)
{

  if (args.arg_type() != MatcherArguments::SINGLE_MATCHER) {
    mlpack::Log::Fatal << "Constructing Single Results without Single Matcher arguments.\n";
  }
  
  std::fill(results_.origin(), results_.origin() + results_.size(), 0);
  std::fill(weighted_results_.origin(), 
            weighted_results_.origin() + weighted_results_.size(), 0.0);
  
} // constructor

npoint_mlpack::SingleResults::SingleResults(const SingleResults& other)
:
tuple_size_(other.tuple_size()),
lower_bounds_(other.lower_bounds()),
upper_bounds_(other.upper_bounds()),
results_(other.results()),
weighted_results_(other.weighted_results()),
RRR_result_(other.RRR_result()),
RRR_weighted_result_(other.RRR_weighted_result()),
num_regions_(other.num_regions())
{}

npoint_mlpack::SingleResults&
npoint_mlpack::SingleResults::operator=(const SingleResults& other)
{
 
  // don't self-assign
  if (this != &other)
  {
    
    tuple_size_ = other.tuple_size();
    lower_bounds_ = other.lower_bounds();
    upper_bounds_ = other.upper_bounds();
    
    RRR_result_ = other.RRR_result();
    RRR_weighted_result_ = other.RRR_weighted_result();
    num_regions_ = other.num_regions();
    
    results_.resize(boost::extents[num_regions_][tuple_size_]);
    results_ = other.results();
    weighted_results_.resize(boost::extents[num_regions_][tuple_size_]);
    weighted_results_ = other.weighted_results();
    
  }
  
  return *this;
  
}


void npoint_mlpack::SingleResults::AddResult_(int region_ind, int num_random, 
                                    int result) {
  
  results_[region_ind][num_random] += result;
  
} // AddResult()

void npoint_mlpack::SingleResults::AddResult_(int region_ind, int num_random, 
                                    long long int result) {
  
  results_[region_ind][num_random] += result;
  
} // AddResult()

void npoint_mlpack::SingleResults::AddRandomResult_(int result) {
  
  RRR_result_ += result;
  
}

void npoint_mlpack::SingleResults::AddRandomResult_(long long int result) {
  
  RRR_result_ += result;
  
}


void npoint_mlpack::SingleResults::ProcessResults(std::vector<int>& region_ids, 
                                        int num_random,
                                        bool is_efficient,
                                        SingleMatcher& matcher) {

  ProcessResults(region_ids, num_random, is_efficient, matcher.results());
  
}

void npoint_mlpack::SingleResults::ProcessResults(std::vector<int>& region_ids, 
                                                  int num_random,
                                                  bool is_efficient,
                                                  EfficientCpuMatcher& matcher) {
  
  ProcessResults(region_ids, num_random, is_efficient, matcher.results());
  
}

void npoint_mlpack::SingleResults::ProcessResults(std::vector<int>& region_ids,
                                                  int num_random,
                                                  bool is_efficient,
                                                  Efficient2ptMatcher& matcher) {
  
  ProcessResults(region_ids, num_random, is_efficient, matcher.results());
  
}

void npoint_mlpack::SingleResults::ProcessResults(std::vector<int>& region_ids,
                                                  int num_random,
                                                  bool is_efficient,
                                                  Efficient4ptMatcher& matcher) {
  
  ProcessResults(region_ids, num_random, is_efficient, matcher.results());
  
}

void npoint_mlpack::SingleResults::ProcessResults(std::vector<int>& region_ids, 
                                                  int num_random,
                                                  bool is_efficient,
                                                  ThreePointSingleMatcher& matcher) {
  
  //mlpack::Log::Info << "Processing results for: " << num_random << " randoms.\n";
  //mlpack::Log::Info << "Found " << matcher.results() << " results.\n\n";
  ProcessResults(region_ids, num_random, is_efficient, matcher.results());
  
}



void npoint_mlpack::SingleResults::ProcessResults(std::vector<int>& region_ids, 
                                        int num_random,
                                        bool is_efficient,
                                        long long int this_result) {
  
  if (num_random == tuple_size_) {
    
    AddRandomResult_(this_result);
    
  }
  // corner case that's important for efficient resampling for a single
  // resampling region
  else if (num_regions_ == 1) {
    AddResult_(0, num_random, this_result);
  }
  else if (is_efficient) {
    // efficient resampling
    
    for (int i = 0; i < num_regions_; i++) {
      
      bool skip_me = false;
      
      for (unsigned int j = 0; j < region_ids.size(); j++) {
        
        if (i == region_ids[j]) {
          skip_me = true;
          break;
        }
        
      } // loop over region_id vector
      
      if (!skip_me) {
        AddResult_(i, num_random, this_result);
      }
      
    } // loop over regions
    
  }
  else {
    // naive resampling
    
    // This is wrong, the result for i is the count when i is excluded
    // so, this count should only be put in one place
    AddResult_(region_ids[0], num_random, this_result);

  }
  
} // ProcessResults()




void npoint_mlpack::SingleResults::PrintResults(std::ostream& stream) {

  std::string d_string(tuple_size_, 'D');
  std::string r_string(tuple_size_, 'R');
  std::string label_string;
  label_string+=d_string;
  label_string+=r_string;
  
  stream << "Single Results: \n\n";
  stream << lower_bounds_ << ", " << upper_bounds_ << "\n";
  
  for (int region_ind = 0; region_ind < num_regions_; region_ind++) {
    
    stream << "Resampling region " << region_ind << "\n";
    
    for (int num_random = 0; num_random < tuple_size_; num_random++) {
      
      std::string this_string(label_string, num_random, tuple_size_);
      stream << this_string << ": \n";
      
      stream << results_[region_ind][num_random];
      stream << "\n";
  
    } // for num_random
    
  } // for region_ind
  
  stream << "\nRRR results: ";
  
  stream << RRR_result_;
  stream << "\n";
  stream << "\n";
  
} // PrintResults()

void npoint_mlpack::SingleResults::PrintResults()
{
  PrintResults(std::cout);
}

boost::multi_array<long long int, 2> npoint_mlpack::SingleResults::results() const
{
  return results_;
}

boost::multi_array<double, 2> npoint_mlpack::SingleResults::
weighted_results() const
{
  return weighted_results_;
}

boost::multi_array<long long int, 2>& npoint_mlpack::SingleResults::results() 
{
  return results_;
}

boost::multi_array<double, 2>& npoint_mlpack::SingleResults::
weighted_results() 
{
  return weighted_results_;
}


long long int npoint_mlpack::SingleResults::RRR_result() const {
  return RRR_result_;
}

long long int npoint_mlpack::SingleResults::RRR_weighted_result() const
{
  return RRR_weighted_result_;
}

int npoint_mlpack::SingleResults::tuple_size() const 
{
  return tuple_size_;
}

arma::mat npoint_mlpack::SingleResults::lower_bounds() const
{
  return lower_bounds_;
}

arma::mat npoint_mlpack::SingleResults::upper_bounds() const
{
  return upper_bounds_;
}


arma::mat& npoint_mlpack::SingleResults::lower_bounds()  
{
  return lower_bounds_;
}

arma::mat& npoint_mlpack::SingleResults::upper_bounds()
{
  return upper_bounds_;
}

int npoint_mlpack::SingleResults::num_regions() const
{
  return num_regions_;
}



void npoint_mlpack::SingleResults::AddResults(const SingleResults& other)
{
 
  for (int i = 0; i < num_regions_; i++)
  {
    for (int j = 0; j < tuple_size_; j++)
    {
      results_[i][j] += other.results()[i][j];
      weighted_results_[i][j] += other.weighted_results()[i][j];
    }
  }
  
  RRR_result_ += other.RRR_result();
  RRR_weighted_result_ += other.RRR_weighted_result();
  
}

npoint_mlpack::SingleResults npoint_mlpack::SingleResults::
operator+(const SingleResults& other) const
{
 
  //mlpack::Log::Info << "Adding results.\n";
  
  // copy other
  SingleResults results_out(other);
  
  // now, loop through and add
  
  results_out.AddResults(*this);
  
  return results_out;

  
}

bool npoint_mlpack::SingleResults::operator==(const SingleResults &other) const
{
  
  bool are_equal = true;
  
  if (tuple_size_ != other.tuple_size()
      || num_regions_ != other.num_regions())
  {
    are_equal = false;
  }

  // now check the random result
  are_equal = (RRR_result_ == other.RRR_result());

  for (int region_ind = 0; region_ind < num_regions_ && are_equal;
       region_ind++)
  {
    
    for (int num_random = 0; num_random < tuple_size_ && are_equal;
         num_random++)
    {
      
      // always use zero for index of resampling region since we're only
      // testing one
      are_equal = (results_[region_ind][num_random] ==
                   other.results()[region_ind][num_random]);
      
    } // loop over num random
    
  } // loop over resampling regions
  
  return are_equal;
  
} // operator ==

bool npoint_mlpack::SingleResults::operator!=(const SingleResults &other) const
{
  return !(*this == other);
}


