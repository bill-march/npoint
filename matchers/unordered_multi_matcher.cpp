/*
 *  efficient_multi_matcher.cpp
 *
 *
 *  Copyright 2011 Georgia Institute of Technology. All rights reserved.
 *
 */

#include "unordered_multi_matcher.hpp"

npoint_mlpack::UnorderedMultiMatcher::PartialResult::PartialResult(const std::vector<int>& results_size)
:
num_results_(results_size[0]),
results_(num_results_, 0)
{}

npoint_mlpack::UnorderedMultiMatcher::PartialResult::~PartialResult() {}

void npoint_mlpack::UnorderedMultiMatcher::PartialResult::Reset()
{
  results_.assign(num_results_, 0);
}

int npoint_mlpack::UnorderedMultiMatcher::PartialResult::num_results() const
{
  return num_results_;
}

const std::vector<long long int>& npoint_mlpack::UnorderedMultiMatcher::PartialResult::results() const
{
  return results_;
}

std::vector<long long int>& npoint_mlpack::UnorderedMultiMatcher::PartialResult::results()
{
  return results_;
}

npoint_mlpack::UnorderedMultiMatcher::PartialResult&
npoint_mlpack::UnorderedMultiMatcher::PartialResult::operator=(const PartialResult& other)
{
  
  if (this != &other) {
    
    num_results_ = other.num_results();
    results_ = other.results();
    
  }
  
  return *this;
  
}

npoint_mlpack::UnorderedMultiMatcher::PartialResult&
npoint_mlpack::UnorderedMultiMatcher::PartialResult::operator+=(const PartialResult& other)
{
  
  if (this != &other)
  {
    
    if (num_results_ != other.num_results()) {
      mlpack::Log::Fatal << "Using operator+= on mismatched UnorderedMultiMatcher PartialResults\n";
    }
    
    for (int i = 0; i < num_results_; i++)
    {
      results_[i] += other.results()[i];
    }
    
  }
  
  return *this;
  
}

const npoint_mlpack::UnorderedMultiMatcher::PartialResult
npoint_mlpack::UnorderedMultiMatcher::PartialResult::operator+(const PartialResult &other) const
{
  PartialResult result = *this;
  result += other;
  return result;
}


npoint_mlpack::UnorderedMultiMatcher::UnorderedMultiMatcher(const std::vector<arma::mat*>& data_in,
                                                            const std::vector<arma::colvec*>& weights_in,
                                                            MatcherArguments& args)
:
data_mat_list_(data_in),
data_weights_list_(weights_in),
tuple_size_(args.tuple_size()),
tuple_size_choose_2_(tuple_size_ * (tuple_size_ - 1) / 2),
lower_matcher_list_(args.lower_matcher_list()),
upper_matcher_list_(args.upper_matcher_list()),
total_matchers_(lower_matcher_list_.size()),
results_(total_matchers_, 0),
weighted_results_(total_matchers_, 0.0),
sorted_upper_bounds_sq_(tuple_size_choose_2_),
sorted_lower_bounds_sq_(tuple_size_choose_2_),
perms_(tuple_size_),
matcher_args_(args),
matcher_ind_(-1),
results_size_(1)
{
  
  results_size_[0] = total_matchers_;
  
  // need to find the max and min matcher in each dimension

  int dim_ind = 0;
  
  for (int i = 0; i < tuple_size_; i++)
  {
    for (int j = i+1; j < tuple_size_; j++)
    {
      
      double this_min = DBL_MAX;
      double this_max = 0.0;
      
      for (int matcher_ind = 0; matcher_ind < total_matchers_; matcher_ind++)
      {

        arma::mat this_lower = lower_matcher_list_[matcher_ind];
        arma::mat this_upper = upper_matcher_list_[matcher_ind];
        
        this_min = std::min(this_min, this_lower(i, j));
        this_max = std::max(this_max, this_upper(i, j));
        
      } // loop over distinct matchers

      sorted_lower_bounds_sq_[dim_ind] = this_min * this_min;
      sorted_upper_bounds_sq_[dim_ind] = this_max * this_max;
      dim_ind++;
      
    }// loop over positions in the tuple (2)
    
  } // loop over positions in the tuple (1)
  
  std::sort(sorted_lower_bounds_sq_.begin(), sorted_lower_bounds_sq_.end());
  std::sort(sorted_upper_bounds_sq_.begin(), sorted_upper_bounds_sq_.end());
  
} // constructor

npoint_mlpack::UnorderedMultiMatcher::~UnorderedMultiMatcher()
{}

bool npoint_mlpack::UnorderedMultiMatcher::TestNodeTuple(NodeTuple& nodes) {
  
  std::vector<double> node_lower;
  std::vector<double> node_upper;
  
  for (int i = 0; i < tuple_size_; i++)
  {
    for (int j = i+1; j < tuple_size_; j++)
    {
      
      double d_lo = nodes.node_list(i)->Bound().MinDistance(nodes.node_list(j)->Bound());
      double d_hi = nodes.node_list(i)->Bound().MaxDistance(nodes.node_list(j)->Bound());
      node_lower.push_back(d_lo);
      node_upper.push_back(d_hi);
      
    }
  }
  
  return TestNodeTuple(node_lower, node_upper);
  
} // TestNodeTuple


bool npoint_mlpack::UnorderedMultiMatcher::TestNodeTuple(std::vector<double>& min_dists_sqr,
                                                         std::vector<double>& max_dists_sqr)
{
  
  bool possibly_valid = true;
  
  std::sort(min_dists_sqr.begin(), min_dists_sqr.end());
  std::sort(max_dists_sqr.begin(), max_dists_sqr.end());
  
  for (unsigned int i = 0; i < min_dists_sqr.size(); i++)
  {
    if (min_dists_sqr[i] > sorted_upper_bounds_sq_[i]
        || max_dists_sqr[i] < sorted_lower_bounds_sq_[i])
    {
      possibly_valid = false;
      break;
    }    
  }

  return possibly_valid;
  
}

void npoint_mlpack::UnorderedMultiMatcher::ComputeBaseCase(NodeTuple& nodes) {
  
  PartialResult this_result(results_size_);
  ComputeBaseCase(nodes, this_result);
  AddResult(this_result);
  
} // ComputeBaseCase

void npoint_mlpack::UnorderedMultiMatcher::ComputeBaseCase(NodeTuple& nodes,
                                                           PartialResult& result)
{
  
  for (int matcher_ind = 0; matcher_ind < total_matchers_; matcher_ind++)
  {
    
    MatcherArguments this_arg = matcher_args_.GenerateMatcher(matcher_ind);
    
    SingleMatcher this_matcher(data_mat_list_, data_weights_list_,
                               this_arg);
    
    this_matcher.ComputeBaseCase(nodes);
    
    // now, get the result out of the matcher and add it to the partial result
    result.results()[matcher_ind] += this_matcher.results();
    
  }

}

void npoint_mlpack::UnorderedMultiMatcher::ComputeBaseCase(NptNode* nodeA,
                                                           NptNode* nodeB,
                                                           std::vector<NptNode*>& nodeC_list,
                                                           PartialResult& result)
{
  
  std::vector<NptNode*> node_list(3);
  node_list[0] = nodeA;
  node_list[1] = nodeB;
  
  for (unsigned int i = 0; i < nodeC_list.size(); i++)
  {
    node_list[2] = nodeC_list[i];
    NodeTuple this_tuple(node_list);
    ComputeBaseCase(this_tuple, result);
  }
  
}

void npoint_mlpack::UnorderedMultiMatcher::AddResult(PartialResult& result)
{
  for (int i = 0; i < total_matchers_; i++)
  {
    results_[i] += result.results()[i];
  }
}


std::vector<long long int>& npoint_mlpack::UnorderedMultiMatcher::results() {
  return results_;
}

std::vector<double>& npoint_mlpack::UnorderedMultiMatcher::weighted_results() {
  return weighted_results_;
}

int npoint_mlpack::UnorderedMultiMatcher::matcher_ind() const
{
  return matcher_ind_;
}

void npoint_mlpack::UnorderedMultiMatcher::set_matcher_ind(int new_ind)
{
  matcher_ind_ = new_ind;
}

std::vector<int>& npoint_mlpack::UnorderedMultiMatcher::results_size()
{
  return results_size_;
}

double npoint_mlpack::UnorderedMultiMatcher::min_dist_sq() const
{
  return sorted_lower_bounds_sq_.front();
}

double npoint_mlpack::UnorderedMultiMatcher::max_dist_sq() const
{
  return sorted_upper_bounds_sq_.back();
}




