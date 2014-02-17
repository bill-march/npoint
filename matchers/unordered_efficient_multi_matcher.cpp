/*
 *  efficient_multi_matcher.cpp
 *
 *
 *  Copyright 2011 Georgia Institute of Technology. All rights reserved.
 *
 */

#include "unordered_efficient_multi_matcher.hpp"

npoint_mlpack::UnorderedEfficientMultiMatcher::PartialResult::PartialResult(const std::vector<int>& results_size)
:
num_results_(results_size[0]),
results_(num_results_, 0)
{}

npoint_mlpack::UnorderedEfficientMultiMatcher::PartialResult::~PartialResult() {}

void npoint_mlpack::UnorderedEfficientMultiMatcher::PartialResult::Reset()
{
  results_.assign(num_results_, 0);
}

int npoint_mlpack::UnorderedEfficientMultiMatcher::PartialResult::num_results() const
{
  return num_results_;
}

const std::vector<long long int>& npoint_mlpack::UnorderedEfficientMultiMatcher::PartialResult::results() const
{
  return results_;
}

std::vector<long long int>& npoint_mlpack::UnorderedEfficientMultiMatcher::PartialResult::results()
{
  return results_;
}

npoint_mlpack::UnorderedEfficientMultiMatcher::PartialResult&
npoint_mlpack::UnorderedEfficientMultiMatcher::PartialResult::operator=(const PartialResult& other)
{
  
  if (this != &other) {
    
    num_results_ = other.num_results();
    results_ = other.results();
    
  }
  
  return *this;
  
}

npoint_mlpack::UnorderedEfficientMultiMatcher::PartialResult&
npoint_mlpack::UnorderedEfficientMultiMatcher::PartialResult::operator+=(const PartialResult& other)
{
  
  if (this != &other)
  {
    
    if (num_results_ != other.num_results()) {
      mlpack::Log::Fatal << "Using operator+= on mismatched UnorderedEfficientMultiMatcher PartialResults\n";
    }
    
    for (int i = 0; i < num_results_; i++)
    {
      results_[i] += other.results()[i];
    }
    
  }
  
  return *this;
  
}

const npoint_mlpack::UnorderedEfficientMultiMatcher::PartialResult
npoint_mlpack::UnorderedEfficientMultiMatcher::PartialResult::operator+(const PartialResult &other) const
{
  PartialResult result = *this;
  result += other;
  return result;
}


npoint_mlpack::UnorderedEfficientMultiMatcher::UnorderedEfficientMultiMatcher(const std::vector<arma::mat*>& data_in,
                                                            const std::vector<arma::colvec*>& weights_in,
                                                            MatcherArguments& args)
:
data_mat_list_(data_in),
data_weights_list_(weights_in),
tuple_size_(2),
tuple_size_choose_2_(1),
lower_matcher_list_(args.lower_matcher_list()),
upper_matcher_list_(args.upper_matcher_list()),
total_matchers_(lower_matcher_list_.size()),
results_(total_matchers_, 0),
weighted_results_(total_matchers_, 0.0),
sorted_upper_bounds_sq_(tuple_size_choose_2_),
sorted_lower_bounds_sq_(tuple_size_choose_2_),
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
  
  lower_bounds_ptr_ = new double*[total_matchers_];
  upper_bounds_ptr_ = new double*[total_matchers_];
  
  for (int i = 0; i < total_matchers_; i++)
  {
    
    lower_bounds_ptr_[i] = new double[1];
    lower_bounds_ptr_[i][0] = lower_matcher_list_[i](0,1) * lower_matcher_list_[i](0,1);
    
    //std::cout << "lower bounds:" << i << " " << lower_bounds_ptr_[i][0] << "\n";
    
    upper_bounds_ptr_[i] = new double[1];
    upper_bounds_ptr_[i][0] = upper_matcher_list_[i](0,1) * upper_matcher_list_[i](0,1);
    //std::cout << "upper bounds:" << i << " " << upper_bounds_ptr_[i][0] << "\n";
    
  }
  
} // constructor

npoint_mlpack::UnorderedEfficientMultiMatcher::~UnorderedEfficientMultiMatcher()
{

  for (int i = 0; i< total_matchers_; i++)
  {
    delete lower_bounds_ptr_[i];
    delete upper_bounds_ptr_[i];
  }
  delete lower_bounds_ptr_;
  delete upper_bounds_ptr_;
  
}

bool npoint_mlpack::UnorderedEfficientMultiMatcher::TestNodeTuple(NodeTuple& nodes) {
  
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


bool npoint_mlpack::UnorderedEfficientMultiMatcher::TestNodeTuple(std::vector<double>& min_dists_sqr,
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

void npoint_mlpack::UnorderedEfficientMultiMatcher::ComputeBaseCase(NodeTuple& nodes) {
  
  PartialResult this_result(results_size_);
  ComputeBaseCase(nodes, this_result);
  AddResult(this_result);
  
} // ComputeBaseCase

void npoint_mlpack::UnorderedEfficientMultiMatcher::ComputeBaseCase(NodeTuple& nodes,
                                                           PartialResult& result)
{
  
  NptNode* nodeA = nodes.node_list(0);
  NptNode* nodeB = nodes.node_list(1);
  
  int numA = nodeA->Count();
  int numB = nodeB->Count();
  
  if (numA > 64 || numB > 64)
  {
    mlpack::Log::Fatal << "Calling base case with large nodes in UnorderedEfficientMultiMatcher.\n";
  }
  
  const double3* pointsA = (double3*)data_mat_list_[0]->colptr(nodeA->Begin());
  const double3* pointsB = (double3*)data_mat_list_[1]->colptr(nodeB->Begin());
  
  NptRuntimes runtime;
  
  //std::cout << "allocating results in base case\n";
  uint64_t** kernel_results = new uint64_t*[total_matchers_];
  for (int i = 0; i < total_matchers_; i++) {
    kernel_results[i] = new uint64_t[1];
  }
  
  ComputeTwoPointCorrelationCountsMultiCPU(kernel_results, runtime,
                                           pointsA, numA,
                                           pointsB, numB,
                                           lower_bounds_ptr_,
                                           upper_bounds_ptr_,
                                           total_matchers_,
                                           &satisfiability[0]);
  
  // Need to handle the overcounting we may do
  // This assumes that nodes are either identical or don't overlap
  int num_same_nodes = 0;
  // not sure if this works
  if (nodeA == nodeB) {
    num_same_nodes++;
  }
  
  // it's impossible for 0 and 2 to be equal but not be equal to 1
  int overcounting_factor;
  if (num_same_nodes == 0) {
    overcounting_factor = 1;
  }
  else if (num_same_nodes == 1) {
    overcounting_factor = 2;
  }
  
  // process and store results
  for (int i = 0; i < total_matchers_; i++)
  {
    
    //std::cout << "kernel results: " << kernel_results[i][0] << "\n";
    // add the result for the ith matcher
    result.results()[i] += kernel_results[i][0] / overcounting_factor;
  }
  
  for (int i = 0; i < total_matchers_; i++) {
    delete kernel_results[i];
  }
  delete kernel_results;
  
}

void npoint_mlpack::UnorderedEfficientMultiMatcher::ComputeBaseCase(NptNode* /*nodeA*/,
                                                           NptNode* /*nodeB*/,
                                                           std::vector<NptNode*>& /*nodeC_list*/,
                                                           PartialResult& /*result*/)
{
  
  mlpack::Log::Fatal << "Calling pairwise traversal base cases on 2pt matcher.\n";
  
}

void npoint_mlpack::UnorderedEfficientMultiMatcher::AddResult(PartialResult& result)
{
  for (int i = 0; i < total_matchers_; i++)
  {
    results_[i] += result.results()[i];
  }
}


std::vector<long long int>& npoint_mlpack::UnorderedEfficientMultiMatcher::results() {
  return results_;
}

std::vector<double>& npoint_mlpack::UnorderedEfficientMultiMatcher::weighted_results() {
  return weighted_results_;
}

int npoint_mlpack::UnorderedEfficientMultiMatcher::matcher_ind() const
{
  return matcher_ind_;
}

void npoint_mlpack::UnorderedEfficientMultiMatcher::set_matcher_ind(int new_ind)
{
  matcher_ind_ = new_ind;
}

std::vector<int>& npoint_mlpack::UnorderedEfficientMultiMatcher::results_size()
{
  return results_size_;
}

double npoint_mlpack::UnorderedEfficientMultiMatcher::min_dist_sq() const
{
  return sorted_lower_bounds_sq_.front();
}

double npoint_mlpack::UnorderedEfficientMultiMatcher::max_dist_sq() const
{
  return sorted_upper_bounds_sq_.back();
}




