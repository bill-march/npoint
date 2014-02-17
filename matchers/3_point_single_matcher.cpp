//
//  3_point_single_matcher.cpp
//  contrib_march
//
//  Created by William March on 7/15/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "3_point_single_matcher.hpp"

npoint_mlpack::ThreePointSingleMatcher::PartialResult::PartialResult(const std::vector<int>& /*results_size*/)
:
results_(0)
{}

npoint_mlpack::ThreePointSingleMatcher::PartialResult::~PartialResult() {}

long long int npoint_mlpack::ThreePointSingleMatcher::PartialResult::results() const
{
  return results_;
}

void npoint_mlpack::ThreePointSingleMatcher::PartialResult::Reset()
{
  results_ = 0;
}

void npoint_mlpack::ThreePointSingleMatcher::PartialResult::Increment()
{
  results_++;
}

npoint_mlpack::ThreePointSingleMatcher::PartialResult&
npoint_mlpack::ThreePointSingleMatcher::PartialResult::operator=(const PartialResult& other)
{
  
  if (this != &other)
  {
    results_ = other.results();
  }
  
  return *this;
  
}

npoint_mlpack::ThreePointSingleMatcher::PartialResult&
npoint_mlpack::ThreePointSingleMatcher::PartialResult::operator+=(const PartialResult& other)
{
  
  if (this != &other)
  {
    results_ += other.results();
  }
  
  return *this;
  
}

const npoint_mlpack::ThreePointSingleMatcher::PartialResult
npoint_mlpack::ThreePointSingleMatcher::PartialResult::operator+(const PartialResult &other) const {
  PartialResult result = *this;
  result += other;
  return result;
}


npoint_mlpack::ThreePointSingleMatcher::ThreePointSingleMatcher(const std::vector<arma::mat*>& data_in, 
                        const std::vector<arma::colvec*>& weights_in,
                        const MatcherArguments& matcher_args)
:
data_mat_list_(data_in),
data_weights_list_(weights_in),
tuple_size_(3),
perms_(tuple_size_),
lower_bounds_sqr_(matcher_args.lower_matcher() % matcher_args.lower_matcher()),
upper_bounds_sqr_(matcher_args.upper_matcher() % matcher_args.upper_matcher()),
results_(0),
weighted_results_(0.0),
num_base_cases_(0),
num_permutations_(perms_.num_permutations()),
num_pairs_considered_(0),
matcher_ind_(-1),
results_size_(1)
{
  results_size_[0] = 1;
} // constructor

int npoint_mlpack::ThreePointSingleMatcher::GetPermIndex_(int perm_index, int pt_index) {
  return perms_.GetPermutation(perm_index, pt_index);
} // GetPermIndex_



bool npoint_mlpack::ThreePointSingleMatcher::TestHrectPair_(double min_dist_sq,
                                                  double max_dist_sq,
                                                  int tuple_ind_1, int tuple_ind_2,
                                                  std::vector<char>& permutation_ok) {
  
  bool any_matches = false;
  
    // iterate over all the permutations
  // Note that we have to go through all of them 
  for (int i = 0; i < num_permutations_; i++) {
    
    // did we already invalidate this one?
    if (!(permutation_ok[i])) {
      continue;
    }
    
    int template_index_1 = GetPermIndex_(i, tuple_ind_1);
    int template_index_2 = GetPermIndex_(i, tuple_ind_2);
    
    double upper_bound_sqr = upper_bounds_sqr_(template_index_1, 
                                               template_index_2);
    double lower_bound_sqr = lower_bounds_sqr_(template_index_1, 
                                               template_index_2);
    
    // are they too far or too close?
    if (min_dist_sq > upper_bound_sqr || max_dist_sq < lower_bound_sqr) {
      
      // this permutation doesn't work
      permutation_ok[i] = false;
      
    }
    else {
      
      // this permutation might work
      any_matches = true;
      
    } // end if
    
  } // for i
  
  return any_matches;
  
} // TestHrectPair()


bool npoint_mlpack::ThreePointSingleMatcher::TestNodeTuple(NodeTuple& nodes)
{
  
  // need to use the permutations (rather than the flag idea)
  bool possibly_valid = true;
  
  // note that this has to enforce symmetry
  std::vector<char> permutation_ok(num_permutations_, true);
  
  // iterate over all nodes
  // IMPORTANT: right now, I'm exiting when I can prune
  // I need to double check that this works
  for (int i = 0; possibly_valid && i < tuple_size_; i++) {
    
    NptNode* node_i = nodes.node_list(i);
    
    // iterate over all nodes > i
    for (int j = i+1; possibly_valid && j < tuple_size_; j++) {
      
      NptNode* node_j = nodes.node_list(j);
      
      double min_dist_sq = node_i->Bound().MinDistance(node_j->Bound());
      double max_dist_sq = node_i->Bound().MaxDistance(node_j->Bound());
      
      // If this ever returns false, we exit the loop because we can prune
      possibly_valid = TestHrectPair_(min_dist_sq, max_dist_sq,
                                      i, j, permutation_ok);
      
    } // for j
    
  } // for i
  
  return possibly_valid;

}

// Important: these are kept in the same order, since they have to match
// up
bool npoint_mlpack::ThreePointSingleMatcher::TestNodeTuple(std::vector<double>& min_dists_sqr_,
                   std::vector<double>& max_dists_sqr_)
{
  
  bool possibly_valid = true;
  
  // note that this has to enforce symmetry
  std::vector<char> permutation_ok(num_permutations_, true);
  
  int dist_ind = 0;
  
  for (int i = 0; i < tuple_size_ && possibly_valid; i++) {
    
    for (int j = i+1; j < tuple_size_ && possibly_valid; j++) {
      
      possibly_valid = TestHrectPair_(min_dists_sqr_[dist_ind],
                                      max_dists_sqr_[dist_ind],
                                      i, j, permutation_ok);
      
      dist_ind++;
      
    }
    
  }
  
  return possibly_valid;

}

bool npoint_mlpack::ThreePointSingleMatcher::TestPermutation_(size_t perm_ind,
                      double dist01,
                      double dist02,
                      double dist12)
{

  int template_index_1;
  int template_index_2;
  
  double upper_bound_sqr;
  double lower_bound_sqr;
  
  template_index_1 = GetPermIndex_(perm_ind, 0);
  template_index_2 = GetPermIndex_(perm_ind, 1);
  
  upper_bound_sqr = upper_bounds_sqr_(template_index_1, 
                                             template_index_2);
  lower_bound_sqr = lower_bounds_sqr_(template_index_1, 
                                             template_index_2);
  
  if (dist01 > upper_bound_sqr || dist01 < lower_bound_sqr) {
    return false;
  }
  
  template_index_1 = GetPermIndex_(perm_ind, 0);
  template_index_2 = GetPermIndex_(perm_ind, 2);
  
  upper_bound_sqr = upper_bounds_sqr_(template_index_1, 
                                      template_index_2);
  lower_bound_sqr = lower_bounds_sqr_(template_index_1, 
                                      template_index_2);
  
  if (dist02 > upper_bound_sqr || dist02 < lower_bound_sqr) {
    return false;
  }
  
  template_index_1 = GetPermIndex_(perm_ind, 1);
  template_index_2 = GetPermIndex_(perm_ind, 2);
  
  upper_bound_sqr = upper_bounds_sqr_(template_index_1, 
                                      template_index_2);
  lower_bound_sqr = lower_bounds_sqr_(template_index_1, 
                                      template_index_2);
  
  if (dist12 > upper_bound_sqr || dist12 < lower_bound_sqr) {
    return false;
  }
  
  return true;
  
} // TestPermutation


void npoint_mlpack::ThreePointSingleMatcher::ComputeBaseCase(NodeTuple& nodes,
                                                             PartialResult& result)
{

  num_base_cases_++;
  
  NptNode* node_i = nodes.node_list(0);
  NptNode* node_j = nodes.node_list(1);
  NptNode* node_k = nodes.node_list(2);
  
  size_t begin_i = node_i->Begin();
  size_t end_i = node_i->End();
  // points in node(0)
  for (size_t i = begin_i; i < end_i; i++) {
    
    arma::colvec vec_i = data_mat_list_[0]->col(i);
    
    size_t begin_j = (node_i == node_j) ? i+1 : node_j->Begin();
    size_t end_j = node_j->End();
    // points in node(1)
    for (size_t j = begin_j; j < end_j; j++) {
      
      arma::colvec vec_j = data_mat_list_[1]->col(j);
      
      double dist_ij = metric_.Evaluate(vec_i, vec_j);
      
      size_t begin_k = (node_k == node_j) ? j+1 : node_k->Begin(); 
      size_t end_k = node_k->End();
      // points in node(2)
      for (size_t k = begin_k; k < end_k; k++) {
        
        arma::colvec vec_k = data_mat_list_[2]->col(k);
        
        double dist_ik = metric_.Evaluate(vec_i, vec_k);
        double dist_jk = metric_.Evaluate(vec_j, vec_k);
        
        bool tuple_works = false;
        
        for (int perm_ind = 0; perm_ind < num_permutations_ && !tuple_works; perm_ind++)
        {
          
          // now, test if the points work in this permutation
          // if so, break and count the tuple
          // if not, on to next permutation
          tuple_works = TestPermutation_(perm_ind, 
                                         dist_ij, 
                                         dist_ik, 
                                         dist_jk);
          
          
        } // for num_permutations

        if (tuple_works) {
          result.Increment();
        }
      
      } // for k
      
    } // for j
    
  } // for i
  
  //mlpack::Log::Info << "Results found in 3pt base case: " << result.result << "\n";
  
} // ComputeBaseCase

void npoint_mlpack::ThreePointSingleMatcher::ComputeBaseCase(NodeTuple& nodes)
{
  
  PartialResult result(results_size_);
  
  ComputeBaseCase(nodes, result);
  
  results_ += result.results();
  
}

void npoint_mlpack::ThreePointSingleMatcher::ComputeBaseCase(NptNode* nodeA,
                     NptNode* nodeB,
                     std::vector<NptNode*>& nodeC_list,
                     PartialResult& result)
{
  
  std::vector<NptNode*> node_list(3);
  node_list[0] = nodeA;
  node_list[1] = nodeB;
  
  for (size_t i = 0; i < nodeC_list.size(); i++) {
  
    PartialResult my_result(results_size_);
    
    node_list[2] = nodeC_list[i];
    
    NodeTuple nodes(node_list);
    
    ComputeBaseCase(nodes, my_result);
    
    result += my_result;
    
  }
  
  // results are still around here
  //mlpack::Log::Info << "\n3pt base case (for list) found " << result.result << " results\n";
  
}

void npoint_mlpack::ThreePointSingleMatcher::AddResult(PartialResult& result)
{
  results_ += result.results();
}

long long int npoint_mlpack::ThreePointSingleMatcher::results() const
{
  return results_;
}

double npoint_mlpack::ThreePointSingleMatcher::weighted_results() const
{
  return weighted_results_;
}

int npoint_mlpack::ThreePointSingleMatcher::tuple_size() const
{
  return tuple_size_;
}

int npoint_mlpack::ThreePointSingleMatcher::num_permutations() const
{
  return num_permutations_;
}

const std::vector<arma::colvec*>& npoint_mlpack::ThreePointSingleMatcher::data_weights_list() const
{
  return data_weights_list_;
}

const npoint_mlpack::Permutations& npoint_mlpack::ThreePointSingleMatcher::perms() const
{
  return perms_;
}

const arma::mat& npoint_mlpack::ThreePointSingleMatcher::lower_bounds_sqr() const
{
  return lower_bounds_sqr_;
}

const arma::mat& npoint_mlpack::ThreePointSingleMatcher::upper_bounds_sqr() const
{
  return upper_bounds_sqr_;
}

int npoint_mlpack::ThreePointSingleMatcher::num_base_cases() const
{
  return num_base_cases_;
}

long long int npoint_mlpack::ThreePointSingleMatcher::num_pairs_considered() const
{
  return num_pairs_considered_;
}

double npoint_mlpack::ThreePointSingleMatcher::min_dist_sq() const
{
  
  return std::min(lower_bounds_sqr_(0,1), std::min(lower_bounds_sqr_(0,2),
                                                   lower_bounds_sqr_(1,2)));
  
}

double npoint_mlpack::ThreePointSingleMatcher::max_dist_sq() const
{
  return std::max(upper_bounds_sqr_(0,1), std::max(upper_bounds_sqr_(0,2),
                                                   upper_bounds_sqr_(1,2)));
}

int npoint_mlpack::ThreePointSingleMatcher::matcher_ind() const
{  
  return matcher_ind_;
}

void npoint_mlpack::ThreePointSingleMatcher::set_matcher_ind(int new_ind)
{  
  matcher_ind_ = new_ind;
}


void npoint_mlpack::ThreePointSingleMatcher::OutputResults()
{
  
  std::string d_string(tuple_size_, 'D');
  std::string r_string(tuple_size_, 'R');
  std::string label_string;
  label_string+=d_string;
  label_string+=r_string;
  
  for (int i = 0; i <= tuple_size_; i++) {
    
    // i is the number of random points in the tuple
    std::string this_string(label_string, i, tuple_size_);
    mlpack::Log::Info << this_string << ": ";
    
    mlpack::Log::Info << results_ << "\n";
    
    //mlpack::Log::Info << "\n\n";
    
  } // for i

}

const std::vector<int>& npoint_mlpack::ThreePointSingleMatcher::results_size() const
{
  return results_size_;
}

