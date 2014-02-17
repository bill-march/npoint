/*
 *  multi_matcher.cpp
 *  
 *
 *  Created by William March on 6/6/11.
 *  Copyright 2011 Georgia Institute of Technology. All rights reserved.
 *
 */

#include "multi_matcher.hpp"

// TODO: think about whether it's worth keeping up with what we pruned before

npoint_mlpack::MultiMatcher::PartialResult::PartialResult(const std::vector<int>& results_size)
:
num_results_(results_size[0]),
results_(num_results_, 0)
{}

npoint_mlpack::MultiMatcher::PartialResult::~PartialResult() {}

void npoint_mlpack::MultiMatcher::PartialResult::Reset()
{
  results_.assign(num_results_, 0);
}

int npoint_mlpack::MultiMatcher::PartialResult::num_results() const
{
  return num_results_;
}

const std::vector<long long int>& npoint_mlpack::MultiMatcher::PartialResult::results() const
{
  return results_;
}

std::vector<long long int>& npoint_mlpack::MultiMatcher::PartialResult::results()
{
  return results_;
}

npoint_mlpack::MultiMatcher::PartialResult&
npoint_mlpack::MultiMatcher::PartialResult::operator=(const PartialResult& other)
{
  
  if (this != &other) {
    
    num_results_ = other.num_results();
    results_ = other.results();
    
  }
  
  return *this;
  
}

npoint_mlpack::MultiMatcher::PartialResult&
npoint_mlpack::MultiMatcher::PartialResult::operator+=(const PartialResult& other)
{
  
  if (this != &other)
  {
    
    if (num_results_ != other.num_results()) {
      mlpack::Log::Fatal << "Using operator+= on mismatched MultiMatcher PartialResults\n";
    }
    
    for (int i = 0; i < num_results_; i++)
    {
      results_[i] += other.results()[i];
    }
    
  }

  return *this;
  
}

const npoint_mlpack::MultiMatcher::PartialResult
npoint_mlpack::MultiMatcher::PartialResult::operator+(const PartialResult &other) const
{
  PartialResult result = *this;
  result += other;
  return result;
}


npoint_mlpack::MultiMatcher::MultiMatcher(const std::vector<arma::mat*>& data_in,
                                const std::vector<arma::colvec*>& weights_in,
                                MatcherArguments& args)
: 
data_mat_list_(data_in),
data_weights_list_(weights_in),
total_matchers_(1),
tuple_size_(args.tuple_size()),
//tuple_size_choose_2_((tuple_size_ * (tuple_size_ - 1)) / 2),
tuple_size_choose_2_(args.num_bands().size()),
//results_(total_matchers_, 0),
//weighted_results_(total_matchers_, 0.0),
min_bands_(args.min_bands()),
max_bands_(args.max_bands()),
num_bands_(args.num_bands()),
band_steps_(tuple_size_choose_2_),
sorted_upper_bounds_sq_(tuple_size_choose_2_),
sorted_lower_bounds_sq_(tuple_size_choose_2_),
perms_(args.tuple_size()), 
num_permutations_(perms_.num_permutations()),
matcher_ind_(-1),
results_size_(1),
do_off_diagonal_(args.do_off_diagonal())
{
  
  if (!do_off_diagonal_) {
    mlpack::Log::Fatal << "Equilateral shapes not implemented for MultiMatcher.\n";
  }
  
  for (int i = 0; i < tuple_size_choose_2_; i++) {
    
    double band_step = (max_bands_[i] - min_bands_[i]) / (double)num_bands_[i];
    band_steps_[i] = band_step;
    
    std::vector<double> bands_i(num_bands_[i]);
    
    std::vector<double> lower_i(num_bands_[i]);
    std::vector<double> upper_i(num_bands_[i]);
    
    if (num_bands_[i] > 1) {
      
      for (int j = 0; j < num_bands_[i]; j++) {
        
        bands_i[j] = min_bands_[i] + (double)j * band_step;
        
        lower_i[j] = (min_bands_[i] + (double)j * band_step)
                      * (min_bands_[i] + (double)j * band_step);
        upper_i[j] = (min_bands_[i] + (double)(j+1) * band_step)
                      * (min_bands_[i] + (double)(j+1) * band_step);
        
      } // for j
      
    } // more than one bandwidth
    else {
      
      bands_i[0] = min_bands_[i];
      lower_i[0] = min_bands_[i] * min_bands_[i];
      upper_i[0] = max_bands_[i] * max_bands_[i];
      
    }
    
    matcher_lower_bounds_sq_.push_back(lower_i);
    matcher_upper_bounds_sq_.push_back(upper_i);
    
    // no longer need half band
    sorted_lower_bounds_sq_[i] = min_bands_[i] * min_bands_[i];
    sorted_upper_bounds_sq_[i] = max_bands_[i] * max_bands_[i];
    
    total_matchers_ *= num_bands_[i];
    
  } // for i
  
  std::sort(sorted_lower_bounds_sq_.begin(), sorted_lower_bounds_sq_.end());
  std::sort(sorted_upper_bounds_sq_.begin(), sorted_upper_bounds_sq_.end());
  
  results_.resize(total_matchers_);
  weighted_results_.resize(total_matchers_);
  
  results_size_[0] = total_matchers_;
  
  //  mlpack::Log::Info << "results inside multi matcher\n";
  for (int i = 0; i < total_matchers_; i++) {
    //mlpack::Log::Info << "results[" << i << "]: " << results_[i] << "\n";
    results_[i] = 0;
    weighted_results_[i] = 0.0;
  
  }
  
} // constructor


size_t npoint_mlpack::MultiMatcher::IndexMatcherDim_(size_t i, size_t j) {
  
  if (i > j) {
    std::swap(i, j);
  }
  
  //assert(i != j);
  
  size_t res = 0;
  
  if (i > 0) {
    for (size_t k = 0; k < i; k++) {
      res += (tuple_size_ - k - 1);
    }
  }
  
  res += (j - i - 1);
  
  return res;
  
} // IndexMatcherDim

bool npoint_mlpack::MultiMatcher::TestNodeTuple(NodeTuple& nodes) {

  std::vector<double> node_lower;
  std::vector<double> node_upper;
  
  // TODO: profile this and try to improve it
  // maybe put this back in the node tuple?
  for (int i = 0; i < tuple_size_; i++) {
    
    for (int j = i+1; j < tuple_size_; j++) {
      
      double lo = nodes.node_list(i)->Bound().MinDistance(nodes.node_list(j)->Bound());
      double hi = nodes.node_list(i)->Bound().MaxDistance(nodes.node_list(j)->Bound());
      
      node_lower.push_back(lo);
      node_upper.push_back(hi);
      
    }
    
  }

  return TestNodeTuple(node_lower, node_upper);
  
} // TestNodeTuple


bool npoint_mlpack::MultiMatcher::TestNodeTuple(std::vector<double>& min_dists_sqr,
                                                std::vector<double>& max_dists_sqr)
{
 
  bool possibly_valid = true;
  
  std::sort(min_dists_sqr.begin(), min_dists_sqr.end());
  std::sort(max_dists_sqr.begin(), max_dists_sqr.end());
  
  for (size_t i = 0; i < sorted_upper_bounds_sq_.size(); i++) {
    
    if (min_dists_sqr[i] > sorted_upper_bounds_sq_[i]) {
      possibly_valid = false;
      break;
    }
    
    if (max_dists_sqr[i] < sorted_lower_bounds_sq_[i]) {
      possibly_valid = false;
      break;
    }
    
  } // for i
  
  return possibly_valid;
  
}



bool npoint_mlpack::MultiMatcher::TestPointPair_(double dist_sq, size_t new_ind, 
                                       size_t old_ind,
                                       std::vector<char>& permutation_ok,
                                       std::vector<std::vector<size_t> >&perm_locations) {

  bool any_matches = false;
  double dist = sqrt(dist_sq);
  
  for (size_t perm_ind = 0; perm_ind < num_permutations_; perm_ind++) {
    
    if (!permutation_ok[perm_ind]) {
      continue;
    }
    
    size_t template_index_1 = GetPermIndex_(perm_ind, new_ind);
    size_t template_index_2 = GetPermIndex_(perm_ind, old_ind);
    
    size_t matcher_ind = IndexMatcherDim_(template_index_1, template_index_2);
    
    double step_size = band_steps_[matcher_ind];
    
    double closest_matcher_val = (dist - min_bands_[matcher_ind]) / step_size;
    
    // Do I need to look at other matchers here?
    unsigned int satisfied_matcher_ind = floor(closest_matcher_val);
    //printf("satisfied matcher ind: %d\n", satisfied_matcher_ind);
    //printf("dist: %g\n", dist);
    
    // Need this to account for large or small distances
    if (satisfied_matcher_ind < matcher_lower_bounds_sq_[matcher_ind].size()) {
      
      if (dist_sq >= matcher_lower_bounds_sq_[matcher_ind][satisfied_matcher_ind]
          && dist_sq <= matcher_upper_bounds_sq_[matcher_ind][satisfied_matcher_ind]) {
        
        any_matches = true;
        perm_locations[perm_ind][matcher_ind] = satisfied_matcher_ind;
        
      }
      else {
        permutation_ok[perm_ind] = false;
      }
    }
    else {
      permutation_ok[perm_ind] = false;
    }
    
    //std::cout << "perm_ok[" << perm_ind << "]: " << permutation_ok[perm_ind] << "\n\n";
    
  } // for perm_ind
  
  //printf("Any matches: %d\n", any_matches);
  
  return any_matches;
  
} // TestPointPair

size_t npoint_mlpack::MultiMatcher::FindResultsInd_(
                              const std::vector<size_t>& perm_locations) {
  
  //std::cout << "Finding results ind\n";
  
  size_t result = 0;
  size_t num_previous_bands = 1;
  
  //for (size_t i = 0; i < perm_locations.size(); i++) {
  for (int i = perm_locations.size() - 1; i >= 0; i--) {
    
    //std::cout << "perm_locations[" << i << "]: " << perm_locations[i] << "\n";
    
    result += perm_locations[i] * num_previous_bands;
    num_previous_bands *= num_bands_[i];
    
  }
  
  return result;
  
} // FindResultsInd

size_t npoint_mlpack::MultiMatcher::GetPermIndex_(size_t perm_index, size_t pt_index) {
  return perms_.GetPermutation(perm_index, pt_index);
} // GetPermIndex_



void npoint_mlpack::MultiMatcher::BaseCaseHelper_(NodeTuple& nodes,
                                        std::vector<char>& permutation_ok,
                                        std::vector<std::vector<size_t> >& perm_locations,
                                        std::vector<size_t>& points_in_tuple,
                                        size_t k,
                                                  PartialResult& result) {
  
  
  // perm_locations[i][j] = k means that in the ith permutation, that 
  // matcher_dists_[j][k] is the current entry in the matcher that this tuple
  // satisfies
  
  //std::vector<char> perm_ok_copy(permutation_ok);
  std::vector<std::vector<size_t> > perm_locations_copy(perm_locations.size());
  
  bool bad_symmetry = false;
  
  NptNode* kth_node = nodes.node_list(k);
  
  // No longer need to check symmetry between nodes, that's taken care of in
  // the recursion
  
  // iterate over possible new points
  for (unsigned int new_point_index = kth_node->Begin(); 
       new_point_index < kth_node->End(); new_point_index++) {
    //for (size_t i = 0; i < point_sets[k].size(); i++) {
    
    bool this_point_works = true;
    
    bad_symmetry = false;
    
    arma::colvec new_point_vec = data_mat_list_[k]->col(new_point_index);
    
    // copy the permutation 
    //perm_ok_copy.assign(permutation_ok.begin(), permutation_ok.end());
    std::vector<char> perm_ok_copy = permutation_ok;
    
    // TODO: check if I can accurately copy this more directly
    for (size_t m = 0; m < perm_locations_copy.size(); m++) {
      //perm_locations_copy[m].assign(perm_locations[m].begin(), 
      //                              perm_locations[m].end());
    
      perm_locations_copy[m] = perm_locations[m];
      
    } // for m
    
    // TODO: double check that I can exit on bad symmetry here
    for (size_t j = 0; j < k && this_point_works && !bad_symmetry; j++) {
      
      unsigned int old_point_ind = points_in_tuple[j];
      NptNode* old_node = nodes.node_list(j);
      
      bad_symmetry = (old_node == kth_node) 
      && (new_point_index <= old_point_ind);
      
      // TODO: if bad_symmetry, can I break out of the loop?
      if (!bad_symmetry) {
        
        arma::colvec old_point_vec = data_mat_list_[j]->col(old_point_ind);
        
        double point_dist_sq = metric_.Evaluate(old_point_vec, 
                                                new_point_vec);
        
        this_point_works = TestPointPair_(point_dist_sq, j, k, 
                                          perm_ok_copy,
                                          perm_locations_copy);
        // perm_locations_copy should now be filled in 
        
      } // check symmetry
      
    } // check existing points
    
    if (this_point_works && !bad_symmetry) {
      
      points_in_tuple[k] = new_point_index;
      
      if (k == (size_t)tuple_size_ - 1) {
        
        // fill in all the results that worked
        
        std::set<size_t> results_set;
        
        // Does all this allow for the possibility that the tuple satisfies
        // more than one matcher
        for (size_t n = 0; n < perm_locations_copy.size(); n++) {
          
          //printf("perm_ok_copy[%u]: %d\n", n, perm_ok_copy[n]);
          //std::cout << "perm_ok_copy[" << n << "]: " << perm_ok_copy[n] << "\n";
          // TODO: might this count an equilateral triangle three times?
          // no, the set takes care of that
          if (perm_ok_copy[n]) {
            size_t results_ind = FindResultsInd_(perm_locations_copy[n]);
            //std::cout << "results ind: " << results_ind << "\n";
            results_set.insert(results_ind);
          }
        } // for n
        
        // Now, iterate through all (distinct) results keys in the set and add
        // them to the total
        std::set<size_t>::iterator it;
        
        for (it = results_set.begin(); it != results_set.end(); it++) {
          
          result.results()[*it]++;
          
        }
        // TODO: add weights here
        
      } // we have a full tuple
      else {
        
        BaseCaseHelper_(nodes, perm_ok_copy, perm_locations_copy,
                        points_in_tuple, k+1, result);
        
      }
      
    } // do we still need to work with these points?
    
  } // iterate over possible new points
  
} // BaseCaseHelper_

void npoint_mlpack::MultiMatcher::ComputeBaseCase(NodeTuple& nodes) {
  
  PartialResult this_result(results_size_);
  ComputeBaseCase(nodes, this_result);
  AddResult(this_result);

} // ComputeBaseCase

void npoint_mlpack::MultiMatcher::ComputeBaseCase(NodeTuple& nodes, 
                                                  PartialResult& result)
{
  
  std::vector<char> permutation_ok(num_permutations_, true);
  
  std::vector<size_t> points_in_tuple(tuple_size_, -1);
  
  std::vector<std::vector<size_t> > perm_locations(num_permutations_);
  
  for (size_t i = 0; i < perm_locations.size(); i++) {
    perm_locations[i].resize(num_bands_.size(), INT_MAX);
  }
  
  BaseCaseHelper_(nodes, permutation_ok, perm_locations,
                  points_in_tuple, 0, result);
  
}

void npoint_mlpack::MultiMatcher::ComputeBaseCase(NptNode* nodeA,
                                                  NptNode* nodeB,
                                                  std::vector<NptNode*>& nodeC_list,
                                                  PartialResult& result)
{

  std::vector<NptNode*> node_list(3);
  node_list[0] = nodeA;
  node_list[1] = nodeB;
  
  // just calling the NodeTuple base case for now
  for (size_t i = 0; i < nodeC_list.size(); i++) 
  {
    
    NptNode* nodeC = nodeC_list[i];
    node_list[2] = nodeC;

    NodeTuple nodes(node_list);
    
    ComputeBaseCase(nodes, result);
    
  }
  
}

void npoint_mlpack::MultiMatcher::AddResult(PartialResult& result)
{
  for (int i = 0; i < total_matchers_; i++)
  {
    results_[i] += result.results()[i];
  }
}


std::vector<long long int>& npoint_mlpack::MultiMatcher::results() {
  return results_;
}

std::vector<double>& npoint_mlpack::MultiMatcher::weighted_results() {
  return weighted_results_;
}


int npoint_mlpack::MultiMatcher::num_permutations() {
  return perms_.num_permutations(); 
}

int npoint_mlpack::MultiMatcher::matcher_ind() const
{
  return matcher_ind_;
}

void npoint_mlpack::MultiMatcher::set_matcher_ind(int new_ind)
{
  matcher_ind_ = new_ind;
}

std::vector<int>& npoint_mlpack::MultiMatcher::results_size() 
{
  return results_size_;
}

double npoint_mlpack::MultiMatcher::min_dist_sq() const
{
  return sorted_lower_bounds_sq_.front();
}

double npoint_mlpack::MultiMatcher::max_dist_sq() const
{
  return sorted_upper_bounds_sq_.back();
}




