/*
 *  single_matcher.cpp
 *  
 *
 *  Created by William March on 2/7/11.
 *  Copyright 2011 Georgia Institute of Technology. All rights reserved.
 *
 */

#include "single_matcher.hpp"

npoint_mlpack::SingleMatcher::PartialResult::PartialResult(const std::vector<int>& /*results_size*/)
:
results_(0)
{}

npoint_mlpack::SingleMatcher::PartialResult::~PartialResult() {}

void npoint_mlpack::SingleMatcher::PartialResult::Reset()
{
  results_ = 0;
}

long long int npoint_mlpack::SingleMatcher::PartialResult::results() const
{
  return results_;
}

void npoint_mlpack::SingleMatcher::PartialResult::IncrementResult()
{
  results_++;
}

npoint_mlpack::SingleMatcher::PartialResult&
npoint_mlpack::SingleMatcher::PartialResult::operator=(const PartialResult& other)
{
  if (this != &other) {
    results_ = other.results();
  }
  
  return *this;
}

npoint_mlpack::SingleMatcher::PartialResult&
npoint_mlpack::SingleMatcher::PartialResult::operator+=(const PartialResult& other)
{
  if (this != &other)
  {
    
    results_ += other.results();
    
  }
  
  return *this;
}

const npoint_mlpack::SingleMatcher::PartialResult
npoint_mlpack::SingleMatcher::PartialResult::operator+(const PartialResult &other) const
{
  PartialResult result = *this;
  result += other;
  return result;
}

npoint_mlpack::SingleMatcher::SingleMatcher(const std::vector<arma::mat*>& data_in,
                                            const std::vector<arma::colvec*>& weights_in,
                                            const MatcherArguments& matcher_args) :
data_mat_list_(data_in), 
data_weights_list_(weights_in),
tuple_size_(matcher_args.lower_matcher().n_cols),
perms_(tuple_size_),
lower_bounds_sqr_(matcher_args.lower_matcher() % matcher_args.lower_matcher()),
upper_bounds_sqr_(matcher_args.upper_matcher() % matcher_args.upper_matcher()),
results_(0),
weighted_results_(0.0),
num_base_cases_(0),
num_permutations_(perms_.num_permutations()),
num_pairs_considered_(0),
matcher_ind_(-1),
results_size_(1,1)
{
  num_permutations_ = perms_.num_permutations();
} // constructor

int npoint_mlpack::SingleMatcher::GetPermIndex_(int perm_index, int pt_index) {
  return perms_.GetPermutation(perm_index, pt_index);
} // GetPermIndex_


bool npoint_mlpack::SingleMatcher::CheckDistances_(double dist_sq, int ind1, 
                                         int ind2) {
  
  //mlpack::Log::Info << "Thread: " << omp_get_thread_num() << " checking distances.\n";

  double upper_bound_sqr = upper_bounds_sqr_(ind1, ind2);
  double lower_bound_sqr = lower_bounds_sqr_(ind1, ind2);
  
  bool retval = dist_sq <= upper_bound_sqr && dist_sq >= lower_bound_sqr;
  
  // This causes a valgrind uninitialized value error
  //bool retval = (dist_sq <= upper_bounds_sqr_(ind1, ind2) && 
  //        dist_sq >= lower_bounds_sqr_(ind1, ind2));
 
  // Adding this print statement makes it work with random data
  // i.e. it finishes (in gdb) with the wrong results because the data were random
  //mlpack::Log::Info << "Check distances: " << lower_bound_sqr << ", " << upper_bound_sqr << "\n";
  
  // using this one instead made it lock up once, but now it seems to work
  // mostly works with this, not always
  //mlpack::Log::Info << "Thread: " << omp_get_thread_num() << " finished with distances.\n";

  return retval;
  //return (dist_sq <= 10.0 && dist_sq >= 0.1);
  
} //CheckDistances_


// note that this assumes that the points have been checked for symmetry
bool npoint_mlpack::SingleMatcher::TestPointPair_(double dist_sq, int tuple_ind_1, 
                                        int tuple_ind_2,
                                        std::vector<char>& permutation_ok) {

  num_pairs_considered_++;
  
  bool any_matches = false;
  
  // iterate over all the permutations
  for (int i = 0; i < num_permutations_; i++) {
    
    // did we already invalidate this one?
    if (!(permutation_ok[i])) {
      continue;
    }
    
    int template_index_1 = GetPermIndex_(i, tuple_ind_1);
    int template_index_2 = GetPermIndex_(i, tuple_ind_2);
    
    //std::cout << "template_indices_checked\n";
    
    // Do the distances work?
    if (CheckDistances_(dist_sq, template_index_1, template_index_2)) {
      any_matches = true;
    }
    else {
      permutation_ok[i] = false;
    }
    
    // IMPORTANT: we can't exit here if any_matches is true
    // This is because the ok permutation might get invalidated later, but we
    // could still end up believing that unchecked ones are ok for this pair
    
  } // for i
  
  return any_matches;
  
} // TestPointPair

// note that for now, there is no subsuming
// this function will need to change if I want to add it
// This returns true if the pair might satisfy the matcher
bool npoint_mlpack::SingleMatcher::TestHrectPair_(double min_dist_sq,
                                                  double max_dist_sq,
                                        int tuple_ind_1, int tuple_ind_2,
                                        std::vector<char>& permutation_ok) {
  
  bool any_matches = false;
  
  //double max_dist_sq = box1.MaxDistance(box2);
  //double min_dist_sq = box1.MinDistance(box2);
  
  // iterate over all the permutations
  // Note that we have to go through all of them 
  for (int i = 0; i < num_permutations_; i++) {
    
    // did we already invalidate this one?
    if (!(permutation_ok[i])) {
      continue;
    }
    
    int template_index_1 = GetPermIndex_(i, tuple_ind_1);
    int template_index_2 = GetPermIndex_(i, tuple_ind_2);
    
    //mlpack::Log::Info << "Thread: " << omp_get_thread_num() << " checking hrect bounds.\n";
    // This causes a valgrind error
    double upper_bound_sqr = upper_bounds_sqr_(template_index_1, 
                                               template_index_2);
    double lower_bound_sqr = lower_bounds_sqr_(template_index_1, 
                                               template_index_2);
    //mlpack::Log::Info << "Thread: " << omp_get_thread_num() << " finished with hrect bounds.\n";
    
    //double upper_bound_sqr = 10.0;
    //double lower_bound_sqr = 1.0;
    
    //mlpack::Log::Info << "TestHrectPair: dists: " << min_dist_sq << ", " << max_dist_sq << "\n";
    //mlpack::Log::Info << "bounds: " << lower_bound_sqr << ", " << upper_bound_sqr << "\n\n";
    
    /*
    printf("max_dist_sq: %g\n", max_dist_sq);
    printf("min_dist_sq: %g\n", min_dist_sq);
    printf("upper_bound_sq: %g\n", upper_bound_sqr);
    printf("lower_bound_sq: %g\n", lower_bound_sqr);
    */
    
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



bool npoint_mlpack::SingleMatcher::TestNodeTuple(NodeTuple& nodes) {
  
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
  
} // TestNodeTuple

bool npoint_mlpack::SingleMatcher::TestNodeTuple(std::vector<double>& min_dists_sqr_,
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


void npoint_mlpack::SingleMatcher::BaseCaseHelper_(NodeTuple& nodes,
                                         std::vector<char>& permutation_ok,
                                         std::vector<int>& points_in_tuple,
                                         int k,
                                                   PartialResult& result) {
  
  //printf("running base case helper\n");

  //std::vector<char> permutation_ok_copy(permutation_ok);
  
  bool bad_symmetry = false;
  
  NptNode* kth_node = nodes.node_list(k);
  
  // iterate over possible kth members of the tuple
  for (unsigned int new_point_index = kth_node->Begin(); 
       new_point_index < kth_node->End(); new_point_index++) {
    
    //printf("checking new point %u\n", new_point_index);

    bool this_point_works = true;
    
    bad_symmetry = false;
    
    // we're dealing with the kth member of the tuple
    //printf("getting new vector\n");
    // Having this print statement keeps it from segfaulting in fast mode
    // arma::colvec new_point_vec = data_mat_list_[k]->col(new_point_index);
    //arma::colvec new_point_vec;
    arma::colvec new_point_vec = data_mat_list_[k]->col(new_point_index);
    
    // putting this here helps 
    std::vector<char> permutation_ok_copy(permutation_ok);

    // TODO: this copies, use unsafe call to do it more efficiently
    //new_point_vec = data_mat_list_[k]->col(new_point_index);

    // TODO: Does this leak memory?
    //permutation_ok_copy.assign(permutation_ok.begin(), permutation_ok.end());
    
    // loop over points already in the tuple and check against them

    //printf("copied permutations, now checking old vectors\n");
    for (int j = 0; !bad_symmetry && this_point_works && j < k; j++) {
      
      
      NptNode* old_node = nodes.node_list(j);
      
      unsigned int old_point_index = points_in_tuple[j];
      
      // TODO: is there a better way?
      // AND if they're from the same set
      
      // Note that we've checked if one node is the ancestor of another in 
      // the generic algorithm
      bad_symmetry = (old_node == kth_node) 
                     && (new_point_index <= old_point_index);
 
      
      if (!bad_symmetry) {
        
        arma::colvec old_point_vec = data_mat_list_[j]->col(old_point_index);
        
        //old_point_vec = data_mat_list_[j]->col(old_point_index);

        double point_dist_sq = metric_.Evaluate(new_point_vec, old_point_vec);
        
        // Putting in just this print statement makes it finish sometimes
        // Seems very erratic this way
        //mlpack::Log::Info << "Testing point distance: " << point_dist_sq << "\n";
        this_point_works = TestPointPair_(point_dist_sq, j, k, 
                                          permutation_ok_copy);
        
      } // check the distances across permutations
      
    } // for j
    
    // point i fits in the tuple
    if (this_point_works && !bad_symmetry) {
      
      points_in_tuple[k] = new_point_index;
      
      // are we finished?
      if (k == tuple_size_ - 1) {
        /*
        std::cout << "Found tuple: ";
        for (int i = 0; i < tuple_size_; i++) {
          std::cout << points_in_tuple[i] << ", ";
        }
        std::cout << "\n";
         */
        result.IncrementResult();
        
      } 
      else {
        
        //printf("recursing \n");
        BaseCaseHelper_(nodes, permutation_ok_copy, points_in_tuple, k+1,
                        result);
        
      } // need to add more points to finish the tuple
      
    } // point i fits
    
  } // for i
  
} // BaseCaseHelper_

void npoint_mlpack::SingleMatcher::ComputeBaseCase(NodeTuple& nodes) {
  PartialResult result(results_size_);
  ComputeBaseCase(nodes, result);
  AddResult(result);
} // BaseCase

void npoint_mlpack::SingleMatcher::ComputeBaseCase(NodeTuple& nodes,
                                                   PartialResult& result)
{

  //mlpack::Timer::Start("single_base_case");
  
  std::vector<char> permutation_ok(num_permutations_, true);
  
  std::vector<int> points_in_tuple(tuple_size_, -1);
  
  //BaseCaseHelper_(point_sets, permutation_ok, points_in_tuple, 0);
  //printf("Doing single base case, %d, %d, %d, points\n", nodes.node_list(0)->Count(),
  // nodes.node_list(1)->Count(), nodes.node_list(2)->Count());
  BaseCaseHelper_(nodes, permutation_ok, points_in_tuple, 0, result);
  
  //lower_bounds_sqr_.print("single lower_bounds");
  //upper_bounds_sqr_.print("single upper_bounds");
  
  //mlpack::Timer::Stop("single_base_case");

}

void npoint_mlpack::SingleMatcher::ComputeBaseCase(NptNode* nodeA,
                     NptNode* nodeB,
                     std::vector<NptNode*>& nodeC_list,
                     PartialResult& result)
{
  
  for (unsigned int i = 0; i < nodeC_list.size(); i++) {
    
    NptNode* nodeC = nodeC_list[i];
    
    std::vector<NptNode*> node_list(3);
    node_list[0] = nodeA;
    node_list[1] = nodeB;
    node_list[2] = nodeC;
    
    NodeTuple nodes(node_list);
    
    ComputeBaseCase(nodes, result);
    
  } // loop over C
  
}


void npoint_mlpack::SingleMatcher::AddResult(PartialResult& result)
{
  
  results_ += result.results();
  
}

void npoint_mlpack::SingleMatcher::OutputResults() {
  
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
  
  
} // OutputResults

int npoint_mlpack::SingleMatcher::results() const {
  return results_;
}

double npoint_mlpack::SingleMatcher::weighted_results() const {
  return weighted_results_;
}

int npoint_mlpack::SingleMatcher::tuple_size() const {
  return tuple_size_;
}

int npoint_mlpack::SingleMatcher::num_permutations() const {
  return num_permutations_;
}

const std::vector<arma::colvec*>& npoint_mlpack::SingleMatcher::data_weights_list() const
{
  return data_weights_list_;
}

const npoint_mlpack::Permutations& npoint_mlpack::SingleMatcher::perms() const {
  return perms_;
}

const arma::mat& npoint_mlpack::SingleMatcher::lower_bounds_sqr() const {
  return lower_bounds_sqr_;
}

const arma::mat& npoint_mlpack::SingleMatcher::upper_bounds_sqr() const {
  return upper_bounds_sqr_;
}

int npoint_mlpack::SingleMatcher::num_base_cases() const {
  return num_base_cases_;
}

long long int npoint_mlpack::SingleMatcher::num_pairs_considered() const {
  return num_pairs_considered_;
}

double npoint_mlpack::SingleMatcher::min_dist_sq() const
{
  
  double min_val = DBL_MAX;
  
  for (int i = 0; i < tuple_size_; i++) {
    
    for (int j = i+1; j < tuple_size_; j++) {
      
      min_val = std::min(min_val, lower_bounds_sqr_(i,j));
      
    }
    
  }
  
  //mlpack::Log::Info << "Min dist in matcher: " << min_val << "\n";
  
  return min_val;
  
}

double npoint_mlpack::SingleMatcher::max_dist_sq() const
{
  
  double max_val = 0.0;
  
  for (int i = 0; i < tuple_size_; i++) {
    
    for (int j = i+1; j < tuple_size_; j++) {
      
      max_val = std::max(max_val, upper_bounds_sqr_(i,j));
      
    }
    
  }
  
  //mlpack::Log::Info << "Max dist in matcher: " << max_val << "\n";

  return max_val;
  
}

int npoint_mlpack::SingleMatcher::matcher_ind() const
{  
  return matcher_ind_;
}

void npoint_mlpack::SingleMatcher::set_matcher_ind(int new_ind)
{  
  matcher_ind_ = new_ind;
}

const std::vector<int>& npoint_mlpack::SingleMatcher::results_size() const
{
  return results_size_;
}



