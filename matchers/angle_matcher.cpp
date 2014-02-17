/*
 *  angle_matcher.cpp
 *  
 *
 *  Created by William March on 7/26/11.
 *  Copyright 2011 Georgia Institute of Technology. All rights reserved.
 *
 */

#include "angle_matcher.hpp"

npoint_mlpack::AngleMatcher::PartialResult::PartialResult(std::vector<int>& sizes)
:
num_r1_(sizes[0]),
num_theta_(sizes[1]),
results_(boost::extents[num_r1_][num_theta_])
{
  std::fill(results_.origin(), results_.origin() + results_.size(), 0);
}

npoint_mlpack::AngleMatcher::PartialResult::~PartialResult() {}

void npoint_mlpack::AngleMatcher::PartialResult::Reset()
{
  std::fill(results_.origin(), results_.origin() + results_.size(), 0);
}

int npoint_mlpack::AngleMatcher::PartialResult::num_r1() const
{
  return num_r1_;
}

int npoint_mlpack::AngleMatcher::PartialResult::num_theta() const
{
  return num_theta_;
}

const boost::multi_array<long long int, 2>& npoint_mlpack::AngleMatcher::PartialResult::results() const
{
  return results_;
}

boost::multi_array<long long int, 2>& npoint_mlpack::AngleMatcher::PartialResult::results()
{
  return results_;
}

void npoint_mlpack::AngleMatcher::PartialResult::Increment(int r1_ind, int theta_ind)
{
  results_[r1_ind][theta_ind]++;
}


npoint_mlpack::AngleMatcher::PartialResult&
npoint_mlpack::AngleMatcher::PartialResult::operator=(const PartialResult& other)
{
  
  if (this != &other) {
    
    num_r1_ = other.num_r1();
    num_theta_ = other.num_theta();
    
    results_.resize(boost::extents[num_r1_][num_theta_]);
    results_ = other.results();
    
  }
  
  return *this;
  
}

npoint_mlpack::AngleMatcher::PartialResult&
npoint_mlpack::AngleMatcher::PartialResult::operator+=(const PartialResult& other)
{
  
  if (this != &other)
  {
    
    if (num_r1_ != other.num_r1() || num_theta_ != other.num_theta()) {
      mlpack::Log::Fatal << "Using operator+= on mismatched AngleMatcher PartialResults\n";
    }
    
    for (int i = 0; i < num_r1_; i++)
    {
      for (int j = 0; j < num_theta_; j++)
      {
        results_[i][j] += other.results()[i][j];
      }
    }
    
  }
  
  return *this;
  
}

const npoint_mlpack::AngleMatcher::PartialResult
npoint_mlpack::AngleMatcher::PartialResult::operator+(const PartialResult &other) const {
  PartialResult result = *this;
  result += other;
  return result;
}

npoint_mlpack::AngleMatcher::AngleMatcher(const std::vector<arma::mat*>& data_in,
                                          const std::vector<arma::colvec*>& weights_in,
                                          MatcherArguments& args)
:
data_mat_list_(data_in),
data_weights_list_(weights_in),
short_sides_(args.short_sides()),
long_side_multiplier_(args.long_side()),
long_sides_(short_sides_.size()),
thetas_(args.thetas()),
r3_sides_(boost::extents[short_sides_.size()][thetas_.size()]),
r1_lower_sqr_(short_sides_.size()),
r1_upper_sqr_(short_sides_.size()),
r2_lower_sqr_(short_sides_.size()),
r2_upper_sqr_(short_sides_.size()),
r3_lower_sqr_(boost::extents[short_sides_.size()][thetas_.size()]),
r3_upper_sqr_(boost::extents[short_sides_.size()][thetas_.size()]),
results_size_(2),
results_(boost::extents[short_sides_.size()][thetas_.size()]),
weighted_results_(boost::extents[short_sides_.size()][thetas_.size()]),
tuple_size_(3),
num_base_cases_(0),
num_pairs_considered_(0),
matcher_ind_(-1),
bin_thickness_factor_(args.bin_size())
{
  
  results_size_[0] = short_sides_.size();
  results_size_[1] = thetas_.size();
  
  std::fill(results_.origin(), results_.origin() + results_.size(), 0);
  std::fill(weighted_results_.origin(),
            weighted_results_.origin() + weighted_results_.size(), 0.0);
  
  double half_thickness = bin_thickness_factor_ / 2.0;
  
  for (unsigned int i = 0; i < short_sides_.size(); i++) {
    
    long_sides_[i] = long_side_multiplier_ * short_sides_[i];
    
    r1_lower_sqr_[i] = ((1.0 - half_thickness) * short_sides_[i])
    * ((1.0 - half_thickness) * short_sides_[i]);
    r1_upper_sqr_[i] = ((1.0 + half_thickness) * short_sides_[i])
    * ((1.0 + half_thickness) * short_sides_[i]);
    
    r2_lower_sqr_[i] = ((1.0 - half_thickness) * long_sides_[i])
    * ((1.0 - half_thickness) * long_sides_[i]);
    r2_upper_sqr_[i] = ((1.0 + half_thickness) * long_sides_[i])
    * ((1.0 + half_thickness) * long_sides_[i]);
    
    for (unsigned int j = 0; j < thetas_.size(); j++) {
      
      r3_sides_[i][j] = ComputeR3_(short_sides_[i],
                                   long_sides_[i],
                                   thetas_[j]);
      
      r3_lower_sqr_[i][j] = ((1.0 - half_thickness) * r3_sides_[i][j])
      * ((1.0 - half_thickness) * r3_sides_[i][j]);
      r3_upper_sqr_[i][j] = ((1.0 + half_thickness) * r3_sides_[i][j])
      * ((1.0 + half_thickness) * r3_sides_[i][j]);
      
    } // for j
    
  } // for i
  
  longest_possible_side_sqr_ = std::max(*std::max_element(r1_upper_sqr_.begin(), r1_upper_sqr_.end()),
                                        std::max(*std::max_element(r2_upper_sqr_.begin(), r2_upper_sqr_.end()),
                                                 *std::max_element(r3_upper_sqr_[short_sides_.size() - 1].begin(), r3_upper_sqr_[short_sides_.size() - 1].end())));
 
  shortest_possible_side_sqr_ = std::min(*std::min_element(r1_lower_sqr_.begin(), r1_lower_sqr_.end()),
                                        std::min(*std::min_element(r2_lower_sqr_.begin(), r2_lower_sqr_.end()),
                                                 *std::min_element(r3_lower_sqr_[0].begin(), r3_lower_sqr_[0].end())));
  
} // constructor

// given two edges and the angle between them, compute the length of the
// third size
// TODO: keep things squared?
double npoint_mlpack::AngleMatcher::ComputeR3_(double r1, double r2, double theta) {
  
  double r3sqr = (r1 * r1) + (r2 * r2) - 2.0 * r1 * r2 * cos(theta);

  return sqrt(r3sqr);
  
}


// Naive base case for now
void npoint_mlpack::AngleMatcher::ComputeBaseCase(NodeTuple& nodes) {
  
  PartialResult this_result(results_size_);
  ComputeBaseCase(nodes, this_result);
  AddResult(this_result);
  
} // ComputeBaseCase


// IMPORTANT: this doesn't use the partial result either
// this means it's not thread safe!!!
void npoint_mlpack::AngleMatcher::ComputeBaseCase(NodeTuple& nodes,
                                                  PartialResult& result)
{
  
  num_base_cases_++;
  
  // IMPORTANT: this assumes that the points are all leaves - i.e. that one
  // of the nodes does not have another as a descendent
  for (unsigned int i = nodes.node_list(0)->Begin();
       i < nodes.node_list(0)->End(); i++) {
    
    arma::colvec vec_i = data_mat_list_[0]->col(i);
    
    int j_begin = (nodes.node_list(0) == nodes.node_list(1)) ? i+1 :
    nodes.node_list(1)->Begin();
    int j_end = nodes.node_list(1)->End();
    
    for (int j = j_begin; j < j_end; j++) {
      
      arma::colvec vec_j = data_mat_list_[1]->col(j);
      
      int k_begin;
      if (nodes.node_list(1) == nodes.node_list(2)) {
        k_begin = j+1;
      }
      else {
        k_begin = nodes.node_list(2)->Begin();
      }
      int k_end = nodes.node_list(2)->End();
      
      for (int k = k_begin; k < k_end; k++) {
        
        std::vector<std::vector<int> > valid_indices(short_sides_.size());
        
        arma::colvec vec_k = data_mat_list_[2]->col(k);
        
        TestPointTuple_(vec_i, vec_j, vec_k, valid_indices);
        
        std::set<std::pair<int, int> > index_pairs;
        
        for (unsigned int r1_ind = 0; r1_ind < short_sides_.size(); r1_ind++)
        {
        
          for (unsigned int theta_ind = 0;
               theta_ind < valid_indices[r1_ind].size();
               theta_ind++) {

            index_pairs.insert(std::pair<int, int>(r1_ind, valid_indices[r1_ind][theta_ind]));
            
          } // iterate over valid thetas
          
        } // found valid r1

        for (std::set<std::pair<int, int> >::iterator it = index_pairs.begin();
             it != index_pairs.end(); it++) {
          result.Increment(it->first, it->second);
        }
        
      } // for k
      
    } // for j
    
  } // for i (first element of list)
  
}

// IMPORTANT: this doesn't really use the partial result
// i.e. it directly counts the tuples and stores the result in the matcher
// So, not thread safe!
void npoint_mlpack::AngleMatcher::ComputeBaseCase(NptNode* nodeA,
                     NptNode* nodeB,
                     std::vector<NptNode*>& nodeC_list,
                     PartialResult& result)
{

  std::vector<NptNode*> node_list(3);
  node_list[0] = nodeA;
  node_list[1] = nodeB;

  for (unsigned int c_ind = 0; c_ind < nodeC_list.size(); c_ind++)
  {
    
    node_list[2] = nodeC_list[c_ind];
    
    NodeTuple tuple(node_list);
    
    ComputeBaseCase(tuple, result);
    
  }
  
} // base case (over list of nodes)



// returns the index of the value of r1 that is satisfied by the tuple
// the list contains the indices of thetas_ that are satisfied by the tuple
// assumes that valid_theta_indices is initialized and empty
// Important: it is possible to have a tuple satisfy more than one matcher
// Return -1 if there is no matcher satisfied by this pair
void npoint_mlpack::AngleMatcher::TestPointTuple_(arma::colvec& vec1,
                                                  arma::colvec& vec2,
                                                  arma::colvec& vec3,
                                                  std::vector<std::vector<int> >& valid_indices)
{ 
  
  double d12_sqr = metric_.Evaluate(vec1, vec2);
  double d13_sqr = metric_.Evaluate(vec1, vec3);
  double d23_sqr = metric_.Evaluate(vec2, vec3);
  
  unsigned int lo_index, hi_index;
  
  //////////////////////////////
  // 12 == r1
  
  // IMPORTANT: this is one past where I really want to be
  lo_index = std::lower_bound(r1_lower_sqr_.begin(), r1_lower_sqr_.end(),
                              d12_sqr) - r1_lower_sqr_.begin();
  
  hi_index = std::upper_bound(r1_upper_sqr_.begin(), r1_upper_sqr_.end(),
                              d12_sqr) - r1_upper_sqr_.begin();
  
  int r1_index_12 = -1;
  
  // this edge is too long or short, so go to the next one
  if (hi_index == r1_upper_sqr_.size() || lo_index == 0) {
    
  }
  else {
    // hi_index is our candidate r1_index, if the other distances work as well
    
    // d12 does satisfy the r1 matcher at this index
    if (r1_lower_sqr_[hi_index] <= d12_sqr && d12_sqr <= r1_upper_sqr_[hi_index])
    {
      
      // now, assume that d13 is r2
      if (r2_lower_sqr_[hi_index] <= d13_sqr
          && d13_sqr <= r2_upper_sqr_[hi_index]) {
        
        // now, find the thetas that r23 satisfies, if any
        
        for (unsigned int r3_ind = 0; r3_ind < thetas_.size(); r3_ind++)
        {
          
          // d12 = r1, d13 = r2, d23 = r3
          if (r3_lower_sqr_[hi_index][r3_ind] <= d23_sqr
              && d23_sqr <= r3_upper_sqr_[hi_index][r3_ind])
          {
            // IDEA: the true distance may lie in the gap between two bins (thus
            // it is invalid)
            // We should catch this in teh checks above, however, if we don't
            // I think we should stop here.
            if (hi_index != lo_index - 1) {
              // now, we've found multiple possible values of r1, which we don't
              // currently handle
              mlpack::Log::Fatal << "Multiple matching values of r1 found in AngleMatcher base case.\n";
            }

            r1_index_12 = hi_index;
            valid_indices[r1_index_12].push_back(r3_ind);
            
          } // if we find a valid theta
          
        } // looping over thetas (can replace this with binary search later)
        
      } // assuming d12 == r1, d13 == r2
      
      // now, test the other permutation with d12== r1
      if (r2_lower_sqr_[hi_index] <= d23_sqr
          && d23_sqr <= r2_upper_sqr_[hi_index])
      {
      
        // d12 = r1, d23 = r2, d13 = r3
        for (unsigned int r3_ind = 0; r3_ind < thetas_.size(); r3_ind++)
        {
          
          if (r3_lower_sqr_[hi_index][r3_ind] <= d13_sqr
              && d13_sqr <= r3_upper_sqr_[hi_index][r3_ind])
          {
            if (hi_index != lo_index - 1) {
              // now, we've found multiple possible values of r1, which we don't
              // currently handle
              mlpack::Log::Fatal << "Multiple matching values of r1 found in AngleMatcher base case.\n";
            }

            r1_index_12 = hi_index;
            valid_indices[r1_index_12].push_back(r3_ind);
            
          } // if we find a valid theta
          
        } // looping over thetas (can replace this with binary search later)
        
      } // assuming d12 == r1, r23 == r2
      
    } // d12 does satisfy the r1 matcher distance at hi_index
    
  } // finding candidate thetas with d12 in the first position
  
  //////////////////////////////
  // d13 == r1
  
  // IMPORTANT: this is one past where I really want to be
  lo_index = std::lower_bound(r1_lower_sqr_.begin(), r1_lower_sqr_.end(),
                              d13_sqr) - r1_lower_sqr_.begin();
  
  hi_index = std::upper_bound(r1_upper_sqr_.begin(), r1_upper_sqr_.end(),
                              d13_sqr) - r1_upper_sqr_.begin();
  
  int r1_index_13 = -1;
  
  // this edge is too long or short, so go to the next one
  if (hi_index == r1_upper_sqr_.size() || lo_index == 0) {
    // do nothing
  }
  else {
    // hi_index is our candidate r1_index, if the other distances work as well
    
    // d12 does satisfy the r1 matcher at this index
    if (r1_lower_sqr_[hi_index] <= d13_sqr && d13_sqr <= r1_upper_sqr_[hi_index])
    {
      
      // now, assume that d12 is r2
      if (r2_lower_sqr_[hi_index] <= d12_sqr
          && d12_sqr <= r2_upper_sqr_[hi_index]) {
        
        // now, find the thetas that r23 satisfies, if any
        
        for (unsigned int r3_ind = 0; r3_ind < thetas_.size(); r3_ind++)
        {
          
          // d13 = r1, d12 = r2, d23 = r3
          if (r3_lower_sqr_[hi_index][r3_ind] <= d23_sqr
              && d23_sqr <= r3_upper_sqr_[hi_index][r3_ind])
          {
            if (hi_index != lo_index - 1) {
              // now, we've found multiple possible values of r1, which we don't
              // currently handle
              mlpack::Log::Fatal << "Multiple matching values of r1 found in AngleMatcher base case.\n";
            }

            r1_index_13 = hi_index;
            valid_indices[r1_index_13].push_back(r3_ind);
            
          } // if we find a valid theta
          
        } // looping over thetas (can replace this with binary search later)
        
      } // assuming d13 == r1, d12 == r2
      
      // now, test the other permutation with d13 == r1
      if (r2_lower_sqr_[hi_index] <= d23_sqr
          && d23_sqr <= r2_upper_sqr_[hi_index])
      {
        
        for (unsigned int r3_ind = 0; r3_ind < thetas_.size(); r3_ind++)
        {
          
          // d13 = r1, d23 = r2, d12 = r3
          if (r3_lower_sqr_[hi_index][r3_ind] <= d12_sqr
              && d12_sqr <= r3_upper_sqr_[hi_index][r3_ind])
          {
            if (hi_index != lo_index - 1) {
              // now, we've found multiple possible values of r1, which we don't
              // currently handle
              mlpack::Log::Fatal << "Multiple matching values of r1 found in AngleMatcher base case.\n";
            }

            r1_index_13 = hi_index;
            valid_indices[r1_index_13].push_back(r3_ind);
            
          } // if we find a valid theta
          
        } // looping over thetas (can replace this with binary search later)
        
      } // assuming d12 == r1, r23 == r2
      
    } // d12 does satisfy the r1 matcher distance at hi_index
    
  } // finding candidate thetas with d12 in the first position
  
  //////////////////////////////
  // d23 == r1
  
  // IMPORTANT: this is one past where I really want to be
  lo_index = std::lower_bound(r1_lower_sqr_.begin(), r1_lower_sqr_.end(),
                              d23_sqr) - r1_lower_sqr_.begin();
  
  hi_index = std::upper_bound(r1_upper_sqr_.begin(), r1_upper_sqr_.end(),
                              d23_sqr) - r1_upper_sqr_.begin();
  
  int r1_index_23 = -1;
  
  // this edge is too long or short, so go to the next one
  if (hi_index == r1_upper_sqr_.size() || lo_index == 0) {
    // do nothing
  }
  else {
    // hi_index is our candidate r1_index, if the other distances work as well
    
    // d12 does satisfy the r1 matcher at this index
    if (r1_lower_sqr_[hi_index] <= d23_sqr && d23_sqr <= r1_upper_sqr_[hi_index])
    {
      
      // now, assume that d12 is r2
      if (r2_lower_sqr_[hi_index] <= d12_sqr
          && d12_sqr <= r2_upper_sqr_[hi_index]) {
        
        // now, find the thetas that r13 satisfies, if any        
        for (unsigned int r3_ind = 0; r3_ind < thetas_.size(); r3_ind++)
        {
          
          // d23 = r1, d12 = r2, d13 = r3
          if (r3_lower_sqr_[hi_index][r3_ind] <= d13_sqr
              && d13_sqr <= r3_upper_sqr_[hi_index][r3_ind])
          {
            
            if (hi_index != lo_index - 1) {
              // now, we've found multiple possible values of r1, which we don't
              // currently handle
              mlpack::Log::Fatal << "Multiple matching values of r1 found in AngleMatcher base case.\n";
            }

            r1_index_23 = hi_index;
            valid_indices[r1_index_23].push_back(r3_ind);
            
          } // if we find a valid theta
          
        } // looping over thetas (can replace this with binary search later)
        
      } // assuming d23 == r1, d12 == r2
      
      // now, test the other permutation with d23 == r1
      if (r2_lower_sqr_[hi_index] <= d13_sqr
          && d13_sqr <= r2_upper_sqr_[hi_index])
      {
        
        for (unsigned int r3_ind = 0; r3_ind < thetas_.size(); r3_ind++)
        {
          
          // d23 = r1, d13 = r2, d12 = r3
          if (r3_lower_sqr_[hi_index][r3_ind] <= d12_sqr
              && d12_sqr <= r3_upper_sqr_[hi_index][r3_ind])
          {
            
            if (hi_index != lo_index - 1) {
              // now, we've found multiple possible values of r1, which we don't
              // currently handle
              mlpack::Log::Fatal << "Multiple matching values of r1 found in AngleMatcher base case.\n";
            }

            r1_index_23 = hi_index;
            valid_indices[r1_index_23].push_back(r3_ind);
            
          } // if we find a valid theta
          
        } // looping over thetas (can replace this with binary search later)
        
      } // assuming d12 == r1, r23 == r2
      
    } // d12 does satisfy the r1 matcher distance at hi_index
    
  } // finding candidate thetas with d23 in the first position
  
} // TestPointTuple

// Trying this again
bool npoint_mlpack::AngleMatcher::TestNodeTuple(NodeTuple& nodes) {
  
  // pruning options: all three distances are shorter than the shortest r1
  // or longer than the longest one
  mlpack::bound::HRectBound<2, false>& box1 = nodes.node_list(0)->Bound();
  mlpack::bound::HRectBound<2, false>& box2 = nodes.node_list(1)->Bound();
  mlpack::bound::HRectBound<2, false>& box3 = nodes.node_list(2)->Bound();
  
  double d12_lower = box1.MinDistance(box2);
  double d12_upper = box1.MaxDistance(box2);
  
  double d13_lower = box1.MinDistance(box3);
  double d13_upper = box1.MaxDistance(box3);
  
  double d23_lower = box2.MinDistance(box3);
  double d23_upper = box2.MaxDistance(box3);
  
  std::vector<double> min_dists(3);
  min_dists[0] = d12_lower;
  min_dists[1] = d13_lower;
  min_dists[2] = d23_lower;
  
  std::vector<double> max_dists(3);
  max_dists[0] = d12_upper;
  max_dists[1] = d13_upper;
  max_dists[2] = d23_upper;
  
  return TestNodeTuple(min_dists, max_dists);
  
} // TestNodeTuple

bool npoint_mlpack::AngleMatcher::
TestNodeTuple(std::vector<double>& min_dists_sqr,
              std::vector<double>& max_dists_sqr)
{
  
  bool possibly_valid = true;
  
  double min_dist = *std::min_element(min_dists_sqr.begin(),
                                      min_dists_sqr.end());
  
  double max_dist = *std::max_element(max_dists_sqr.begin(),
                                      max_dists_sqr.end());
  
  if (min_dist > longest_possible_side_sqr_
      || max_dist < shortest_possible_side_sqr_) {
    possibly_valid = false;
  }
  
  return possibly_valid;

}

void npoint_mlpack::AngleMatcher::AddResult(PartialResult& result)
{
  
  for (unsigned int i = 0; i < short_sides_.size(); i++)
  {
    for (unsigned int j = 0; j < thetas_.size(); j++)
    {
      results_[i][j] += result.results()[i][j];
    }
  }
  
}



void npoint_mlpack::AngleMatcher::OutputResults() {
  
  for (size_t j = 0; j < results_.size(); j++) {
    
    for (size_t k = 0; k < results_[j].size(); k++) {
      
      mlpack::Log::Info << "Matcher: ";
      mlpack::Log::Info << "R1: " << short_sides_[j] << ", ";
      mlpack::Log::Info << "R2: " << (short_sides_[j] * long_side_multiplier_) << ", ";
      mlpack::Log::Info << "theta: " << thetas_[k] << ": ";
      
      mlpack::Log::Info << results_[j][k] << "\n";
      
    } // for k
    
  } // for j
  
  mlpack::Log::Info << "\n\n";
  
} // OutputResults

std::vector<int>& npoint_mlpack::AngleMatcher::results_size()
{
  return results_size_;
}

int npoint_mlpack::AngleMatcher::tuple_size() const {
  return tuple_size_;
}

boost::multi_array<long long int, 2>& npoint_mlpack::AngleMatcher::results() {
  return results_;
}

boost::multi_array<double, 2>& npoint_mlpack::AngleMatcher::weighted_results() {
  return weighted_results_;
}

double npoint_mlpack::AngleMatcher::min_dist_sq() const
{
  return shortest_possible_side_sqr_;
}

double npoint_mlpack::AngleMatcher::max_dist_sq() const
{
  return longest_possible_side_sqr_;
}

int npoint_mlpack::AngleMatcher::num_base_cases() const
{
  return num_base_cases_;
}

long long int npoint_mlpack::AngleMatcher::num_pairs_considered() const
{
  return num_pairs_considered_;
}

int npoint_mlpack::AngleMatcher::matcher_ind() const
{
  return matcher_ind_;
}

void npoint_mlpack::AngleMatcher::set_matcher_ind(int new_ind)
{
  matcher_ind_ = new_ind;
}

