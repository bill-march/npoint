/*
 *  node_tuple.cpp
 *  
 *
 *  Created by William March on 2/14/11.
 *  Copyright 2011 Georgia Institute of Technology. All rights reserved.
 *
 */

#include "node_tuple.hpp"

npoint_mlpack::NptStat::NptStat()
: leaf_id(-1)
{}

npoint_mlpack::NptStat::NptStat(const arma::mat& /* dataset */, 
        const size_t /* begin */, 
        const size_t /* count */)
: leaf_id(-1)
{}

npoint_mlpack::NptStat::NptStat(const NptStat& other)
:
leaf_id(other.leaf_id)
{}

npoint_mlpack::NptStat::NptStat(mlpack::tree::BinarySpaceTree<mlpack::bound::HRectBound<2, false>,
                                NptStat>& /*node*/)
:
leaf_id(-1)
{}

npoint_mlpack::NptStat::NptStat(const arma::mat& /* dataset */,
        const size_t /* begin */,
        const size_t /* count */,
        const NptStat& /* leftStat */,
        const NptStat& /* rightStat */)
: leaf_id(-1)
{}



npoint_mlpack::NodeTuple::NodeTuple(std::vector<NptNode*>& list_in) 
:
node_list_(list_in),
same_nodes_(list_in.size()), 
tuple_size_(list_in.size())
{
  
  same_nodes_[0] = 0;
  for (unsigned int i = 1; i < node_list_.size(); i++) {
    
    if (node_list_[i] == node_list_[i-1]) {
      
      same_nodes_[i] = same_nodes_[i-1];
      
    }
    else {
      
      same_nodes_[i] = 1 + same_nodes_[i-1];
      
    }
    
  }
    
  UpdateSplitInd_();
  
} // constructor (init)

// use this constructor to make children in the recursion
npoint_mlpack::NodeTuple::NodeTuple(NodeTuple& parent, bool is_left) 
: 
node_list_(parent.get_node_list()),
same_nodes_(parent.get_same_nodes()),
tuple_size_(parent.tuple_size())
{

  ind_to_split_ = parent.ind_to_split();
  
  // assuming that the symmetry has already been checked
  if (is_left) {
    node_list_[ind_to_split_] = parent.node_list(ind_to_split_)->Left();
  }
  else {
    node_list_[ind_to_split_] = parent.node_list(ind_to_split_)->Right();        
  }
  
  UpdateSplitInd_();
  
} // constructor (children)

// Invariant: a tuple of points must go in increasing order of index (within
// the data and random parts of the tuple)
// Therefore, for a tuple of nodes to be valid, it must be possible that some 
// tuple of points taken from it satisfies this requirement
// Therefore, if the begin of one node is larger than the end of a node that 
// preceeds it, the symmetry is violated

// NOTE: this assumes that nodes from the same data set are in consecutive 
// order in the node_list
bool npoint_mlpack::NodeTuple::CheckSymmetry(size_t split_ind, bool is_left) {
  
  
  int start_point = -1;
  int end_point = tuple_size_;
  
  //return true;
  
  // This uses the assumption that the symmetry was correct before
  // Therefore, we only need to check the new node against the others from the
  // same set
  int this_id = same_nodes_[split_ind];
  // start_point is the first index i where same_nodes_[i] = same_nodes_[split_ind]
  // end_point is the last one where this is true
  
  for (unsigned int i = 0; i <= split_ind; i++) {
    if (this_id == same_nodes_[i]) {
      start_point = i;
      break;
    }
  }
  // If the true end_point is tuple_size, it won't get set here, but it was
  // already set above.
  for (int i = split_ind + 1; i < tuple_size_; i++) {
    if (this_id != same_nodes_[i]) {
      end_point = i;
      break;
    }
  }
  
  assert(start_point >= 0);
  
  // only check the new node for symmetry with respect to the others
  if (is_left) {
    for (unsigned int i = start_point; i < split_ind; i++) {
      
      if (node_list_[split_ind]->Left()->End() <= node_list_[i]->Begin()) {
        return false;
      }
      
    } // for i
    
    for (int i = split_ind; i < end_point; i++) {
      
      if (node_list_[i]->End() <= node_list_[split_ind]->Left()->Begin()) {
        return false;
      } 
      
    } // for i
    
  }
  else { // is right

    for (unsigned int i = start_point; i < split_ind; i++) {
      
      if (node_list_[split_ind]->Right()->End() <= node_list_[i]->Begin()) {
        return false;
      }
      
    } // for i
    
    for (int i = split_ind + 1; i < end_point; i++) {
      
      if (node_list_[i]->End() <= node_list_[split_ind]->Right()->Begin()) {
        return false;
      } 
      
    } // for i    
    
  }
  return true;
  
} // CheckSymmetry_



void npoint_mlpack::NodeTuple::UpdateSplitInd_() {
  
  unsigned int split_size = 0;
  
  all_leaves_ = true;
  
  for (int i = 0; i < tuple_size_; i++) {
    
    if (!(node_list_[i]->IsLeaf())) {
      
      all_leaves_ = false;

      if (node_list_[i]->Count() > split_size) {
        split_size = node_list_[i]->Count();
        ind_to_split_ = i;
      }
        
    } // is a leaf?
    
  } // for i
  
} // UpdateSplitInd_

int npoint_mlpack::NodeTuple::NumPossibleTuples() {
  
  double num_tuples = 1;
  int num_repeats = 1;
  
  for (int i = 0; i < tuple_size_; i++) {
    
    // Accounting for repeats (same node multiple times) in the node list
    // Note that this assumes that any two repeated nodes will appear next to 
    // each other in the list.
    if (i > 0 && node_list_[i] == node_list_[i-1]) {
      num_repeats++;
      double this_contribution = (node_list_[i]->Count() - (num_repeats - 1));
      //std::cout << "this_contribution: " << this_contribution << "\n";
      this_contribution = this_contribution / num_repeats; 
      //std::cout << "this_contribution: " << this_contribution << "\n";
      num_tuples *= this_contribution;
      //num_tuples *= (node_list_[i]->Count() - (num_repeats - 1)) / num_repeats;
    }
    else {
      num_repeats = 1;
      num_tuples *= node_list_[i]->Count();
    }
    
  }
  
  return (int)num_tuples;
  
} // NumPossibleTuples

int npoint_mlpack::NodeTuple::NumPairwiseDistances() {
  
  int num_points = 0;
  
  for (int i = 0; i < tuple_size_; i++) {
    
    num_points += node_list_[i]->Count();
    
    
  }
  
  return (num_points * (num_points - 1)) / 2;
  
} // NumPairwiseDistances


const std::vector<npoint_mlpack::NptNode*>& npoint_mlpack::NodeTuple::get_node_list() const {
  return node_list_;
}

const std::vector<int>& npoint_mlpack::NodeTuple::get_same_nodes() const {
  return same_nodes_;
}

bool npoint_mlpack::NodeTuple::all_leaves() {
  return all_leaves_;
}

size_t npoint_mlpack::NodeTuple::ind_to_split() const {
  return ind_to_split_;
}

npoint_mlpack::NptNode*& npoint_mlpack::NodeTuple::node_list(size_t i) {
  return node_list_[i];
}

size_t npoint_mlpack::NodeTuple::tuple_size() const {
  return tuple_size_;
}




