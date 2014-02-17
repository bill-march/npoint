/*
 *  generic_npt_alg_impl.hpp
 *  
 *
 *  Created by William March on 8/25/11.
 *  Copyright 2011 Georgia Institute of Technology. All rights reserved.
 *
 */

#ifndef __MLPACK_METHODS_NPOINT_INFRASTRUCTURE_GENERIC_NPT_ALG_IMPL_HPP
#define __MLPACK_METHODS_NPOINT_INFRASTRUCTURE_GENERIC_NPT_ALG_IMPL_HPP


template<class TMatcher>
npoint_mlpack::GenericNptAlg<TMatcher>::GenericNptAlg(std::vector<NptNode*>& trees_in, 
                                                      TMatcher& matcher_in)
:
matcher_(matcher_in), 
trees_(trees_in),
tuple_size_(trees_.size()),
num_prunes_(0),
num_base_cases_(0),
num_point_tuples_considered_(0),
num_pairwise_distances_computed_(0)
{} 

template<class TMatcher>
bool npoint_mlpack::GenericNptAlg<TMatcher>::CanPrune_(NodeTuple& nodes) {
  
  //std::cout << "checking prune.\n";
  return !(matcher_.TestNodeTuple(nodes));
  
} // CanPrune


template<class TMatcher>
void npoint_mlpack::GenericNptAlg<TMatcher>::BaseCase_(NodeTuple& nodes) {
  
  matcher_.ComputeBaseCase(nodes);
  
} // BaseCase_


template<class TMatcher>
void npoint_mlpack::GenericNptAlg<TMatcher>::DepthFirstRecursion_(NodeTuple& nodes) {
  
  if (nodes.all_leaves()) {
    
    //std::cout << "base case\n";
    BaseCase_(nodes);
    // IMPORTANT: this is unnecessary work, take it out for timing runs
    //num_distances_ += nodes.NumPossibleTuples();
    num_point_tuples_considered_ += nodes.NumPossibleTuples();
    num_pairwise_distances_computed_ += nodes.NumPairwiseDistances();
    num_base_cases_++;
    
  } 
  else if (CanPrune_(nodes)) {
    
    num_prunes_++;
    
  }
  else {
    
    // split nodes and call recursion
    
    //std::cout << "splitting nodes\n";
    
    // TODO: can I infer something about one check from the other?
    
    // left child
    if (nodes.CheckSymmetry(nodes.ind_to_split(), true)) {
      // do left recursion
      
      //mlpack::Log::Info << "recursing\n";
      
      NodeTuple left_child(nodes, true);
      num_node_tuples_considered_++;
      DepthFirstRecursion_(left_child);
      
    }
    // TODO: should I count these
    /*
    else {
      mlpack::Log::Info << "symmetry prune\n";
    }
     */
    // right child
    if (nodes.CheckSymmetry(nodes.ind_to_split(), false)) {
      
      //mlpack::Log::Info << "recursing\n";

      NodeTuple right_child(nodes, false);
      num_node_tuples_considered_++;
      DepthFirstRecursion_(right_child);
      
    }
    /*
    else {
      mlpack::Log::Info << "symmetry prune\n";
    }
     */
  
  } // recurse 
  
} // DepthFirstRecursion_


template<class TMatcher>
void npoint_mlpack::GenericNptAlg<TMatcher>::Compute() {
  
  //std::vector<NptNode*> node_list;
  //int next_same = 0;
  
  /*
  data_tree_root_->Print();
  node_list[0]->Print();
  node_list[0]->left()->Print();
  
   */
  
  // matcher needs to know num_random_ too to store counts correctly
  //NodeTuple nodes(node_list, nodes_same);

  // Need this check because some of the test cases can generate resampling 
  // regions with no points in them.
  bool empty_node = false;
  for (int i = 0; i < tuple_size_; i++) {
    if (trees_[i] == NULL) {
      empty_node = true;
      break;
    }
  }

  
  if (!empty_node) {
      
    NodeTuple nodes(trees_);
    
    num_node_tuples_considered_++;
    DepthFirstRecursion_(nodes);
    
  }
    //mlpack::Log::Info << "generic num_base_cases: " << num_base_cases_ << "\n";
  //mlpack::Log::Info << "generic num_prunes: " << num_prunes_ << "\n";
  
} // Compute

template<class TMatcher>
int npoint_mlpack::GenericNptAlg<TMatcher>::num_prunes() const {
  return num_prunes_;
}

template<class TMatcher>
int npoint_mlpack::GenericNptAlg<TMatcher>::num_base_cases() const {
  return num_base_cases_;
}

template<class TMatcher>
long long int npoint_mlpack::GenericNptAlg<TMatcher>::num_point_tuples_considered() const {
  return num_point_tuples_considered_;
}

template<class TMatcher>
long long int npoint_mlpack::GenericNptAlg<TMatcher>::num_node_tuples_considered() const {
  return num_node_tuples_considered_;
}

template<class TMatcher>
long long int npoint_mlpack::GenericNptAlg<TMatcher>::num_pairwise_distances_computed() const {
  return num_pairwise_distances_computed_;
}


template<class TMatcher>
void npoint_mlpack::GenericNptAlg<TMatcher>::PrintStats() {
  
  mlpack::Log::Info << "num_prunes: " << num_prunes_ << "\n";
  mlpack::Log::Info << "num_base_cases: " << num_base_cases_ << "\n";
  mlpack::Log::Info << "num_point_tuples: " << num_point_tuples_considered_ << "\n";
  mlpack::Log::Info << "num_node_tuples: " << num_node_tuples_considered_ << "\n";
  mlpack::Log::Info << "num_pairwise_distances: " << num_pairwise_distances_computed_ << "\n";
  
}




#endif

