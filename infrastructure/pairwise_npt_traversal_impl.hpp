//
//  pairwise_npt_travesal_impl.hpp
//  
//
//  Created by William March on 4/27/12.
//  Copyright (c) 2012 Georgia Institute of Technology. All rights reserved.
//

#ifndef __MLPACK_METHODS_NPOINT_INFRASTRUCTURE_PAIRWISE_NPT_TRAVERSAL_IMPL_HPP
#define __MLPACK_METHODS_NPOINT_INFRASTRUCTURE_PAIRWISE_NPT_TRAVERSAL_IMPL_HPP


template <class TMatcher>
npoint_mlpack::PairwiseNptTraversal<TMatcher>::PairwiseNptTraversal(std::vector<NptNode*>& trees_in,
                                                          TMatcher& matcher_in)
:
matcher_(matcher_in),
min_matcher_dist_sq_(matcher_.min_dist_sq()),
max_matcher_dist_sq_(matcher_.max_dist_sq()),
trees_(trees_in),
tuple_size_(3),
num_prunes_(0),
num_base_cases_(0),
num_point_tuples_considered_(0),
num_node_tuples_considered_(0),
num_pairwise_distances_computed_(0),
// IMPORTANT: this is a hack
same_01_(trees_[0] == trees_[1]),
same_02_(trees_[0] == trees_[2]),
same_12_(trees_[1] == trees_[2]),
leaf_id_(0)
//num_leaves_(num_leaves),
//leaf_list_(num_leaves_, NULL)
{
  
  // Need to compute the number of leaves
  
  num_leaves_ = ComputeNumLeaves_(trees_[0]);
  
  //mlpack::Log::Info << "Num Leaves in pairwise traversal constructor: " << num_leaves_ << "\n";
  
  leaf_list_.resize(num_leaves_, NULL);
  
  /*
  std::vector<NptNode*>::iterator it;
  
  int i = 0;
  for (it = leaf_list_.begin(); it != leaf_list_.end(); it++)
  {
    mlpack::Log::Info << "leaf_list_[" << i <<"]: " << *it << "\n";
    i++;
    
  }
   */
  
} // constructor

// this also needs to set the leaf ids
template<class TMatcher>
long long int npoint_mlpack::PairwiseNptTraversal<TMatcher>::ComputeNumLeaves_(NptNode* tree)
{
 
  // we might call this traverser with empty sets in the resampling or
  // distributed settings
  // since this is called by the constructor, we need to check here
  if (!tree)
  {
    return 0;
  }
  else {
    
    if (tree->IsLeaf()) {
      //mlpack::Log::Info << "In base case for num leaves\n";
      // need to set this 
      tree->Stat().leaf_id = leaf_id_;
      leaf_id_++;
      
      return 1;
    }
    else {
      return (ComputeNumLeaves_(tree->Left()) + ComputeNumLeaves_(tree->Right()));
    }
  
  }
  
} // ComputeNumLeaves_()

template<class TMatcher>
void npoint_mlpack::PairwiseNptTraversal<TMatcher>::Compute() 
{
  
  // fill in the interaction lists
  
  if (!trees_[0] || !trees_[1] || !trees_[2])
  {
    // we don't have any work to do, just return
    return;
  }
    
  // we might need two lists if trees_1 and trees_[2] are different
  // we don't need interactions between 1 and 2 because they'll still have to 
  // interact with 0
  // we're only throwing out things that can't possibly interact with 0, so
  // they can't contribute to any triples with 0 and we're fine
  //std::map<NptNode*, std::vector<NptNode*> > interaction_list_01;
  //std::map<NptNode*, std::vector<NptNode*> > interaction_list_02;
  std::vector<std::vector<NptNode*> > interaction_list_01(num_leaves_);
  std::vector<std::vector<NptNode*> > interaction_list_02(num_leaves_);
    
  mlpack::Timer::Start("fill_list_time");
  
  //std::cout << "Filling list 01.\n";
  FillLists_(trees_[0], trees_[1], same_01_, 
             interaction_list_01);
    
  if (!same_12_) {
    //std::cout << "Filling list 02.\n";
    FillLists_(trees_[0], trees_[2], same_02_,
               interaction_list_02);
  }
  
  //mlpack::Log::Info << "List 01 length: " << interaction_list_01.size() << "\n";
  
  //std::cout << "Finished filling lists.\n";
  
  mlpack::Timer::Stop("fill_list_time");
  
  mlpack::Timer::Start("traverse_list_time");
  if (same_12_) {
    //std::cout << "Traversing 1 List.\n";
    TraverseLists_(interaction_list_01);
  }
  else {
    //std::cout << "Traversing 2 Lists.\n";
    TraverseLists_(interaction_list_01, interaction_list_02);
  }
  mlpack::Timer::Stop("traverse_list_time");
  
} // compute()

// returns true if we can throw the pair away
template<class TMatcher>
bool npoint_mlpack::PairwiseNptTraversal<TMatcher>::CanPrunePair_(NptNode* node1, NptNode* node2, 
                                                        bool same_tree) 
{
  
  // Check symmetry
  if (same_tree) {
    if (node2->End() <= node1->Begin()) {
      // we're pruning due to symmetry
      return true;
    }
  }

  // Check if we can prune
  double min_dist_sq = node1->Bound().MinDistance(node2->Bound());
  double max_dist_sq = node1->Bound().MaxDistance(node2->Bound());
  
  //std::cout << "max matcher: " << max_matcher_dist_sq_;
  //std::cout << ", min matcher: " << min_matcher_dist_sq_ << "\n";
  
  return (min_dist_sq > max_matcher_dist_sq_) 
          || (max_dist_sq < min_matcher_dist_sq_);
  //return false;
  
}

template <class TMatcher>
void npoint_mlpack::PairwiseNptTraversal<TMatcher>::FillLists_(NptNode* node1, NptNode* node2, 
                                                               bool same_tree,
                                                     std::vector<std::vector<NptNode*> >& list)
{
 
  //mlpack::Log::Info << "Filling lists.\n";
  
  if (node1->IsLeaf() && node2->IsLeaf()) {
   
    //mlpack::Log::Info << "Filling lists base case.\n";
    
    list[node1->Stat().leaf_id].push_back(node2);
    
    //std::cout << "list size: " << list[node1].size() << "\n";
    
    if (!leaf_list_[node1->Stat().leaf_id]) {
      //mlpack::Log::Info << "Setting leaf list at " << node1->Stat().leaf_id;
      //mlpack::Log::Info << "with node " << node1 << "\n";
      leaf_list_[node1->Stat().leaf_id] = node1;
    }
    
  }
  else if (node1->IsLeaf()) {
    
    NptNode* left_child = node2->Left();
    NptNode* right_child = node2->Right();
    
    if (!CanPrunePair_(node1, left_child, same_tree)) {
      
      FillLists_(node1, left_child, same_tree, list);
      
    }
    else {
      num_prunes_++;
    }

    if (!CanPrunePair_(node1, right_child, same_tree)) {
      
      FillLists_(node1, right_child, same_tree, list);
      
    }
    else {
      num_prunes_++;
    }

  }
  else if (node2->IsLeaf()) {
    
    NptNode* left_child = node1->Left();
    NptNode* right_child = node1->Right();
    
    if (!CanPrunePair_(left_child, node2, same_tree)) {
      
      FillLists_(left_child, node2, same_tree, list);
      
    }
    else {
      num_prunes_++;
    }

    if (!CanPrunePair_(right_child, node2, same_tree)) {
      
      FillLists_(right_child, node2, same_tree, list);
      
    }
    else {
      num_prunes_++;
    }
    
  }
  else {
    // both need to be split
    
    NptNode* left_1 = node1->Left();
    NptNode* right_1 = node1->Right();
    
    NptNode* left_2 = node2->Left();
    NptNode* right_2 = node2->Right();
    
    if (!CanPrunePair_(left_1, left_2, same_tree)) {
      FillLists_(left_1, left_2, same_tree, list);
    }
    else {
      num_prunes_++;
    }

    if (!CanPrunePair_(left_1, right_2, same_tree)) {
      FillLists_(left_1, right_2, same_tree, list);
    }
    else {
      num_prunes_++;
    }

    if (!CanPrunePair_(right_1, left_2, same_tree)) {
      FillLists_(right_1, left_2, same_tree, list);
    }
    else {
      num_prunes_++;
    }

    if (!CanPrunePair_(right_1, right_2, same_tree)) {
      FillLists_(right_1, right_2, same_tree, list);
    }
    else {
      num_prunes_++;
    }

  }
  
} // FillLists_()


// This is the single list case -- i.e we assume B and C come from the same 
// data set
template <class TMatcher>
void npoint_mlpack::PairwiseNptTraversal<TMatcher>::TraverseLists_(std::vector<std::vector<NptNode*> >& node_map)
{
  
  //std::cout << "Traversing list\n";
  typename TMatcher::PartialResult total_result(matcher_.results_size());

  std::vector<typename TMatcher::PartialResult> thread_results(omp_get_max_threads(),
                                                               matcher_.results_size());
  
  // TODO: remove the critical at the end of this
#pragma omp parallel for schedule(dynamic, 1)
  for (long long int leaf_id = 0; leaf_id < num_leaves_; leaf_id++) {
   
    NptNode* nodeA = leaf_list_[leaf_id];
    
    std::vector<NptNode*>& list = node_map[leaf_id];
    
    /*
    #pragma omp critical
    {
      list = node_map[nodeA];
    }
     */
    //std::vector<NptNode*>& list = node_map[nodeA];
    
    //std::cout << "Traversing list, node has " << list.size() << " nodes.\n";
    
    if (list.size() > 0) {
    
      //std::vector<NptNode*>::iterator nodeB_it;
      for (size_t nodeB_it = 0; nodeB_it < list.size(); nodeB_it++) {
        
        NptNode* nodeB = list[nodeB_it];
        
        double min_dist_sq_AB = nodeA->Bound().MinDistance(nodeB->Bound());
        double max_dist_sq_AB = nodeA->Bound().MaxDistance(nodeB->Bound());
        
        std::vector<double> min_dists_sq(3);
        std::vector<double> max_dists_sq(3);
        
        min_dists_sq[0] = min_dist_sq_AB;
        max_dists_sq[0] = max_dist_sq_AB;
        
        // IMPORTANT: there may be a problem here if we count a point with 
        // itself,
        // shouldn't come up, though
        std::vector<NptNode*> base_cases;
        
        //std::vector<NptNode*>::iterator nodeC_it;
        // TODO: move the single base case to here?
        for (size_t nodeC_it = nodeB_it; nodeC_it < list.size(); nodeC_it++) {
          
          NptNode* nodeC = list[nodeC_it];
          
          double min_dist_sq_AC = nodeA->Bound().MinDistance(nodeC->Bound());
          double max_dist_sq_AC = nodeA->Bound().MaxDistance(nodeC->Bound());

          double min_dist_sq_BC = nodeB->Bound().MinDistance(nodeC->Bound());
          double max_dist_sq_BC = nodeB->Bound().MaxDistance(nodeC->Bound());
          
          min_dists_sq[1] = min_dist_sq_AC;
          min_dists_sq[2] = min_dist_sq_BC;
          
          max_dists_sq[1] = max_dist_sq_AC;
          max_dists_sq[2] = max_dist_sq_BC;

          if (!CanPrune_(min_dists_sq, max_dists_sq)) {
            
            //mlpack::Log::Info << "Found a full base case.\n";
            //BaseCase_(nodeA, nodeB, nodeC);
            base_cases.push_back(nodeC);
            
          } // can we prune it?
          //else {
          //  num_prunes_++;
          //}

        }// loop over C
        
        //std::cout << "Doing base case with " << base_cases.size() << " C's.\n";
        // we've collected all the C's that go with this A B pair, now run them
        if (base_cases.size() > 0) {
          BaseCase_(nodeA, nodeB, base_cases, thread_results[omp_get_thread_num()]);
          num_base_cases_++;
        }
          
      }// loop over B
      
    } // if there is work to do
    
    //mlpack::Log::Info << "Found " << my_result.result << " results in single list base case.\n";

    /*
#pragma omp critical
    {
      total_result += my_result;
      my_result.Reset();
    }
     */
  } // loop over A (parallel)
 
  for (int i = 0; i < omp_get_max_threads(); i++)
  {
    total_result += thread_results[i];
  }
  
  matcher_.AddResult(total_result);
  
} // TraverseLists_

template<class TMatcher>
bool npoint_mlpack::PairwiseNptTraversal<TMatcher>::CanPrune_(std::vector<double>& min_dists_sq,
                                                              std::vector<double>& max_dists_sq)
{
  /*
  std::cout << "Pruning: (" << min_dist_AB << ", " << max_dist_AB << "), (";
  std::cout << min_dist_AC << ", " << max_dist_AC << "), (";
  std::cout << min_dist_BC << ", " << max_dist_BC << ")\n";
   */
  // Symmetry should already have been checked
  bool ret_val =  !(matcher_.TestNodeTuple(min_dists_sq,
                                           max_dists_sq));
  
  //std::cout << "Result: " << ret_val << "\n\n";
  
  return ret_val;
  //return false;
  
}

template <class TMatcher>
void npoint_mlpack::PairwiseNptTraversal<TMatcher>::TraverseLists_(std::vector<std::vector<NptNode*> >& node_mapB,
                                                         std::vector<std::vector<NptNode*> >& node_mapC)
{
  
  // iterate through the list
  typename TMatcher::PartialResult total_result(matcher_.results_size());
  
  std::vector<typename TMatcher::PartialResult> thread_results(omp_get_max_threads(),
                                                               matcher_.results_size());
  
#pragma omp parallel for schedule(dynamic, 1) 
  for (long long int leaf_id = 0; leaf_id < num_leaves_; leaf_id++) {

    NptNode* nodeA = leaf_list_[leaf_id];

    std::vector<NptNode*>& list_B = node_mapB[leaf_id];
    std::vector<NptNode*>& list_C = node_mapC[leaf_id];
    
    /*
    #pragma omp critical
    {
      list_B = node_mapB[nodeA];
      list_C = node_mapC[nodeA];
    }
     */
    
    //std::cout << "Traversing list, node has " << list_B.size() << " B nodes";
    //std::cout << "and " << list_C.size() << " C nodes.\n";
    
    if (list_B.size() > 0 && list_C.size() > 0) {
      
      //std::vector<NptNode*>::iterator nodeB_it;
      for (size_t nodeB_it = 0; nodeB_it < list_B.size(); nodeB_it++) {
        
        NptNode* nodeB = list_B[nodeB_it];
        
        double min_dist_sq_AB = nodeA->Bound().MinDistance(nodeB->Bound());
        double max_dist_sq_AB = nodeA->Bound().MaxDistance(nodeB->Bound());
        
        std::vector<double> min_dists_sq(3);
        std::vector<double> max_dists_sq(3);
        
        min_dists_sq[0] = min_dist_sq_AB;
        max_dists_sq[0] = max_dist_sq_AB;
        
        // IMPORTANT: there may be a problem here if we count a point with 
        // itself,
        // shouldn't come up, though
        std::vector<NptNode*> base_cases;
        
        //std::vector<NptNode*>::iterator nodeC_it;
        for (size_t nodeC_it = 0; nodeC_it < list_C.size(); nodeC_it++) {
          
          NptNode* nodeC = list_C[nodeC_it];
          
          double min_dist_sq_AC = nodeA->Bound().MinDistance(nodeC->Bound());
          double max_dist_sq_AC = nodeA->Bound().MaxDistance(nodeC->Bound());
          
          double min_dist_sq_BC = nodeB->Bound().MinDistance(nodeC->Bound());
          double max_dist_sq_BC = nodeB->Bound().MaxDistance(nodeC->Bound());
          
          min_dists_sq[1] = min_dist_sq_AC;
          min_dists_sq[2] = min_dist_sq_BC;
          
          max_dists_sq[1] = max_dist_sq_AC;
          max_dists_sq[2] = max_dist_sq_BC;
          
          if (!CanPrune_(min_dists_sq, max_dists_sq)) {
            
            //BaseCase_(nodeA, nodeB, nodeC);
            base_cases.push_back(nodeC);
            
          } // can we prune it?
          //else {
          //  num_prunes_++;
          //}
          
        }// loop over C
        
        //std::cout << "Doing base case with " << base_cases.size() << " C's.\n";
        // we've collected all the C's that go with this A B pair, now run them
        BaseCase_(nodeA, nodeB, base_cases, thread_results[omp_get_thread_num()]);
        // this won't be updated correctly when using multiple threads
        //num_base_cases_++;
        
      }// loop over B
      
    } // do we have any work to do
/*
#pragma omp critical
    {
      total_result += my_result;
      my_result.Reset();
    }
  */
  } // loop over A (parallel)
  
  for (int i = 0; i < omp_get_max_threads(); i++)
  {
    total_result += thread_results[i];
  }
  
  matcher_.AddResult(total_result);

} // TraverseLists

template<class TMatcher>
void npoint_mlpack::PairwiseNptTraversal<TMatcher>::BaseCase_(NptNode* nodeA,
                                                    NptNode* nodeB,
                                                    std::vector<NptNode*>& nodeC_list,
                                                              typename TMatcher::PartialResult& result)
{

  matcher_.ComputeBaseCase(nodeA, nodeB, nodeC_list, result);
  
}

template<class TMatcher>
int npoint_mlpack::PairwiseNptTraversal<TMatcher>::num_prunes() const {
  return num_prunes_;
}

template<class TMatcher>
int npoint_mlpack::PairwiseNptTraversal<TMatcher>::num_base_cases() const {
  return num_base_cases_;
}

template<class TMatcher>
long long int npoint_mlpack::PairwiseNptTraversal<TMatcher>::num_point_tuples_considered() const {
  return num_point_tuples_considered_;
}

template<class TMatcher>
long long int npoint_mlpack::PairwiseNptTraversal<TMatcher>::num_node_tuples_considered() const {
  return num_node_tuples_considered_;
}

template<class TMatcher>
long long int npoint_mlpack::PairwiseNptTraversal<TMatcher>::num_pairwise_distances_computed() const {
  return num_pairwise_distances_computed_;
}

template<class TMatcher>
void npoint_mlpack::PairwiseNptTraversal<TMatcher>::PrintStats()
{
  
  mlpack::Log::Info << "\nTwo-Stage kd-tree Traversal Stats (may be incorrect for parallel): \n";
  mlpack::Log::Info << "num prunes: " << num_prunes_ << "\n";
  mlpack::Log::Info << "num base cases: " << num_base_cases_ << "\n";
  mlpack::Log::Info << "num pairwise distances computed: " << num_pairwise_distances_computed_ << "\n";
  mlpack::Log::Info << "num triples considered: " << num_point_tuples_considered_ << "\n\n";
  
  
}


#endif
