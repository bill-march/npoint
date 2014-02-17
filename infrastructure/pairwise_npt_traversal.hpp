//
//  pairwise_npt_traversal.hpp
//  
//
//  Created by William March on 4/27/12.
//  Copyright (c) 2012 Georgia Institute of Technology. All rights reserved.
//

#ifndef __MLPACK_METHODS_NPOINT_INFRASTRUCTURE_PAIRWISE_NPT_TRAVERSAL_HPP
#define __MLPACK_METHODS_NPOINT_INFRASTRUCTURE_PAIRWISE_NPT_TRAVERSAL_HPP


// IMPORTANT: this only works for 3 point for now
// also, only works for NptNodes, anything else will require another template
// argument and the tree to have a leaf_id in it's Stat()
#include "node_tuple.hpp"

namespace npoint_mlpack {
  
    
  template<class TMatcher>
  class PairwiseNptTraversal {
    
  private:
    
    // Matcher owns the data
    TMatcher& matcher_;
    
    double min_matcher_dist_sq_;
    double max_matcher_dist_sq_;
    
    // a list of tree roots
    // NOTE: trees_.size() is the number of distinct sets in this computation
    std::vector<NptNode*> trees_;
    
    // how many times should each tree appear in the tuple?
    //std::vector<int> multiplicities_;
    
    // only works on 3 point 
    int tuple_size_;
    
    int num_prunes_;
    int num_base_cases_;
    // for now, this is an estimate of the number of pairwise distances computed
    long long int num_point_tuples_considered_;
    long long int num_node_tuples_considered_;
    long long int num_pairwise_distances_computed_;
    
    // marks if the tree nodes are the same
    bool same_01_;
    bool same_02_;
    bool same_12_;
    
    // starts at 0
    long long int leaf_id_;
    
    long long int num_leaves_;
    std::vector<NptNode*> leaf_list_;
    
    // for each leaf node, we have a list of nodes that may interact with it
    // in the case of a DDR or DRR computation, we'll need two lists
    
    
    
    bool CanPrunePair_(NptNode* node1, NptNode* node2, bool same_tree);
    
    
    void BaseCase_(NptNode* node1, NptNode* node2, NptNode* node3);
    
    
    void FillLists_(NptNode* node1, NptNode* node2, bool same_tree, 
                    std::vector<std::vector<NptNode*> >& list);
    
    void TraverseLists_(std::vector<std::vector<NptNode*> >& node_map);
    
    void TraverseLists_(std::vector<std::vector<NptNode*> >& node_mapB,
                        std::vector<std::vector<NptNode*> >& node_mapC);
    
    bool CanPrune_(std::vector<double>& min_dists_sq,
                   std::vector<double>& max_dists_sq);
    
    void BaseCase_(NptNode* nodeA,
                   NptNode* nodeB,
                   std::vector<NptNode*>& nodeC_list,
                   typename TMatcher::PartialResult& result);
    
    long long int ComputeNumLeaves_(NptNode* tree);
    
  public:
    
    PairwiseNptTraversal(std::vector<NptNode*>& trees_in,
                         TMatcher& matcher_in);
    
    //PairwiseNptTraversal();
    
    int num_prunes() const;
    
    int num_base_cases() const;
    
    long long int num_point_tuples_considered() const;
    
    long long int num_node_tuples_considered() const;
    
    long long int num_pairwise_distances_computed() const;
    
    void Compute();
    
    void PrintStats();
    
  }; // class
  
} // namespace


#include "pairwise_npt_traversal_impl.hpp"

#endif
