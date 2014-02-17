/*
 *  node_tuple.hpp
 *  
 *
 *  Created by William March on 2/14/11.
 *
 */

#ifndef __MLPACK_METHODS_NPOINT_INFRASTRUCTURE_NODE_TUPLE_HPP
#define __MLPACK_METHODS_NPOINT_INFRASTRUCTURE_NODE_TUPLE_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/tree/bounds.hpp>
#include <mlpack/core/tree/binary_space_tree.hpp>

/*
 *  This class holds an ordered tuple of n tree nodes (for an n-point 
 *  calculation).  It can be viewed as a node in the recursion tree during 
 *  a computation.  
 *
 *  This class is responsible for the symmetry in the computation.
 *
 *  TODO: think about how to make this more efficient by incorporating threading
 *  Also, look out for overhead from the construction.
 *
 */

namespace npoint_mlpack {
  
  // This only exists for the parallel pairwise traversal, we need 
  // an index for each node
  class NptStat {
    
  public:
    
    // is negative if not a leaf
    long long int leaf_id;
    
    NptStat();
    
    NptStat(const NptStat& other);
    
    NptStat(mlpack::tree::BinarySpaceTree<mlpack::bound::HRectBound<2, false>,
            NptStat>& node);

    NptStat(const arma::mat& /* dataset */,
            const size_t /* begin */, 
            const size_t /* count */);
    
    NptStat(const arma::mat& /* dataset */,
              const size_t /* begin */,
              const size_t /* count */,
              const NptStat& /* leftStat */,
            const NptStat& /* rightStat */);
    
  }; // NptStat

  // Need the false here because I want squared distances
  typedef mlpack::tree::BinarySpaceTree<mlpack::bound::HRectBound<2, false>,
                                        NptStat> NptNode;
  
  // this is now just responsible for checking symmetry
  class NodeTuple { 
    
  private:
    
    std::vector<NptNode*> node_list_;
    
    // IMPORTANT: this doesn't get updated after the first split
    // That's ok, it doesn't store whether the nodes in this tuple are the same
    // it stores whether they came from the same data set
    std::vector<int> same_nodes_;
    
    int tuple_size_;
    
    // this is the position of the node we should split next
    size_t ind_to_split_;
    
    bool all_leaves_;
    
    
    //////////////// functions ///////////////////
    
    void UpdateSplitInd_();

    
  public:
    
    // constructor - only use this one to make the original node tuple
    // at the start of the algorithm
    // The copy constructor will be used for the others
    NodeTuple(std::vector<NptNode*>& list_in);
    
    // use this constructor to make children in the recursion
    NodeTuple(NodeTuple& parent, bool is_left);
    
    const std::vector<NptNode*>& get_node_list() const;
    
    const std::vector<int>& get_same_nodes() const;
    
    bool all_leaves();
    
    size_t ind_to_split() const;
    
    NptNode*& node_list(size_t i);
    
    size_t tuple_size() const;
    
    int NumPossibleTuples();
    
    int NumPairwiseDistances();
    
    bool CheckSymmetry(size_t split_ind, bool is_left);

    
  }; // class
  
} // namespace


#endif
