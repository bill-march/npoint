//
//  efficient_resampling_driver.hpp
//  contrib_march
//
//  Created by William March on 7/9/12.
//  Copyright (c) 2012 Georgia Institute of Technology. All rights reserved.
//

#ifndef __MLPACK_NPOINT_RESAMPLING_CLASSES_EFFICIENT_RESAMPLING_DRIVER_HPP_
#define __MLPACK_NPOINT_RESAMPLING_CLASSES_EFFICIENT_RESAMPLING_DRIVER_HPP_

#include "../matchers/matcher_arguments.hpp"
#include "../infrastructure/resampling_helper.hpp"
#include "resampling_splitter.hpp"

#include <omp.h>

namespace npoint_mlpack {

  template <class TMatcher, class TTraversal, class TTree, class TResults>
  class EfficientResamplingDriver {
    
  private:
    
    arma::mat data_all_mat_;
    arma::colvec data_all_weights_;
    
    arma::mat random_mat_;
    arma::colvec random_weights_;
    
    size_t num_resampling_regions_;
    
    bool do_random_;
    
    TTree* random_tree_;
    
    std::vector<TTree*> data_trees_;
    
    std::vector<arma::mat*> data_mats_;
    std::vector<arma::colvec*> data_weights_;
    
    // We need to be able to handle ghost points in the distributed case

    // true if we need to handle ghost points
    bool have_ghosts_;
    
    std::vector<TTree*> ghost_data_trees_;
    std::vector<arma::mat*> ghost_data_mats_;
    std::vector<arma::colvec*> ghost_data_weights_;
    
    TTree* ghost_random_tree_;
    
    arma::mat ghost_random_mat_;
    arma::colvec ghost_random_weights_;
    
    int tuple_size_;
    
    int leaf_size_;
    
    int num_x_partitions_;
    int num_y_partitions_;
    int num_z_partitions_;
    
    double x_step_;
    double y_step_;
    double z_step_;
    
    ResamplingHelper resampling_helper_;
    
    int num_points_;
    
    TResults results_;
  
    std::vector<size_t> current_tuple_sets_;
    
    MatcherArguments& matcher_args_;
    
    //! True if this class owns the pointers to the individual jackknife
    //! subsamples of data, false otherwise.  This is needed for the
    //! constructor that passes in already partitioned data.
    bool owns_data_;
    
    int total_computations_;
    
    // Some driver stats
    long long int num_prunes_;
    long long int num_base_cases_;
    long long int num_pairs_considered_;
    long long int num_triples_considered_;
    
    ////////// functions ////////////
    
    int FindRegion_(arma::colvec& col);

    //void SplitData_();
    
    void BuildTrees_();

    void BuildTree_(TTree** tree, arma::mat& data);
    
    void Next_(std::vector<arma::mat*>& this_comp_mats,
               std::vector<arma::colvec*>& this_comp_weights,
               std::vector<TTree*>& this_comp_trees,
               std::vector<int>& this_region,
               int& this_num_randoms);
    
    void IncrementCurrentTupleSets_();
    
    void IncrementCurrentTupleSetsHelper_(int i);
    

    
  public:

    // version with data already split - don't need to know anything about
    // the sizes or number of resampling regions
    // except, we still need to know the number to make the results class
    // We can get the number from the length of the data mats vector
    // IMPORTANT: is this class now responsible for freeing the memory for the
    // data matrices? if so, the destructor currently does this
    EfficientResamplingDriver(std::vector<arma::mat*>& data_mats_in,
                              std::vector<arma::colvec*>& data_weights_in,
                              std::vector<arma::mat* >& data_ghost_mats_in,
                              std::vector<arma::colvec*>& data_ghost_weights_in,
                              arma::mat& random, arma::colvec& rweights,
                              arma::mat& random_ghost,
                              arma::colvec& rweights_ghost,
                              MatcherArguments& matcher_args,
                              int tuple_size,
                              int leaf_size);

    // no ghosts, pre-split data
    EfficientResamplingDriver(std::vector<arma::mat*>& data_mats_in,
                              std::vector<arma::colvec*>& data_weights_in,
                              arma::mat& random, arma::colvec& rweights,
                              MatcherArguments& matcher_args,
                              int tuple_size,
                              int leaf_size);

    // data haven't been split yet
    EfficientResamplingDriver(arma::mat& data, arma::colvec& weights,
                              arma::mat& random, arma::colvec& rweights,
                              MatcherArguments& matcher_args,
                              int num_x_regions, int num_y_regions, 
                              int num_z_regions,
                              ResamplingHelper& helper,
                              int tuple_size,
                              int leaf_size = 20);
    
    ~EfficientResamplingDriver();
    
    // TODO: should these still be public (or even exist)?
    
    arma::mat* random_mat();
    
    arma::colvec* random_weights();
    
    // TODO: do I rebuild this every time? 
    TTree* random_tree();
    
    // a flag so the Drivers know how to handle results
    bool is_efficient();
    
    int total_num_computations();
    
    // actually does the computation
    void Compute();
    
    void PrintRegionSizes() const;
    
    void PrintResults();
    
    TResults& results();

  }; // class

} // namespace


#include "efficient_resampling_driver_impl.hpp"

#endif

