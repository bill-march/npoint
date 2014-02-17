//
//  naive_resampling_driver.hpp
//  contrib_march
//
//  Created by William March on 6/13/12.
//  Copyright (c) 2012 Georgia Institute of Technology. All rights reserved.
//

#ifndef __MLPACK_METHODS_NPOINT_RESAMPLING_CLASSES_NAIVE_RESAMPLING_DRIVER_HPP
#define __MLPACK_METHODS_NPOINT_RESAMPLING_CLASSES_NAIVE_RESAMPLING_DRIVER_HPP

#include "../infrastructure/node_tuple.hpp"
#include "../matchers/matcher_arguments.hpp"
#include "../infrastructure/resampling_helper.hpp"
#include "resampling_splitter.hpp"

#include <omp.h>

namespace npoint_mlpack {
  
  // For now, I'm assuming that these go together properly
  // i.e. the main makes sure that it doesn't create a resampling class
  // with a bad combination of templates
  template <class TMatcher, class TTraversal, class TTree, class TResults>
  class NaiveResamplingDriver {
    
  private:
    
    size_t num_resampling_regions_;
    
    // we don't need this - just store it in the individual ones
    //arma::mat data_all_mat_;
    //arma::colvec data_all_weights_;
    
    arma::mat random_mat_;
    arma::colvec random_weights_;
    TTree* random_tree_;
    
    // These are the data that live in individual 
    std::vector<arma::mat*> data_mats_;
    std::vector<arma::colvec*> data_weights_;
    
    // Thread safe versions
    
    std::vector<arma::mat*> thread_data_mats_;
    std::vector<arma::colvec*> thread_data_weights_;
    std::vector<TTree*> thread_data_trees_;
    // This is the region the thread currently has
    std::vector<size_t> thread_region_id_;
    
    // randoms for each thread
    std::vector<arma::mat*> thread_random_mats_;
    std::vector<arma::colvec*> thread_random_weights_;
    std::vector<TTree*> thread_random_trees_;
    
    // The regions that the last computation is working on
    // this is what tells us what to get next
    size_t previous_region_;
    size_t previous_num_random_;
    
    
    MatcherArguments matcher_args_;
    
    bool owns_data_;
    
    TResults results_;
    
    // Resampling params
    
    size_t tuple_size_;
    
    int leaf_size_;
    
    bool do_random_;
    
    int num_x_partitions_;
    int num_y_partitions_;
    int num_z_partitions_;

    ResamplingHelper resampling_helper_;
    
    double x_step_;
    double y_step_;
    double z_step_;
    
    size_t num_points_;
    
    int total_computations_;
    
    // Some driver stats
    long long int num_prunes_;
    long long int num_base_cases_;
    long long int num_pairs_considered_;
    long long int num_triples_considered_;
    
    
    //////////// Functions ////////////////
    
    int FindRegion_(arma::colvec& col);
    
    //void SplitData_(arma::mat& data, arma::colvec& weights);
    
    void GetNextData_(int thread_id, size_t region_needed);
    
    void BuildTree_(TTree** tree, arma::mat& data);
    
    //void BuildTree_(Octree* tree, arma::mat& data);
    
    // all lists have length tuple_size
    // for naive, this_region is filled with the region that's being excluded
    void Next_(std::vector<arma::mat*>& this_comp_mats,
               std::vector<arma::colvec*>& this_comp_weights,
               std::vector<TTree*>& this_comp_trees,
               std::vector<int>& this_region,
               int& this_num_randoms);
    
  public:
    
    // version with data already split - don't need to know anything about
    // the sizes or number of resampling regions
    // except, we still need to know the number to make the results class
    // We can get the number from the length of the data mats vector
    NaiveResamplingDriver(std::vector<arma::mat*>& data_mats_in,
                          std::vector<arma::colvec*>& data_weights_in,
                          std::vector<arma::mat* >& data_ghost_mats_in,
                          std::vector<arma::colvec*>& data_ghost_weights_in,
                          arma::mat& random, arma::colvec& rweights,
                          arma::mat& random_ghost, arma::colvec& rweights_ghost,
                          MatcherArguments& matcher_args,
                          int tuple_size,
                          int leaf_size);

    // pre-split, no ghosts
    NaiveResamplingDriver(std::vector<arma::mat*>& data_mats_in,
                          std::vector<arma::colvec*>& data_weights_in,
                          arma::mat& random, arma::colvec& rweights,
                          MatcherArguments& matcher_args,
                          int tuple_size,
                          int leaf_size);

    NaiveResamplingDriver(arma::mat& data, arma::colvec& weights,
                          arma::mat& random, arma::colvec& rweights,
                          MatcherArguments& matcher_args,
                          int num_x_regions, int num_y_regions,
                          int num_z_regions,
                          ResamplingHelper& helper,
                          int tuple_size,
                          int leaf_size = 20);
    
    ~NaiveResamplingDriver();
    
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
  
} // npt


#include "naive_resampling_driver_impl.hpp"


#endif
