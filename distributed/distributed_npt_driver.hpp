/**
 * distributed_npt_driver.hpp
 * @author Bill March (march@gatech.edu)
 *
 * Headers for distributed driver.
 */

#ifndef _NPOINT_MLPACK_DISTRIBUTED_DISTRIBUTED_NPT_DRIVER_HPP_
#define _NPOINT_MLPACK_DISTRIBUTED_DISTRIBUTED_NPT_DRIVER_HPP_

#include <mlpack/core.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>
#include "../resampling_classes/resampling_splitter.hpp"
#include "../results/single_results.hpp"
#include "../results/multi_results.hpp"
#include "../results/angle_results.hpp"
#include "arma_serialization.hpp"
#include "../infrastructure/resampling_helper.hpp"

namespace npoint_mlpack
{

using namespace boost::mpi;
  
/**
 * Sits in between the top level main and the resampling drivers.
 * Each process has an arbitrary subset of the data and randoms.  First, we 
 * shuffle the points so that each process has a local subset, including all
 * points that it needs to do its part of the computation.  Then, each 
 * process does its computation on the points it owns, and we collect the 
 * results on the master process.
 *
 * Templated by the driver that we'll use for the local computation.
 */
template <class TResamplingDriver, class TResults>
class DistributedNptDriver
{
  
private:
  
  communicator comm_;
  
  arma::mat data_all_mat_;
  arma::colvec data_all_weights_;
  
  arma::mat random_mat_;
  arma::colvec random_weights_;
  
  ResamplingSplitter* point_splitter_;
  ResamplingSplitter* ghost_splitter_;
  
  int num_resampling_regions_;
  
  bool do_random_;
  
  std::vector<arma::mat*> data_mats_;
  std::vector<arma::colvec*> data_weights_;
  
  arma::mat ghost_data_all_mat_;
  arma::colvec ghost_data_all_weights_;
  
  arma::mat ghost_random_mat_;
  arma::colvec ghost_random_weights_;
  
  std::vector<arma::mat*> ghost_data_mats_;
  std::vector<arma::colvec*> ghost_data_weights_;
  
  int tuple_size_;
  
  int leaf_size_;
  
  int num_x_regions_;
  int num_y_regions_;
  int num_z_regions_;
  
  double x_step_;
  double y_step_;
  double z_step_;

  int num_x_processes_;
  int num_y_processes_;
  int num_z_processes_;
  
  double x_proc_step_;
  double y_proc_step_;
  double z_proc_step_;
  
  size_t num_points_;
  size_t num_randoms_;
  
  size_t num_ghost_points_;
  size_t num_ghost_randoms_;
  
  MatcherArguments matcher_args_;
  double max_matcher_;
  
  ResamplingHelper resampling_helper_;
  
  //! The results for my part of the computation
  TResults results_;
    
  ///////////////////// functions //////////////////////////
  
  int FindProcess_(arma::colvec& col);

  void FindPointsOwners_(arma::mat& data_in, arma::colvec& weights_in,
                         arma::mat& randoms_in, arma::colvec& rweights_in);

  void ShufflePoints_(std::vector<size_t>& points_per_process,
                      std::vector<arma::mat>& points_to_send,
                      std::vector<arma::colvec>& weights_to_send,
                      arma::mat& destination_mat,
                      arma::colvec& destination_weights);

  void FindGhostPointsOwners_(arma::mat& data_in, arma::colvec& weights_in,
                              arma::mat& randoms_in, arma::colvec& rweights_in);

  void FindGhostProcess_(int my_rank, arma::colvec& col,
                         std::set<int>& processes);

  void GatherResults_(TResults& result);
  
  
public:
  
  DistributedNptDriver(communicator& comm,
                       arma::mat& data_in, arma::colvec& weights_in,
                       arma::mat& randoms_in, arma::colvec& rweights_in,
                       MatcherArguments& matcher_args,
                       ResamplingHelper& resampling_helper,
                       int num_x_processes,
                       int num_y_processes,
                       int num_z_processes,
                       int num_x_regions,
                       int num_y_regions,
                       int num_z_regions,
                       int tuple_size,
                       int leaf_size = 20);

  ~DistributedNptDriver();
  
  TResults& results();
  
  void Compute();
  
  
}; // class

} // namespace


#include "distributed_npt_driver_impl.hpp"


#endif
