/**
 * resampling_splitter.hpp
 *
 * @author Bill March (march@gatech.edu)
 *
 * Headers for ResamplingSplitter class.
 */

#ifndef _NPOINT_MLPACK_RESAMPLING_CLASSES_RESAMPLING_SPLITTER_HPP_
#define _NPOINT_MLPACK_RESAMPLING_CLASSES_RESAMPLING_SPLITTER_HPP_

#include <mlpack/core.hpp>
#include "../infrastructure/resampling_helper.hpp"

namespace npoint_mlpack
{
  
  /**
   * Takes in the entire data set and information on jackknife subsamples
   */
  class ResamplingSplitter
  {
   
  private:
    
    ResamplingHelper resampling_helper_;
    
    size_t num_resampling_regions_;
    
    arma::mat data_all_mat_;
    arma::colvec data_all_weights_;
    
    // These are the data that live in individual subsamples
    std::vector<arma::mat*> data_mats_;
    std::vector<arma::colvec*> data_weights_;
    
    int num_x_partitions_;
    int num_y_partitions_;
    int num_z_partitions_;
    
    double x_step_;
    double y_step_;
    double z_step_;
    
    size_t num_points_;
    std::vector<size_t> num_points_per_region_;
    
    /////////// functions /////////////////////
    
    int FindRegion_(arma::colvec& col);
    
    void SplitData_();
    
    
  public:
    
    ResamplingSplitter(arma::mat& data, arma::colvec& weights,
                       int num_x_regions, int num_y_regions, int num_z_regions,
                       ResamplingHelper& resampling_helper);
    
    ~ResamplingSplitter();
    
    std::vector<arma::mat*>& data_mats();
    
    std::vector<arma::colvec*>& data_weights();
    
    double x_step() const 
      {return x_step_;}

    double y_step() const 
      {return y_step_;}

    double z_step() const 
      {return z_step_;}
    
  }; // class
  
} // namespace


#endif
