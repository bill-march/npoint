/**
 * resampling_splitter.hpp
 *
 * @author Bill March (march@gatech.edu)
 *
 * Definitions for ResamplingSplitter class.
 */

#include "resampling_splitter.hpp"


npoint_mlpack::ResamplingSplitter::ResamplingSplitter(arma::mat& data,
                                                      arma::colvec& weights,
                                                      int num_x_regions,
                                                      int num_y_regions,
                                                      int num_z_regions,
                                                      ResamplingHelper& helper)
:
resampling_helper_(helper),
num_resampling_regions_(num_x_regions * num_y_regions * num_z_regions),
data_all_mat_(data.memptr(), data.n_rows, data.n_cols, false),
data_all_weights_(weights),
data_mats_(num_resampling_regions_),
data_weights_(num_resampling_regions_),
num_x_partitions_(num_x_regions),
num_y_partitions_(num_y_regions),
num_z_partitions_(num_z_regions),
x_step_(resampling_helper_.x_size() / (double)num_x_partitions_),
y_step_(resampling_helper_.y_size() / (double)num_y_partitions_),
z_step_(resampling_helper_.z_size() / (double)num_z_partitions_),
num_points_(data.n_cols),
num_points_per_region_(num_resampling_regions_, 0)
{
  
  for (size_t i = 0; i < num_resampling_regions_; i++) {
    
    data_mats_[i] = new arma::mat;
    data_weights_[i] = new arma::colvec;
    
  }
  
  SplitData_();
  
} // constructor

npoint_mlpack::ResamplingSplitter::~ResamplingSplitter()
{
  
  // free the data and weights
  for (size_t i = 0; i < num_resampling_regions_; i++) {
    
    if (data_mats_[i]) {
      delete data_mats_[i];
      data_mats_[i] = NULL;
    }
    if (data_weights_[i]) {
      delete data_weights_[i];
      data_weights_[i] = NULL;
    }
    
  } // loop over resampling regions
  
} // destructor


int npoint_mlpack::ResamplingSplitter::FindRegion_(arma::colvec& col)
{

  //int x_ind = floor((col(0) + resampling_helper_.epsilon() - resampling_helper_.x_min()) / (x_step_ + resampling_helper_.epsilon()));
  //int y_ind = floor((col(1) + resampling_helper_.epsilon() - resampling_helper_.y_min()) / (y_step_ + resampling_helper_.epsilon()));
  //int z_ind = floor((col(2) + resampling_helper_.epsilon() - resampling_helper_.z_min()) / (z_step_ + resampling_helper_.epsilon()));
  
  int x_ind = (int)floor((col(0) - resampling_helper_.x_min() ) / x_step_);
  int y_ind = (int)floor((col(1) - resampling_helper_.y_min() ) / y_step_);
  int z_ind = (int)floor((col(2) - resampling_helper_.z_min() ) / z_step_);
  
  int region_ind = (x_ind + num_x_partitions_ * y_ind
                    + num_x_partitions_ * num_y_partitions_ * z_ind);

  assert(region_ind >= 0);
  assert(region_ind < (int)num_resampling_regions_);
  
  return region_ind;
  
}


void npoint_mlpack::ResamplingSplitter::SplitData_()
{
  
  std::vector<std::vector<size_t> > points_in_region_id(num_resampling_regions_);
  
  // loop over the points, determine which region each belongs to
  for (size_t i = 0; i < num_points_; i++)
  {
    
    arma::colvec col_i;
    col_i = data_all_mat_.col(i);
    
    // This won't work because of stupid armadillo crap
    //int region_id = FindRegion_(data_all_mat_.col(i));
    
    int region_id = FindRegion_(col_i);
    
    num_points_per_region_[region_id]++;
    points_in_region_id[region_id].push_back(i);
    
  }
  
  // now, loop over regions and insert all the points in them
  // allocate space for them first
  for (size_t region_ind = 0; region_ind < num_resampling_regions_;
       region_ind++)
  {
    
    data_mats_[region_ind]->resize(3, num_points_per_region_[region_ind]);
    
    for (size_t point_ind = 0; point_ind < num_points_per_region_[region_ind];
         point_ind++)
    {
      
      data_mats_[region_ind]->col(point_ind) = data_all_mat_.col(points_in_region_id[region_ind][point_ind]);
      
    }
    
  }
  
} // SplitData

std::vector<arma::mat*>& npoint_mlpack::ResamplingSplitter::data_mats() 
{
  return data_mats_;
}

std::vector<arma::colvec*>& npoint_mlpack::ResamplingSplitter::data_weights() 
{
  return data_weights_;
}

