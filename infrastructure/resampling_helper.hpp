/**
 * @file resampling_helper.hpp
 * @author Bill March (march@gatech.edu)
 * 
 * A helper class to handle data that live in irregular regions (i.e not the
 * unit cube.
 */

#ifndef NPOINT_MLPACK_INFRASTRUCTURE_RESAMPLING_HELPER_HPP_
#define NPOINT_MLPACK_INFRASTRUCTURE_RESAMPLING_HELPER_HPP_

#include <mlpack/core.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

namespace npoint_mlpack {
  
  class ResamplingHelper {
    
    friend class boost::serialization::access;
    
  private:
    
    double x_min_, x_max_;
    double y_min_, y_max_;
    double z_min_, z_max_;
    
    double epsilon_;
    
  public:

    template<class Archive>
    void serialize(Archive & ar, const unsigned int /*version*/)
    {
      
      ar & x_min_;
      ar & x_max_;

      ar & y_min_;
      ar & y_max_;

      ar & z_min_;
      ar & z_max_;
      
    } // serialization for boost mpi stuff

    ResamplingHelper(const arma::mat& data);
    
    ResamplingHelper(const ResamplingHelper& other);
    
    // for when we won't need to use it
    ResamplingHelper();
    
    void Init(const arma::mat& data);
    
    double x_min() const;
    double x_max() const;
    
    double y_min() const;
    double y_max() const;
    
    double z_min() const;
    double z_max() const;
    
    double x_size() const;
    double y_size() const;
    double z_size() const;
    
    double epsilon() const;
    
  }; // class
  
} // namespace



#endif
