/**
 * @file resampling_helper.cpp
 * @author Bill March (march@gatech.edu)
 */

#include "resampling_helper.hpp"

using namespace npoint_mlpack;

ResamplingHelper::ResamplingHelper(const arma::mat& data)
:
x_min_(DBL_MAX),
x_max_(-DBL_MAX),
y_min_(DBL_MAX),
y_max_(-DBL_MAX),
z_min_(DBL_MAX),
z_max_(-DBL_MAX),
epsilon_(1e-8)
{
 
  Init(data);
  
} // constructor

void ResamplingHelper::Init(const arma::mat& data)
{
  
  epsilon_ = 1e-8;
  
  x_min_ = DBL_MAX;
  x_max_ = -DBL_MAX;

  y_min_ = DBL_MAX;
  y_max_ = -DBL_MAX;

  z_min_ = DBL_MAX;
  z_max_ = -DBL_MAX;

  // Find the min and max extent in each dimension
  for (size_t i = 0; i < data.n_cols; i++)
  {
    
    x_min_ = std::min(data(0,i), x_min_);
    x_max_ = std::max(data(0,i), x_max_);
    
    y_min_ = std::min(data(1,i), y_min_);
    y_max_ = std::max(data(1,i), y_max_);
    
    z_min_ = std::min(data(2,i), z_min_);
    z_max_ = std::max(data(2,i), z_max_);
    
  }
  
  x_min_ = x_min_ - epsilon_;
  x_max_ = x_max_ + epsilon_;
  
  y_min_ = y_min_ - epsilon_;
  y_max_ = y_max_ + epsilon_;

  z_min_ = z_min_ - epsilon_;
  z_max_ = z_max_ + epsilon_;
  
}


ResamplingHelper::ResamplingHelper(const ResamplingHelper& other)
:
x_min_(other.x_min()),
x_max_(other.x_max()),
y_min_(other.y_min()),
y_max_(other.y_max()),
z_min_(other.z_min()),
z_max_(other.z_max()),
epsilon_(other.epsilon())
{} // constructor

ResamplingHelper::ResamplingHelper()
:
x_min_(DBL_MAX),
x_max_(-DBL_MAX),
y_min_(DBL_MAX),
y_max_(-DBL_MAX),
z_min_(DBL_MAX),
z_max_(-DBL_MAX),
epsilon_(1e-8)
{}


double ResamplingHelper::x_min() const
{
  return x_min_;
}

double ResamplingHelper::x_max() const
{
  return x_max_;
}

double ResamplingHelper::y_min() const
{
  return y_min_;
}

double ResamplingHelper::y_max() const
{
  return y_max_;
}

double ResamplingHelper::z_min() const
{
  return z_min_;
}

double ResamplingHelper::z_max() const
{
  return z_max_;
}

double ResamplingHelper::x_size() const
{
  return x_max_ - x_min_;
}

double ResamplingHelper::y_size() const
{
  return y_max_ - y_min_;
}

double ResamplingHelper::z_size() const
{
  return z_max_ - z_min_;
}

double ResamplingHelper::epsilon() const
{
  return epsilon_;
}




