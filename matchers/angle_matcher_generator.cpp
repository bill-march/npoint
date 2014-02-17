//
//  angle_matcher_generator.cpp
//  contrib_march
//
//  Created by William March on 9/25/12.
//
//

#include "angle_matcher_generator.hpp"

void npoint_mlpack::AngleMatcherGenerator::FillInMatchers_()
{
  
  for (unsigned int r1_ind = 0; r1_ind < short_sides_.size(); r1_ind++)
  {
    
    for (unsigned int theta_ind = 0; theta_ind < thetas_.size(); theta_ind++)
    {
      
      double r1_lo = short_sides_[r1_ind] - 0.5 * bin_size_ * short_sides_[r1_ind];
      double r1_hi = short_sides_[r1_ind] + 0.5 * bin_size_ * short_sides_[r1_ind];
      
      double r2_val = short_sides_[r1_ind] * long_sides_;
      double r2_lo = r2_val - 0.5 * bin_size_ * r2_val;
      double r2_hi = r2_val + 0.5 * bin_size_ * r2_val;
      
      double r3_val = sqrt(short_sides_[r1_ind] * short_sides_[r1_ind]
                           + r2_val * r2_val
                           - 2 * short_sides_[r1_ind] *
                                      r2_val * cos(thetas_[theta_ind]));
      double r3_lo = r3_val - 0.5 * bin_size_ * r3_val;
      double r3_hi = r3_val + 0.5 * bin_size_ * r3_val;
      
      arma::mat lower_matcher(3,3);
      lower_matcher << 0.0 << r1_lo << r2_lo << arma::endr
                    << r1_lo << 0.0 << r3_lo << arma::endr
                    << r2_lo << r3_lo << 0.0 << arma::endr;
      
      arma::mat upper_matcher(3,3);
      upper_matcher << 0.0 << r1_hi << r2_hi << arma::endr
                    << r1_hi << 0.0 << r3_hi << arma::endr
                    << r2_hi << r3_hi << 0.0 << arma::endr;
      
      //std::cout << "lower_matcher\n" << lower_matcher << "\n\n";
      //std::cout << "upper_matcher\n" << upper_matcher << "\n\n";
      
      
      matcher_lower_bounds_.push_back(lower_matcher);
      matcher_upper_bounds_.push_back(upper_matcher);
      
    } // loop over thetas
    
  } // loop over r1
    
} // FillInMatchers_

npoint_mlpack::AngleMatcherGenerator::AngleMatcherGenerator()
{
  
  // don't want to do anything in this case, not going to use it
  
}

// Need this because we'll have to initialize this in matcherArguments late
// sometimes
void npoint_mlpack::AngleMatcherGenerator::Init(std::vector<double>& short_sides,
                                                double long_side,
                                                std::vector<double>& thetas,
                                                double bin_size)
{
 
  short_sides_ = short_sides;
  long_sides_ = long_side;
  thetas_ = thetas;
  bin_size_ = bin_size;
  
  FillInMatchers_();
  
}

npoint_mlpack::AngleMatcherGenerator::AngleMatcherGenerator(std::vector<double>& short_sides,
                                                double long_sides,
                                                std::vector<double>& thetas,
                                                double bin_size)
:
short_sides_(short_sides),
long_sides_(long_sides),
thetas_(thetas),
bin_size_(bin_size)
{
  
  FillInMatchers_();
  
}

arma::mat& npoint_mlpack::AngleMatcherGenerator::lower_matcher(int r1_ind,
                                                               int theta_ind) {

  int array_ind = r1_ind * thetas_.size() + theta_ind;
  return matcher_lower_bounds_[array_ind];
  
}

arma::mat& npoint_mlpack::AngleMatcherGenerator::upper_matcher(int r1_ind,
                                                               int theta_ind) {
  int array_ind = r1_ind * thetas_.size() + theta_ind;
  return matcher_upper_bounds_[array_ind];
}

int npoint_mlpack::AngleMatcherGenerator::num_matchers() {
  return matcher_lower_bounds_.size();
}

const std::vector<arma::mat>&
npoint_mlpack::AngleMatcherGenerator::matcher_lower_bounds() const
{
  return matcher_lower_bounds_;
}

const std::vector<arma::mat>&
npoint_mlpack::AngleMatcherGenerator::matcher_upper_bounds() const
{
  return matcher_upper_bounds_;
}


void npoint_mlpack::AngleMatcherGenerator::Print() {
  
  for (size_t i = 0; i < matcher_lower_bounds_.size(); i++) {
    
    matcher_lower_bounds_[i].print("Lower bound: ");
    matcher_upper_bounds_[i].print("Upper bound: ");
    std::cout << "\n";
    
  }
  
} // Print




