//
//  angle_matcher_generator.hpp
//  contrib_march
//
//  Created by William March on 9/25/12.
//
//

#ifndef _NPOINT_MLPACK_MATCHERS_ANGLE_MATCHER_GENERATOR_HPP_
#define _NPOINT_MLPACK_MATCHERS_ANGLE_MATCHER_GENERATOR_HPP_

#include <mlpack/core.hpp>

namespace npoint_mlpack
{

  class AngleMatcherGenerator
  {

  private:
    
    const static int tuple_size_ = 3;
    
    std::vector<double> short_sides_;
    double long_sides_;
    std::vector<double> thetas_;
    double bin_size_;
    
    std::vector<arma::mat> matcher_lower_bounds_;
    std::vector<arma::mat> matcher_upper_bounds_;
    
    std::vector<std::vector<double> > matcher_dists_;
    
    void FillInMatchers_();
    

  public:
    
    // empty constructor for when we aren't going to use the generator
    AngleMatcherGenerator();
    
    AngleMatcherGenerator(std::vector<double>& short_sides,
                          double long_sides,
                          std::vector<double>& thetas,
                          double bin_size);
    
    void Init(std::vector<double>& short_sides,
              double long_side,
              std::vector<double>& thetas,
              double bin_size);
    
    arma::mat& lower_matcher(int r1_ind, int theta_ind);
    
    arma::mat& upper_matcher(int r1_ind, int theta_ind);
    
    const std::vector<arma::mat>& matcher_lower_bounds() const;
    
    const std::vector<arma::mat>& matcher_upper_bounds() const;
    
    int num_matchers();
    
    void Print();
    
  }; // class

} // namespace

#endif // inclusion guards
