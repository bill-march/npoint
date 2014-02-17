/**
 * @file matcher_arguments.cpp
 * @author Bill March (march@gatech.edu)
 *
 * Definitions for the argument class for all matcher types.
 */


#include "matcher_arguments.hpp"

// empty constructor
npoint_mlpack::MatcherArguments::MatcherArguments()
{}

// Single matcher
npoint_mlpack::MatcherArguments::MatcherArguments(arma::mat& matcher_dists, 
                                                  double bandwidth)
:
tuple_size_(matcher_dists.n_rows),
lower_matcher_(matcher_dists.n_rows, matcher_dists.n_cols),
upper_matcher_(matcher_dists.n_rows, matcher_dists.n_cols),
total_matchers_(1)
{
  
  arg_type_ = SINGLE_MATCHER;
  
  double half_band = 0.5 * bandwidth;
  
  // Now, fill in the distance constraint matrices
  for (size_t i = 0; i < tuple_size_; i++) {
    
    for (size_t j = i+1; j < tuple_size_; j++) {
      
      lower_matcher_(i,j) = matcher_dists(i,j) - half_band;
      upper_matcher_(i,j) = matcher_dists(i,j) + half_band;
      
      lower_matcher_(j,i) = lower_matcher_(i,j);
      upper_matcher_(j,i) = upper_matcher_(i,j);
      
    } // for j
    
  } // for i
  
} // constructor (single matcher)

// Single matcher # 2
npoint_mlpack::MatcherArguments::MatcherArguments(arma::mat& lower_matcher, 
                                                  arma::mat& upper_matcher)
:
tuple_size_(lower_matcher.n_cols),
lower_matcher_(lower_matcher),
upper_matcher_(upper_matcher),
total_matchers_(1)
{

  arg_type_ = SINGLE_MATCHER;
  template_type_ = TEMPLATE_SINGLE_MATCHER;

}

// Angle matcher
npoint_mlpack::MatcherArguments::MatcherArguments(std::vector<double>& short_sides, 
                                                  double long_side,
                                                  std::vector<double>& thetas, 
                                                  double bin_size)
:
tuple_size_(3),
short_sides_(short_sides),
long_side_(long_side),
thetas_(thetas),
bin_size_(bin_size),
total_matchers_(short_sides.size() * thetas.size()),
// only works with three point
angle_generator_(short_sides_,
                 long_side_,
                 thetas_,
                 bin_size_)
{

  arg_type_ = ANGLE_MATCHER;
  template_type_ = TEMPLATE_ANGLE_MATCHER;

}

npoint_mlpack::MatcherArguments::MatcherArguments(double min_r1,
                                                  double max_r1,
                                                  int num_r1,
                                                  double min_theta,
                                                  double max_theta,
                                                  int num_theta,
                                                  double long_side,
                                                  double bin_size)
:
tuple_size_(3),
short_sides_(num_r1),
long_side_(long_side),
thetas_(num_theta),
bin_size_(bin_size),
total_matchers_(num_r1 * num_theta),
angle_generator_()
{
  
  arg_type_ = ANGLE_MATCHER;
  template_type_ = TEMPLATE_ANGLE_MATCHER;
  
  // fill in the arrays
  FillAngleVectors_(min_r1, max_r1, num_r1,
                    min_theta, max_theta, num_theta);
  
  angle_generator_.Init(short_sides_, long_side_,
                        thetas_, bin_size_);
  
}

void npoint_mlpack::MatcherArguments::FillAngleVectors_(double r1_min,
                                                        double r1_max,
                                                        int num_r1,
                                                        double theta_min,
                                                        double theta_max,
                                                        int num_theta)
{
  
  // dividing by zero if only one r1
  double r1_step = 1.0;
  if (num_r1 > 1) {
    r1_step = (r1_max - r1_min) / ((double)num_r1 - 1.0);
  }
  
  if (num_r1 > 1) {
    for (int i = 0; i < num_r1; i++) {
      short_sides_[i] = r1_min + (double)i * r1_step;
    }
  }
  else {
    short_sides_[0] = r1_min;
  }
  
  double theta_step = 1.0;
  if (num_theta > 1) {
    theta_step = (theta_max - theta_min) / ((double)num_theta - 1.0);
  }
  
  if (num_theta > 1) {
    for (int i = 0; i < num_theta; i++) {
      thetas_[i] = theta_min + (double)i * theta_step;
    }
  }
  else {
    thetas_[0] = theta_min;
  }
  
  
} // FillVectors_()



// Multi matcher
npoint_mlpack::MatcherArguments::MatcherArguments(arma::mat& matcher_mat,
                                                  int tuple_size,
                                                  bool do_off)
:
tuple_size_(tuple_size),
min_bands_(matcher_mat.n_rows),
max_bands_(matcher_mat.n_rows),
num_bands_(matcher_mat.n_rows),
total_matchers_(1),
do_off_diagonal_(do_off)
{

  arg_type_ = MULTI_MATCHER;
  template_type_ = TEMPLATE_MULTI_MATCHER;

  for (unsigned int i = 0; i < matcher_mat.n_rows; i++) {
    
    min_bands_[i] = matcher_mat(i,0);
    max_bands_[i] = matcher_mat(i,1);
    num_bands_[i] = (int)matcher_mat(i,2);
    total_matchers_ *= num_bands_[i];
    
  }
  
  if (!do_off_diagonal_) {
    total_matchers_ = (int)matcher_mat(0,2);
  }

  // Have to do this here because we don't know the mins and maxes when this is
  // first called
  generator_.Init(min_bands_, max_bands_, num_bands_, tuple_size_);
  
} // multi matcher constructor

npoint_mlpack::MatcherArguments::MatcherArguments(std::vector<arma::mat>& lower_bounds,
                 std::vector<arma::mat>& upper_bounds)
:
tuple_size_(lower_bounds[0].n_cols),
lower_matcher_list_(lower_bounds),
upper_matcher_list_(upper_bounds),
arg_type_(UNORDERED_MULTI_MATCHER),
template_type_(TEMPLATE_MULTI_MATCHER),
total_matchers_(lower_bounds.size())
{
  
  // no generators for now
  
}


npoint_mlpack::MatcherArguments npoint_mlpack::MatcherArguments::
GenerateMatcher(int matcher_ind)
{
  
  if (arg_type_ == MULTI_MATCHER)
  {
    
    // This just uses the generator
    arma::mat& lower_matcher = generator_.lower_matcher(matcher_ind);
    arma::mat& upper_matcher = generator_.upper_matcher(matcher_ind);
    
    MatcherArguments args(lower_matcher, upper_matcher);
    
    return args;
    
  }
  else if (arg_type_ == ANGLE_MATCHER)
  {
    
    // need to turn matcher ind into r1 ind and theta ind
    int r1_ind = matcher_ind / thetas_.size();
    int theta_ind = matcher_ind % thetas_.size();
    
    arma::mat& lower_matcher = angle_generator_.lower_matcher(r1_ind,
                                                              theta_ind);
    arma::mat& upper_matcher = angle_generator_.upper_matcher(r1_ind,
                                                              theta_ind);
    
    MatcherArguments args(lower_matcher, upper_matcher);
    
    return args;
    
  }
  else if (arg_type_ == UNORDERED_MULTI_MATCHER) {
    
    arma::mat& lower_matcher = lower_matcher_list_[matcher_ind];
    arma::mat& upper_matcher = upper_matcher_list_[matcher_ind];
    
    MatcherArguments args(lower_matcher, upper_matcher);
    
    return args;
    
  }
  else
  {
    
    mlpack::Log::Fatal << "Calling GenerateMatcher with unsupported matcher type.\n";
  
    MatcherArguments args;
    
    return args;
    
  }

} // GenerateMatcher

arma::mat& npoint_mlpack::MatcherArguments::LowerMatcher(int matcher_ind)
{
  
  if (arg_type_ == MULTI_MATCHER)
  {
    
    // This just uses the generator
    return generator_.lower_matcher(matcher_ind);
    
  }
  else if (arg_type_ == ANGLE_MATCHER)
  {
    
    // need to turn matcher ind into r1 ind and theta ind
    int r1_ind = matcher_ind / thetas_.size();
    int theta_ind = matcher_ind % thetas_.size();
    
    return angle_generator_.lower_matcher(r1_ind, theta_ind);
    
  }
  else if (arg_type_ == UNORDERED_MULTI_MATCHER) {
    
    return lower_matcher_list_[matcher_ind];
    
  }
  else
  {
    
    mlpack::Log::Fatal << "Calling LowerMatcher with unsupported matcher type.\n";
    
    // need this so the compiler doesn't complain
    return lower_matcher_;
    
  }
  
} // LowerMatcher

arma::mat& npoint_mlpack::MatcherArguments::UpperMatcher(int matcher_ind)
{
  
  if (arg_type_ == MULTI_MATCHER)
  {
    
    // This just uses the generator
    return generator_.upper_matcher(matcher_ind);
    
  }
  else if (arg_type_ == ANGLE_MATCHER)
  {
    
    // need to turn matcher ind into r1 ind and theta ind
    int r1_ind = matcher_ind / thetas_.size();
    int theta_ind = matcher_ind % thetas_.size();
    
    return angle_generator_.upper_matcher(r1_ind, theta_ind);
    
  }
  else if (arg_type_ == UNORDERED_MULTI_MATCHER) {
    
    return upper_matcher_list_[matcher_ind];
    
  }
  else
  {
    
    mlpack::Log::Fatal << "Calling Upper with unsupported matcher type.\n";
    
    // need this so the compiler doesn't complain
    return upper_matcher_;
    
  }
  
} // LowerMatcher

npoint_mlpack::MatcherArguments npoint_mlpack::MatcherArguments::
GenerateMatcher(int r1_ind, int theta_ind)
{
  
  if (arg_type_ != ANGLE_MATCHER)
  {
    
    mlpack::Log::Fatal << "Calling GenerateMatcher with unsupported matcher type.\n";
    
  }
  
  arma::mat& lower_matcher = angle_generator_.lower_matcher(r1_ind, theta_ind);
  arma::mat& upper_matcher = angle_generator_.upper_matcher(r1_ind, theta_ind);
  
  MatcherArguments args(lower_matcher, upper_matcher);
  
  return args;
  
}

npoint_mlpack::MatcherArguments npoint_mlpack::MatcherArguments::
Generate2ptMatchers()
{
  
  if (arg_type_ != ANGLE_MATCHER)
  {
    mlpack::Log::Fatal << "Calling Generate2ptMatchers with unsupported matcher type.\n";
  }
  
  std::vector<arma::mat> lower_matcher_list;
  std::vector<arma::mat> upper_matcher_list;
  
  // first, store the r1 values (upper and lower)
 
  double half_thickness = bin_size_ / 2.0;
  
  for (unsigned int i = 0; i < short_sides_.size(); i++)
  {
    
    double lower_val = (1.0 - half_thickness) * short_sides_[i];
    double upper_val = (1.0 + half_thickness) * short_sides_[i];
    
    arma::mat lower_mat;
    lower_mat << 0.0 << lower_val << arma::endr << lower_val << 0.0 << arma::endr;
    lower_matcher_list.push_back(lower_mat);
    
    arma::mat upper_mat;
    upper_mat << 0.0 << upper_val << arma::endr << upper_val << 0.0 << arma::endr;
    upper_matcher_list.push_back(upper_mat);
    
    double lower_val_r2 = (1.0 - half_thickness) * short_sides_[i] * long_side_;
    double upper_val_r2 = (1.0 + half_thickness) * short_sides_[i] * long_side_;

    arma::mat lower_mat_r2;
    lower_mat_r2 << 0.0 << lower_val_r2 << arma::endr << lower_val_r2 << 0.0 << arma::endr;
    lower_matcher_list.push_back(lower_mat_r2);
    
    arma::mat upper_mat_r2;
    upper_mat_r2 << 0.0 << upper_val_r2 << arma::endr << upper_val_r2 << 0.0 << arma::endr;
    upper_matcher_list.push_back(upper_mat_r2);
    
    // now, for the thetas
    for (unsigned int j = 0; j < thetas_.size(); j++)
    {
      
      double r3_val = sqrt(short_sides_[i] * short_sides_[i]
      + long_side_ * short_sides_[i] * long_side_ * short_sides_[i]
      - 2.0 * short_sides_[i] * long_side_ * short_sides_[i] * cos(thetas_[j]));
      
      double lower_dist_r3 = (1.0 - half_thickness) * r3_val;
      double upper_dist_r3 = (1.0 + half_thickness) * r3_val;
      
      arma::mat lower_mat_r3;
      lower_mat_r3 << 0.0 << lower_dist_r3 << arma::endr << lower_dist_r3 << 0.0 << arma::endr;
      lower_matcher_list.push_back(lower_mat_r3);
      
      arma::mat upper_mat_r3;
      upper_mat_r3 << 0.0 << upper_dist_r3 << arma::endr << upper_dist_r3 << 0.0 << arma::endr;
      upper_matcher_list.push_back(upper_mat_r3);
      
    } // loop over thetas
    
  } // loop over r1
  
  MatcherArguments retval(lower_matcher_list, upper_matcher_list);
  return retval;
  
} // generate 2pt matcher


size_t npoint_mlpack::MatcherArguments::tuple_size() const
{
  return tuple_size_;
}

// Note that the accessors have some safeguards to make sure you're using the
// MatcherArguments correctly.  However, these aren't comprehensive and aren't
// foolproof, so be careful.  
const arma::mat& npoint_mlpack::MatcherArguments::lower_matcher() const
{
  if (arg_type_ != SINGLE_MATCHER) {
    mlpack::Log::Fatal << "Calling SINGLE_MATCHER accessors for a different kind of argument.\n";
  }
  return lower_matcher_;
}

const arma::mat& npoint_mlpack::MatcherArguments::upper_matcher() const
{

  if (arg_type_ != SINGLE_MATCHER) {
    mlpack::Log::Fatal << "Calling SINGLE_MATCHER accessors for a different kind of argument.\n";
  }
  return upper_matcher_;
}

arma::mat& npoint_mlpack::MatcherArguments::lower_matcher()
{
  if (arg_type_ != SINGLE_MATCHER) {
    mlpack::Log::Fatal << "Calling SINGLE_MATCHER accessors for a different kind of argument.\n";
  }
  return lower_matcher_;
}

arma::mat& npoint_mlpack::MatcherArguments::upper_matcher()
{
  
  if (arg_type_ != SINGLE_MATCHER) {
    mlpack::Log::Fatal << "Calling SINGLE_MATCHER accessors for a different kind of argument.\n";
  }
  return upper_matcher_;
}

const std::vector<double>& npoint_mlpack::MatcherArguments::min_bands() const
{
  if (arg_type_ != MULTI_MATCHER) {
    mlpack::Log::Fatal << "Calling MULTI_MATCHER accessors for a different kind of argument.\n";
  }
  return min_bands_;
}

const std::vector<double>& npoint_mlpack::MatcherArguments::max_bands() const
{
  if (arg_type_ != MULTI_MATCHER) {
    mlpack::Log::Fatal << "Calling MULTI_MATCHER accessors for a different kind of argument.\n";
  }
  return max_bands_;

}

const std::vector<int>& npoint_mlpack::MatcherArguments::num_bands() const
{
  if (arg_type_ != MULTI_MATCHER) {
    mlpack::Log::Fatal << "Calling MULTI_MATCHER accessors for a different kind of argument.\n";
  }
  return num_bands_;

}

const std::vector<double>& npoint_mlpack::MatcherArguments::short_sides() const
{
  if (arg_type_ != ANGLE_MATCHER) {
    mlpack::Log::Fatal << "Calling ANGLE_MATCHER accessors for a different kind of argument.\n";
  }
  return short_sides_;
}

double npoint_mlpack::MatcherArguments::long_side() const
{
  if (arg_type_ != ANGLE_MATCHER) {
    mlpack::Log::Fatal << "Calling ANGLE_MATCHER accessors for a different kind of argument.\n";
  }
  return long_side_;
}

const std::vector<double>& npoint_mlpack::MatcherArguments::thetas() const
{
  if (arg_type_ != ANGLE_MATCHER) {
    mlpack::Log::Fatal << "Calling ANGLE_MATCHER accessors for a different kind of argument.\n";
  }
  return thetas_;
}

double npoint_mlpack::MatcherArguments::bin_size() const
{
  if (arg_type_ != ANGLE_MATCHER) {
    mlpack::Log::Fatal << "Calling ANGLE_MATCHER accessors for a different kind of argument.\n";
  }
  return bin_size_;
}

std::vector<arma::mat>& npoint_mlpack::MatcherArguments::lower_matcher_list() 
{
  if (arg_type_ != UNORDERED_MULTI_MATCHER) {
    mlpack::Log::Fatal << "Calling UNORDERED_MULTI_MATCHER accessors for a different kind of argument.\n";
  }
  return lower_matcher_list_;
}

std::vector<arma::mat>& npoint_mlpack::MatcherArguments::upper_matcher_list() 
{
  if (arg_type_ != UNORDERED_MULTI_MATCHER) {
    mlpack::Log::Fatal << "Calling UNORDERED_MULTI_MATCHER accessors for a different kind of argument.\n";
  }
  return upper_matcher_list_;
}

npoint_mlpack::MatcherArguments::MatcherTemplateType npoint_mlpack::MatcherArguments::template_type()
{
  return template_type_;
}


void npoint_mlpack::MatcherArguments::set_template_type(npoint_mlpack::MatcherArguments::MatcherTemplateType type)
{
  template_type_ = type;
}

int npoint_mlpack::MatcherArguments::total_matchers() const
{
  return total_matchers_;
}

npoint_mlpack::MatcherArguments::MatcherArgumentType npoint_mlpack::MatcherArguments::arg_type()
{
  return arg_type_;
}

// Returns the value of the largest matcher upper bound
double npoint_mlpack::MatcherArguments::max_matcher() const
{
  
  double max_val = 0.0;
  
  if (arg_type_ == SINGLE_MATCHER)
  {
    
    for (unsigned int i = 0; i < upper_matcher_.n_rows; i++)
    {
      
      for (unsigned int j = i+1; j < upper_matcher_.n_cols; j++)
      {
        
        max_val = std::max(max_val, upper_matcher_(i,j));
        
      }
      
    }
    
  }
  else if (arg_type_ == MULTI_MATCHER)
  {
    
    if (!do_off_diagonal_) {
      max_val = max_bands_[0];
    }
    else {
      for (unsigned int i = 0; i < max_bands_.size(); i++) {
        max_val = std::max(max_val, max_bands_[i]);
      }
    }
    
  }
  else if (arg_type_ == ANGLE_MATCHER)
  {
    
    double r1_max = short_sides_.back();
    double r2_max = long_side_ * r1_max;
    // IMPORTANT: this may not be right if the largest theta is small
    double r3_max = sqrt(r1_max * r1_max + r2_max * r2_max - 2.0 * r1_max * r2_max * cos(thetas_.back()));
    
    r1_max += 0.5 * bin_size_ * r1_max;
    r2_max += 0.5 * bin_size_ * r2_max;
    r3_max += 0.5 * bin_size_ * r3_max;
    
    double this_max = std::max(r1_max, r2_max);
    max_val = std::max(this_max, r3_max);
    
    //max_val = 20.0;
    
    //std::cout << "angle max val: " << max_val << "\n";
    
  }
  else if (arg_type_ == UNORDERED_MULTI_MATCHER)
  {
    
    for (unsigned int i = 0; i < upper_matcher_list_.size(); i++)
    {
      
      const arma::mat& this_mat = upper_matcher_list_[i];
      
      for (unsigned int j = 0; j < tuple_size_; j++)
      {
        
        for (unsigned int k = j+1; k < tuple_size_; k++)
        {
          
          max_val = std::max(max_val, this_mat(j,k));
          
        } // for k
        
      } // for j
      
    } // for i
    
  }
  else {
    
    mlpack::Log::Fatal << "Other matcher types not yet supported.\n";
    return -1.0;
    
  }
  
  return max_val;
  
}

bool npoint_mlpack::MatcherArguments::do_off_diagonal() const
{
  return do_off_diagonal_;
}

void npoint_mlpack::MatcherArguments::set_off_diagonal(bool do_them)
{
  do_off_diagonal_ = do_them;
}



