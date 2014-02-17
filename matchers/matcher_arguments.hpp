/**
 * @file matcher_arguments.hpp
 * @author Bill March (march@gatech.edu)
 *
 * Header for the argument class for all matcher types.
 */


#ifndef __MLPACK_METHODS_NPOINT_MATCHERS_MATCHER_ARGUMENTS_HPP
#define __MLPACK_METHODS_NPOINT_MATCHERS_MATCHER_ARGUMENTS_HPP


#include <mlpack/core.hpp>
#include "multi_matcher_generator.hpp"
#include "angle_matcher_generator.hpp"

namespace npoint_mlpack {
  
  /**
   * A generic class for all matchers.  Takes in the command line arguments and 
   * stores them in a general format so that the templated drivers can use the 
   * same constructor for all matchers.  Knows what kind of matcher (single,
   * multi, angle) it is for, and what kind of matcher it will be passed to.
   * The distinction here is that a multi matcher may be filled in by 
   * repeatedly calling single matchers.  The matcher argument knows if this is
   * the case.
   */
  class MatcherArguments {
    
    
  public:
  
    /**
     * This is the type of argument the matcher has.  
     *
     * SINGLE_MATCHER: only one set of distance constraints.
     * MULTI_MATCHER: several sets of distance constraints - specified by 
     * min val, max val, num vals for each of the (n choose 2) distances
     * ANGLE_MATCHER: several sets of constraints - specified by a range of
     * r1 values, a multiplier for r2, and a range of angles.  Only works for 
     * 3pt.
     */
    enum MatcherArgumentType
    {
      SINGLE_MATCHER,
      MULTI_MATCHER,
      ANGLE_MATCHER,
      UNORDERED_MULTI_MATCHER
    };
    
    /**
     * This is the matcher type used in the template of the driver that this 
     * argument is being called for. We need this to be able to solve a multi
     * matcher by iterating through many calls to single matchers.
     */
    enum MatcherTemplateType
    {
      TEMPLATE_SINGLE_MATCHER,
      TEMPLATE_MULTI_MATCHER,
      TEMPLATE_ANGLE_MATCHER,
      TEMPLATE_EFFICIENT_MATCHER
    };
    
  private:
    
    //! n
    size_t tuple_size_;

    //! For a single matcher, these are the distance constraints
    arma::mat lower_matcher_;
    arma::mat upper_matcher_;
    
    //! For a multi matcher, these are the distance constraints
    //! Each of these vectors is of length (n choose 2), with one value for 
    //! each dimension of the matcher.
    std::vector<double> min_bands_;
    std::vector<double> max_bands_;
    //! The number of separate distance values in each dimension of the matcher
    std::vector<int> num_bands_;
    
    //! For an angle matcher, these are the distance constraints
    std::vector<double> short_sides_;
    double long_side_;
    std::vector<double> thetas_;
    double bin_size_;
    
    //! For unordered multi matcher
    std::vector<arma::mat> lower_matcher_list_;
    std::vector<arma::mat> upper_matcher_list_;
    
    //! This is the type of argument, listed above. This will be set
    //! automatically in the constructor.
    MatcherArgumentType arg_type_;

    //! This is the type of driver template, listed above.
    //! This needs to be set by the driver.
    MatcherTemplateType template_type_;
    
    //! The total number of matchers to compute
    int total_matchers_;
    
    //! The generator class - this is for turning multi matcher inputs into the 
    //! three vector format used above.
    MultiMatcherGenerator generator_;
    
    //! Generator class for angle matchers
    AngleMatcherGenerator angle_generator_;
    
    //! For multi-matcher, do we care about only equilateral shapes?
    bool do_off_diagonal_;
    
    ////////// Helper functions ///////////////////
    
    void FillAngleVectors_(double r1_min, double r1_max, int num_r1,
                           double theta_min, double theta_max, int num_theta);

    
  public:
    
    /**
     * Default constructor.  Only used to make error messages work - use it and
     * you'll break things.
     */
    MatcherArguments();
    
    /**
     * Constructor for a single matcher with a common thickness for all 
     * dimensions.
     *
     * @param matcher_dists The distance constraints for the matcher.
     * @param bandwidth The thickness of all dimensions in the matcher.
     */
    MatcherArguments(arma::mat& matcher_dists, double bandwidth);  
    
    /**
     * Constructor for a single matcher with independent constraints in all
     * dimensions.
     *
     * @param lower_matcher The lower bound distances.
     * @param upper_matcher The upper bound distances.
     */
    MatcherArguments(arma::mat& lower_matcher, arma::mat& upper_matcher);
    
    /**
     * Constructor for the angle matchers.  
     *
     * @param short_sides The values of r1
     * @param long_side The multiplier for r2 - i.e. r2 = (long_side) * r1
     * @param thetas The values of theta
     * @param bin_size The thickness factor.  The lower (upper) bound is 
     * r - (+) bin_size*r
     */
    MatcherArguments(std::vector<double>& short_sides, double long_side,
                     std::vector<double>& thetas, double bin_size);
    
    /**
     * Constructor for the angle matchers.  
     *
     * @param min_r1 The smallest value of r1
     * @param max_r1 The largest value of r1
     * @param num_r1 The number of bins to divide (max_r1 - min_r1) into
     * @param min_theta The smallest value of theta
     * @param max_theta The largest value of theta
     * @param num_theta The number of bins to divide (max_theta - min_theta) 
     * @param long_side The multiplier for r2 - i.e. r2 = (long_side) * r1
     * @param bin_size The thickness factor.  The lower (upper) bound is
     * r - (+) bin_size*r / 2
     */
    MatcherArguments(double min_r1, double max_r1, int num_r1,
                     double min_theta, double max_theta, int num_theta,
                     double long_side, double bin_size);

    
    /**
     * Constructor for multi matchers.
     *
     * @param matcher_mat The matrix of matcher info.  Has (n choose 2) rows
     * and 3 columns.  Row i contains r_min, r_max, num_r for the ith 
     * dimension
     * @param tuple_size The value of n
     * @param do_off True if we should do the off diagonal elements (this is 
     * the default value.
     */
    MatcherArguments(arma::mat& matcher_mat, int tuple_size,
                     bool do_off = true);
    
    /**
     * Constructor for unordered multi-matchers
     */
    MatcherArguments(std::vector<arma::mat>& lower_bounds,
                     std::vector<arma::mat>& upper_bounds);
    
    /**
     * For multi matchers, This creates a new MatcherArguments for a single 
     * matcher.
     *
     * @param matcher_ind The single matcher to generate.  Note that the 
     * ordering is the same one the results class will use. When calling this
     * make sure to set the matcher's matcher_ind to the same value.
     */
    MatcherArguments GenerateMatcher(int matcher_ind);
    
    /**
     * For angle matchers, This creates a new MatcherArguments for a single
     * matcher.
     *
     * @param r1_ind The r1 value of the matcher the caller wants
     * @param theta_ind The theta value of the matcher the caller wants
     */
    MatcherArguments GenerateMatcher(int r1_ind, int theta_ind);
    
    /**
     * For an angle matcher, this makes a new matcher arguments that contains 
     * all the two point distances in the original matcher.
     */
    MatcherArguments Generate2ptMatchers();

    arma::mat& LowerMatcher(int matcher_ind);

    arma::mat& UpperMatcher(int matcher_ind);

    
    size_t tuple_size() const;
    
    const arma::mat& lower_matcher() const;
    
    const arma::mat& upper_matcher() const;
    
    const std::vector<double>& min_bands() const;
    
    const std::vector<double>& max_bands() const;
    
    const std::vector<int>& num_bands() const;
    
    const std::vector<double>& short_sides() const;
    
    double long_side() const;
    
    const std::vector<double>& thetas() const;

    double bin_size() const;
    
    arma::mat& lower_matcher();
    
    arma::mat& upper_matcher();
    
    std::vector<arma::mat>& lower_matcher_list();
    
    std::vector<arma::mat>& upper_matcher_list();
    
    MatcherTemplateType template_type();
    
    void set_template_type(MatcherTemplateType type);
    
    MatcherArgumentType arg_type();
    
    int total_matchers() const;
    
    double max_matcher() const;
    
    bool do_off_diagonal() const;
    
    void set_off_diagonal(bool do_them);
    
  }; // class
  
  
} // namespace




#endif
