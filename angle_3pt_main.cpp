/**
 * @file angle_3pt_main.cpp
 * @author Bill March (march@gatech.edu)
 *
 * Computes raw correlation counts for angle matchers on a single node.  
 *
 * The three point correlation is defined by the lengths of the three sides of a 
 * triangle.  An angle matcher consists of a set of values (r1) for one leg of
 * the triangle, a multiplier l such that the second leg r2 = l * r1, and 
 * a set of values of the angle theta between the two legs.  This program 
 * computes the raw correlation counts for each matcher obtained by a choice of 
 * r1 and theta from the values provided.
 */

#include "resampling_classes/naive_resampling_driver.hpp"
#include "resampling_classes/efficient_resampling_driver.hpp"

#include "matchers/single_matcher.hpp"
#include "matchers/angle_matcher.hpp"
#include "matchers/efficient_angle_matcher.hpp"
#include "matchers/efficient_cpu_matcher.hpp"
#include "matchers/3_point_single_matcher.hpp"

#include "infrastructure/generate_random_problem.hpp"
#include "infrastructure/generic_npt_alg.hpp"
#include "infrastructure/pairwise_npt_traversal.hpp"
#include "infrastructure/resampling_helper.hpp"

#include "results/angle_results.hpp"
#include "results/multi_results.hpp"

#include "matchers/matcher_arguments.hpp"

#include <mlpack/core.hpp>

PROGRAM_INFO("3-point correlation estimation using angle matchers.",
             "The three point correlation is defined by the lengths of the "
             "three sides of a triangle.  An angle matcher consists of a set "
             "of values (r1) for one leg of "
             "the triangle, a multiplier l such that the second leg r2 = l * r1, "
             "and a set of values of the angle theta between the two legs. "
             "This program computes the raw correlation counts for each "
             "matcher obtained by a choice of r1 and theta from the values "
             "provided.  It uses a single node with optional jackknife resampling.");
PARAM_STRING_REQ("data", "Point coordinates.", "d");
PARAM_STRING("random", "Optional Poisson set coordinates.", "r", "fake");
PARAM_INT("num_random", "If random isn't specified, this will generate the given number of points", "R", 0);
//PARAM_DOUBLE("box_x_length", "Length of the box containing the data in the x direction.", "a", 1.0);
//PARAM_DOUBLE("box_y_length", "Length of the box containing the data in the y direction.", "b", 1.0);
//PARAM_DOUBLE("box_z_length", "Length of the box containing the data in the z direction.", "c", 1.0);
PARAM_INT("leaf_size", "Maximum number of points in a leaf node.", "c", 32);
PARAM_INT("num_x_regions", "Number of regions to divide the input into along the x coordinate.",
          "x", 1);
PARAM_INT("num_y_regions", "Number of regions to divide the input into along the y coordinate.",
          "y", 1);
PARAM_INT("num_z_regions", "Number of regions to divide the input into along the z coordinate.",
          "z", 1);
//PARAM_INT("num_threads", "Total number of threads to use.  Leaving it as 0 will cause the system to use the default number.  This has no effect for the single thread code.", "T", -1);
PARAM_STRING("resampler", "Select the type of resampling to use.  Options are \"naive\" or \"efficient\".", "s", "efficient");
PARAM_STRING("traversal", "Select the type of tree traversal to use.  Options are \"multi\" and \"pairwise\".", "t", "pairwise");
PARAM_STRING("kernel", "Select the type of base case computation.  Options are \"general\" (the old version for general n), \"simple\" for a specialized 3-point case, and \"efficient\" for the hardware optimized version.", "k", "efficient");

PARAM_STRING("matcher", "Select how to handle the multiple matchers. Options are \"single\" (multiple traversals, one for each matcher), and \"angle\" (one traversal, consider all matchers simulataneously.", "M", "angle");
PARAM_STRING_REQ("matchers", "A 3 column, 3 row csv, row 0 is r1min, max, num_r1, row 1 is same for theta, row 2 is r2_mult, bin_thick, garbage", "m");

using namespace mlpack;
using namespace npoint_mlpack;

int main(int argc, char* argv[])
{
  
  // Command Line Parsing and setup
  CLI::ParseCommandLine(argc, argv);
  
  omp_set_nested(1);
  /*
  int num_threads = CLI::GetParam<int>("num_threads");
  if (num_threads > 0) {
    omp_set_num_threads(num_threads);
  }
   */
  omp_set_num_threads(1);
  
  //////////////////////////////////////////////////////////////////////.
  // Resampler info
  // Tree info
  int leaf_size = CLI::GetParam<int>("leaf_size");
  
  // Resampler info
  //double box_x_length = CLI::GetParam<double>("box_x_length");
  //double box_y_length = CLI::GetParam<double>("box_y_length");
  //double box_z_length = CLI::GetParam<double>("box_z_length");
  
  int num_x_regions = CLI::GetParam<int>("num_x_regions");
  int num_y_regions = CLI::GetParam<int>("num_y_regions");
  int num_z_regions = CLI::GetParam<int>("num_z_regions");
  
  ///////////////////////////////////////////////////////////////////////
  
  // Loading the data and randoms
  std::string data_filename = CLI::GetParam<std::string>("data");
  arma::mat data_in, data_mat;
  data_in.load(data_filename, arma::raw_ascii);
  
  // THIS IS BAD: do it better
  if (data_in.n_rows > data_in.n_cols) {
    data_mat = arma::trans(data_in);
  }
  else {
    data_mat = data_in;
  }
  data_in.reset();
  
  ResamplingHelper helper(data_mat);
  
  // We don't do weights yet.
  arma::colvec weights;
  weights.set_size(data_mat.n_cols);
  weights.fill(1.0);
  
  arma::mat random_mat;
  arma::colvec random_weights;
  if (CLI::HasParam("random")) {
    
    std::string random_filename = CLI::GetParam<std::string>("random");
    
    arma::mat random_in;
    random_in.load(random_filename, arma::raw_ascii);
    
    // THIS IS BAD: do it better
    if (random_in.n_rows > random_in.n_cols) {
      random_mat = arma::trans(random_in);
    }
    else {
      random_mat = random_in;
    }
    random_in.reset();
    
  } // do we have two sets?
  else {
    
    int num_random = CLI::GetParam<int>("num_random");
    
    random_mat.set_size(3, num_random);
    
    if (num_random > 0) {
      GenerateRandomProblem generator(0.1, 0.5, 0.1, 0.5,
                                      num_random, num_random+1,
                                      helper);
      generator.GenerateRandomSet(random_mat);
    }
    
  }
  random_weights.set_size(random_mat.n_cols);
  random_weights.fill(1.0);
  
  /////////////////////////////////////////////////////////////////////
  
  // Reading in matcher info
  std::string matcher_filename = CLI::GetParam<std::string>("matchers");
  arma::mat matcher_mat;
  matcher_mat.load(matcher_filename, arma::raw_ascii);
  
  MatcherArguments matcher_args(matcher_mat(0,0), matcher_mat(0,1),
                                matcher_mat(0,2),
                                matcher_mat(1,0), matcher_mat(1,1),
                                matcher_mat(1,2),
                                matcher_mat(2,0), matcher_mat(2,1));
  
  int tuple_size = 3;
  
  ////////////////////////////////////////////////////////////////////
  // Do the algorithm
  
  AngleResults* results = new AngleResults();
  
  std::string resampler_str = CLI::GetParam<std::string>("resampler");
  std::string traversal_str = CLI::GetParam<std::string>("traversal");
  std::string kernel_str = CLI::GetParam<std::string>("kernel");
  std::string matcher_str = CLI::GetParam<std::string>("matcher");
  
  if (CLI::GetParam<std::string>("resampler") == "efficient")
  {
    // efficient resampling
    
    // traversals
    if (CLI::GetParam<std::string>("traversal") == "pairwise")
    {
      // pairwise traversal
      
      if (CLI::GetParam<std::string>("kernel") == "efficient")
      {
        // efficient kernel
        
        // angle matcher
        if (CLI::GetParam<std::string>("matcher") == "angle") {
          
          Log::Info << "Doing Efficient Resampling, Pairwise Traversal, Efficient Kernel, Angle Matcher computation.\n";
          
          EfficientResamplingDriver<EfficientAngleMatcher,
          PairwiseNptTraversal<EfficientAngleMatcher>,
          NptNode, AngleResults>
          driver(data_mat,
                 weights,
                 random_mat,
                 random_weights,
                 matcher_args,
                 num_x_regions,
                 num_y_regions,
                 num_z_regions,
                 helper,
                 tuple_size,
                 leaf_size);
          
          driver.Compute();
          
          *results = driver.results();
          
        }
        else {
          // single matcher
          
          Log::Info << "Doing Efficient Resampling, Pairwise Traversal, Efficient Kernel, Single Matcher computation.\n";

          matcher_args.set_template_type(MatcherArguments::TEMPLATE_SINGLE_MATCHER);
          
          EfficientResamplingDriver<EfficientCpuMatcher,
          PairwiseNptTraversal<EfficientCpuMatcher>,
          NptNode, AngleResults>
          driver(data_mat,
                 weights,
                 random_mat,
                 random_weights,
                 matcher_args,
                 num_x_regions,
                 num_y_regions,
                 num_z_regions,
                 helper,
                 tuple_size,
                 leaf_size);
          
          driver.Compute();

          *results = driver.results();
          
        } // single matcher 
        
      } // efficient kernel
      else if (CLI::GetParam<std::string>("kernel") == "simple")
      {
        
        // efficient resampling, pairwise traversal, 3pt simple kernel
        Log::Info << "Doing Efficient Resampling, Pairwise Traversal, Simple Kernel, 3point Single Matcher computation.\n";

        matcher_args.set_template_type(MatcherArguments::TEMPLATE_SINGLE_MATCHER);
        
        EfficientResamplingDriver<ThreePointSingleMatcher,
        PairwiseNptTraversal<ThreePointSingleMatcher>,
        NptNode, AngleResults>
        driver(data_mat,
               weights,
               random_mat,
               random_weights,
               matcher_args,
               num_x_regions,
               num_y_regions,
               num_z_regions,
               helper,
               tuple_size,
               leaf_size);
        
        driver.Compute();
        *results = driver.results();
        
      }
      else {
        // general
        // angle matcher
        if (CLI::GetParam<std::string>("matcher") == "angle") {
          
          Log::Info << "Doing Efficient Resampling, Pairwise Traversal, General Kernel, Angle Matcher computation.\n";
          
          EfficientResamplingDriver<AngleMatcher,
          PairwiseNptTraversal<AngleMatcher>,
          NptNode, AngleResults>
          driver(data_mat,
                 weights,
                 random_mat,
                 random_weights,
                 matcher_args,
                 num_x_regions,
                 num_y_regions,
                 num_z_regions,
                 helper,
                 tuple_size,
                 leaf_size);
          
          driver.Compute();
          
          *results = driver.results();
          
        }
        else {
          // single matcher
          
          Log::Info << "Doing Efficient Resampling, Pairwise Traversal, General Kernel, Single Matcher computation.\n";
          
          matcher_args.set_template_type(MatcherArguments::TEMPLATE_SINGLE_MATCHER);
          
          EfficientResamplingDriver<SingleMatcher,
          PairwiseNptTraversal<SingleMatcher>,
          NptNode, AngleResults>
          driver(data_mat,
                 weights,
                 random_mat,
                 random_weights,
                 matcher_args,
                 num_x_regions,
                 num_y_regions,
                 num_z_regions,
                 helper,
                 tuple_size,
                 leaf_size);
          
          driver.Compute();
          
          *results = driver.results();
          
        } // single matcher

      } // simple kernel
      
    } // pairwise traversal
    else
    {
      // generic multi tree traversal
      if (CLI::GetParam<std::string>("kernel") == "efficient")
      {
        // efficient kernel
        
        // angle matcher
        if (CLI::GetParam<std::string>("matcher") == "angle") {
          
          Log::Info << "Doing Efficient Resampling, Multi-tree Traversal, Efficient Kernel, Angle Matcher computation.\n";
          
          EfficientResamplingDriver<EfficientAngleMatcher,
          GenericNptAlg<EfficientAngleMatcher>,
          NptNode, AngleResults>
          driver(data_mat,
                 weights,
                 random_mat,
                 random_weights,
                 matcher_args,
                 num_x_regions,
                 num_y_regions,
                 num_z_regions,
                 helper,
                 tuple_size,
                 leaf_size);
          
          driver.Compute();
          
          *results = driver.results();
          
        }
        else {
          // single matcher
          
          Log::Info << "Doing Efficient Resampling, Multi-tree Traversal, Efficient Kernel, Single Matcher computation.\n";
          
          matcher_args.set_template_type(MatcherArguments::TEMPLATE_SINGLE_MATCHER);
          
          EfficientResamplingDriver<EfficientCpuMatcher,
          GenericNptAlg<EfficientCpuMatcher>,
          NptNode, AngleResults>
          driver(data_mat,
                 weights,
                 random_mat,
                 random_weights,
                 matcher_args,
                 num_x_regions,
                 num_y_regions,
                 num_z_regions,
                 helper,
                 tuple_size,
                 leaf_size);
          
          driver.Compute();
          
          *results = driver.results();
          
        } // single matcher
        
      } // efficient kernel
      else if (CLI::GetParam<std::string>("kernel") == "simple")
      {
        
        // efficient resampling, pairwise traversal, 3pt simple kernel
        Log::Info << "Doing Efficient Resampling, Multi-tree Traversal, Simple Kernel, 3point Single Matcher computation.\n";
        
        matcher_args.set_template_type(MatcherArguments::TEMPLATE_SINGLE_MATCHER);
        
        EfficientResamplingDriver<ThreePointSingleMatcher,
        GenericNptAlg<ThreePointSingleMatcher>,
        NptNode, AngleResults>
        driver(data_mat,
               weights,
               random_mat,
               random_weights,
               matcher_args,
               num_x_regions,
               num_y_regions,
               num_z_regions,
               helper,
               tuple_size,
               leaf_size);
        
        driver.Compute();
        *results = driver.results();
        
      }
      else {
        // general
        // angle matcher
        if (CLI::GetParam<std::string>("matcher") == "angle") {
          
          Log::Info << "Doing Efficient Resampling, Multi-tree Traversal, General Kernel, Angle Matcher computation.\n";
          
          EfficientResamplingDriver<AngleMatcher,
          GenericNptAlg<AngleMatcher>,
          NptNode, AngleResults>
          driver(data_mat,
                 weights,
                 random_mat,
                 random_weights,
                 matcher_args,
                 num_x_regions,
                 num_y_regions,
                 num_z_regions,
                 helper,
                 tuple_size,
                 leaf_size);
          
          driver.Compute();
          
          *results = driver.results();
          
        }
        else {
          // single matcher
          
          Log::Info << "Doing Efficient Resampling, Multi-tree Traversal, General Kernel, Single Matcher computation.\n";
          
          matcher_args.set_template_type(MatcherArguments::TEMPLATE_SINGLE_MATCHER);
          
          EfficientResamplingDriver<SingleMatcher,
          GenericNptAlg<SingleMatcher>,
          NptNode, AngleResults>
          driver(data_mat,
                 weights,
                 random_mat,
                 random_weights,
                 matcher_args,
                 num_x_regions,
                 num_y_regions,
                 num_z_regions,
                 helper,
                 tuple_size,
                 leaf_size);
          
          driver.Compute();
          *results = driver.results();
          
        } // single matcher
        
      } // simple kernel
      
    } // multi tree traversal
    
  } // efficient resampling
  else {
    // doing naive
    // traversals
    if (CLI::GetParam<std::string>("traversal") == "pairwise")
    {
      // pairwise traversal
      
      if (CLI::GetParam<std::string>("kernel") == "efficient")
      {
        // efficient kernel
        
        // angle matcher
        if (CLI::GetParam<std::string>("matcher") == "angle") {
          
          Log::Info << "Doing Naive Resampling, Pairwise Traversal, Efficient Kernel, Angle Matcher computation.\n";
          
          NaiveResamplingDriver<EfficientAngleMatcher,
          PairwiseNptTraversal<EfficientAngleMatcher>,
          NptNode, AngleResults>
          driver(data_mat,
                 weights,
                 random_mat,
                 random_weights,
                 matcher_args,
                 num_x_regions,
                 num_y_regions,
                 num_z_regions,
                 helper,
                 tuple_size,
                 leaf_size);
          
          driver.Compute();
          
          *results = driver.results();
          
        }
        else {
          // single matcher
          
          Log::Info << "Doing Naive Resampling, Pairwise Traversal, Efficient Kernel, Single Matcher computation.\n";
          
          matcher_args.set_template_type(MatcherArguments::TEMPLATE_SINGLE_MATCHER);
          
          NaiveResamplingDriver<EfficientCpuMatcher,
          PairwiseNptTraversal<EfficientCpuMatcher>,
          NptNode, AngleResults>
          driver(data_mat,
                 weights,
                 random_mat,
                 random_weights,
                 matcher_args,
                 num_x_regions,
                 num_y_regions,
                 num_z_regions,
                 helper,
                 tuple_size,
                 leaf_size);
          
          driver.Compute();
          
          *results = driver.results();
          
        } // single matcher
        
      } // efficient kernel
      else if (CLI::GetParam<std::string>("kernel") == "simple")
      {
        
        // efficient resampling, pairwise traversal, 3pt simple kernel
        Log::Info << "Doing Naive Resampling, Pairwise Traversal, Simple Kernel, 3point Single Matcher computation.\n";
        
        matcher_args.set_template_type(MatcherArguments::TEMPLATE_SINGLE_MATCHER);
        
        NaiveResamplingDriver<ThreePointSingleMatcher,
        PairwiseNptTraversal<ThreePointSingleMatcher>,
        NptNode, AngleResults>
        driver(data_mat,
               weights,
               random_mat,
               random_weights,
               matcher_args,
               num_x_regions,
               num_y_regions,
               num_z_regions,
               helper,
               tuple_size,
               leaf_size);
        
        driver.Compute();
        *results = driver.results();
        
      }
      else {
        // general
        // angle matcher
        if (CLI::GetParam<std::string>("matcher") == "angle") {
          
          Log::Info << "Doing Naive Resampling, Pairwise Traversal, General Kernel, Angle Matcher computation.\n";
          
          NaiveResamplingDriver<AngleMatcher,
          PairwiseNptTraversal<AngleMatcher>,
          NptNode, AngleResults>
          driver(data_mat,
                 weights,
                 random_mat,
                 random_weights,
                 matcher_args,
                 num_x_regions,
                 num_y_regions,
                 num_z_regions,
                 helper,
                 tuple_size,
                 leaf_size);
          
          driver.Compute();
          
          *results = driver.results();
          
        }
        else {
          // single matcher
          
          Log::Info << "Doing Naive Resampling, Pairwise Traversal, General Kernel, Single Matcher computation.\n";
          
          matcher_args.set_template_type(MatcherArguments::TEMPLATE_SINGLE_MATCHER);
          
          NaiveResamplingDriver<SingleMatcher,
          PairwiseNptTraversal<SingleMatcher>,
          NptNode, AngleResults>
          driver(data_mat,
                 weights,
                 random_mat,
                 random_weights,
                 matcher_args,
                 num_x_regions,
                 num_y_regions,
                 num_z_regions,
                 helper,
                 tuple_size,
                 leaf_size);
          
          driver.Compute();
          
          *results = driver.results();
          
        } // single matcher
        
      } // simple kernel
      
    } // pairwise traversal
    else
    {
      // generic multi tree traversal
      if (CLI::GetParam<std::string>("kernel") == "efficient")
      {
        // efficient kernel
        
        // angle matcher
        if (CLI::GetParam<std::string>("matcher") == "angle") {
          
          Log::Info << "Doing Naive Resampling, Multi-tree Traversal, Efficient Kernel, Angle Matcher computation.\n";
          
          NaiveResamplingDriver<EfficientAngleMatcher,
          GenericNptAlg<EfficientAngleMatcher>,
          NptNode, AngleResults>
          driver(data_mat,
                 weights,
                 random_mat,
                 random_weights,
                 matcher_args,
                 num_x_regions,
                 num_y_regions,
                 num_z_regions,
                 helper,
                 tuple_size,
                 leaf_size);
          
          driver.Compute();
          
          *results = driver.results();
          
        }
        else {
          // single matcher
          
          Log::Info << "Doing Naive Resampling, Multi-tree Traversal, Efficient Kernel, Single Matcher computation.\n";
          
          matcher_args.set_template_type(MatcherArguments::TEMPLATE_SINGLE_MATCHER);
          
          NaiveResamplingDriver<EfficientCpuMatcher,
          GenericNptAlg<EfficientCpuMatcher>,
          NptNode, AngleResults>
          driver(data_mat,
                 weights,
                 random_mat,
                 random_weights,
                 matcher_args,
                 num_x_regions,
                 num_y_regions,
                 num_z_regions,
                 helper,
                 tuple_size,
                 leaf_size);
          
          driver.Compute();
          
          *results = driver.results();
          
        } // single matcher
        
      } // efficient kernel
      else if (CLI::GetParam<std::string>("kernel") == "simple")
      {
        
        // efficient resampling, pairwise traversal, 3pt simple kernel
        Log::Info << "Doing Naive Resampling, Multi-tree Traversal, Simple Kernel, 3point Single Matcher computation.\n";
        
        matcher_args.set_template_type(MatcherArguments::TEMPLATE_SINGLE_MATCHER);
        
        NaiveResamplingDriver<ThreePointSingleMatcher,
        GenericNptAlg<ThreePointSingleMatcher>,
        NptNode, AngleResults>
        driver(data_mat,
               weights,
               random_mat,
               random_weights,
               matcher_args,
               num_x_regions,
               num_y_regions,
               num_z_regions,
               helper,
               tuple_size,
               leaf_size);
        
        driver.Compute();
        *results = driver.results();
        
      }
      else {
        // general
        // angle matcher
        if (CLI::GetParam<std::string>("matcher") == "angle") {
          
          Log::Info << "Doing Naive Resampling, Multi-tree Traversal, General Kernel, Angle Matcher computation.\n";
          
          NaiveResamplingDriver<AngleMatcher,
          GenericNptAlg<AngleMatcher>,
          NptNode, AngleResults>
          driver(data_mat,
                 weights,
                 random_mat,
                 random_weights,
                 matcher_args,
                 num_x_regions,
                 num_y_regions,
                 num_z_regions,
                 helper,
                 tuple_size,
                 leaf_size);
          
          driver.Compute();
          
          *results = driver.results();
          
        }
        else {
          // single matcher
          
          Log::Info << "Doing Naive Resampling, Multi-tree Traversal, General Kernel, Single Matcher computation.\n";
          
          matcher_args.set_template_type(MatcherArguments::TEMPLATE_SINGLE_MATCHER);
          
          NaiveResamplingDriver<SingleMatcher,
          GenericNptAlg<SingleMatcher>,
          NptNode, AngleResults>
          driver(data_mat,
                 weights,
                 random_mat,
                 random_weights,
                 matcher_args,
                 num_x_regions,
                 num_y_regions,
                 num_z_regions,
                 helper,
                 tuple_size,
                 leaf_size);
          
          driver.Compute();
          
          *results = driver.results();
          
        } // single matcher
        
      } // simple kernel
      
    } // multi tree traversal
    
  } // naive resampling
  
  //////////////////////////////////////////////////////////////////
  // Process / output results
  
  results->PrintResults();
  
  
  return 0;
  
} // main


