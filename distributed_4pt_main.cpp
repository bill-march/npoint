/**
 * @file 4_point_main.cpp
 * @author Bill March (march@gatech.edu)
 *
 * Does single node 4-point computations.
 */

#include "distributed/distributed_npt_driver.hpp"
#include "resampling_classes/naive_resampling_driver.hpp"
#include "resampling_classes/efficient_resampling_driver.hpp"
#include "matchers/single_matcher.hpp"
#include "matchers/efficient_4pt_matcher.hpp"
#include "infrastructure/generate_random_problem.hpp"
#include "infrastructure/generic_npt_alg.hpp"
#include "results/single_results.hpp"
#include "matchers/matcher_arguments.hpp"
#include "infrastructure/resampling_helper.hpp"

#include <mlpack/core.hpp>

PROGRAM_INFO("Distributed 4-point correlation estimation.", "Does 4pcf correlation counts for a single matcher on many nodes with optional jackknife resampling.");
PARAM_STRING_REQ("data", "Point coordinates.", "d");
PARAM_STRING("random", "Optional Poisson set coordinates.", "r", "fake");
PARAM_INT("num_random", "If random isn't specified, this will generate the given number of points", "R", 0);
PARAM_STRING_REQ("matcher_lower_bounds", "The lower bound distances for the matcher, stored in a symmetric 4x4 matrix..", "l")
PARAM_STRING_REQ("matcher_upper_bounds", "The upper bound distances for the matcher, stored in a symmetric 4x4 matrix..", "u")
//PARAM_DOUBLE("box_x_length", "Length of the box containing the data in the x direction.", "a", 1.0);
//PARAM_DOUBLE("box_y_length", "Length of the box containing the data in the y direction.", "b", 1.0);
//PARAM_DOUBLE("box_z_length", "Length of the box containing the data in the z direction.", "c", 1.0);
PARAM_INT("leaf_size", "Maximum number of points in a leaf node.", "i", 16);
PARAM_INT("num_x_regions", "Number of regions to divide the input into along the x coordinate.",
          "x", 1);
PARAM_INT("num_y_regions", "Number of regions to divide the input into along the y coordinate.",
          "y", 1);
PARAM_INT("num_z_regions", "Number of regions to divide the input into along the z coordinate.",
          "z", 1);
PARAM_INT("num_x_procs", "Number of processes along the x dimension.", "e", 1);
PARAM_INT("num_y_procs", "Number of processes along the y dimension.", "f", 1);
PARAM_INT("num_z_procs", "Number of processes along the z dimension.", "g", 1);
//PARAM_INT("num_threads", "Total number of threads to use.  Leaving it as 0 will cause the system to use the default number.  This has no effect for the single thread code.", "T", -1);

PARAM_STRING("resampler", "Select the type of resampling to use.  Only does efficient right now.", "s", "efficient");
PARAM_STRING("kernel", "Select the type of base case computation.  Options are \"general\" (the old version for general n) and \"efficient\" for the hardware optimized version.", "k", "efficient");

using namespace mlpack;
using namespace npoint_mlpack;

int main(int argc, char* argv[])
{
  
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
  
  CLI::ParseCommandLine(argc, argv);
  
  //mlpack::Log::Info << "Starting parallel computation.\n";
  
  omp_set_nested(1);
  /*
  int num_threads = CLI::GetParam<int>("num_threads");
  if (num_threads > 0) {
    omp_set_num_threads(num_threads);
  }
   */
  omp_set_num_threads(1);
  
  int num_procs = world.size();
  
  // need the three factors of num_procs that are closest together
  // will be between 1 and (num_procs)^{1/3}
  int num_x_procs = CLI::GetParam<int>("num_x_procs");
  int num_y_procs = CLI::GetParam<int>("num_y_procs");
  int num_z_procs = CLI::GetParam<int>("num_z_procs");
  
  if (num_procs != num_x_procs * num_y_procs * num_z_procs) {
    Log::Fatal << "Division of space among processes does not match the number of processes.\n";
    return 1;
  }
  
  //double box_x_length = CLI::GetParam<double>("box_x_length");
  //double box_y_length = CLI::GetParam<double>("box_y_length");
  //double box_z_length = CLI::GetParam<double>("box_z_length");
  
  int leaf_size = CLI::GetParam<int>("leaf_size");
  
  arma::mat matcher_lower_bounds, matcher_upper_bounds;
  
  matcher_lower_bounds.load(CLI::GetParam<std::string>("matcher_lower_bounds"));
  matcher_upper_bounds.load(CLI::GetParam<std::string>("matcher_upper_bounds"));
  
  MatcherArguments matcher_args(matcher_lower_bounds,
                                matcher_upper_bounds);
  
  unsigned int tuple_size = matcher_lower_bounds.n_cols;
  
  if (tuple_size != 4) {
    
    mlpack::Log::Fatal << "Matchers need to be 4 x 4 matrices!\n";
    return 1;
    
  }
  
  if (tuple_size != matcher_upper_bounds.n_cols ||
      tuple_size != matcher_lower_bounds.n_rows) {
    
    mlpack::Log::Fatal << "Upper and lower matchers need to both be 4 x 4!\n";
    return 1;
    
  }
  
  int num_x_regions = CLI::GetParam<int>("num_x_regions");
  int num_y_regions = CLI::GetParam<int>("num_y_regions");
  int num_z_regions = CLI::GetParam<int>("num_z_regions");
  
  // These are the data that this process is responsible for passing to the
  // distributed driver.
  arma::mat data_mat, random_mat;
  arma::colvec data_weights, random_weights;
  
  ResamplingHelper helper;
  
  // root process reads in data, sends out chunks
  if (world.rank() == 0) {
  
    // The total data and random inputs.  These will be split up and scattered
    // to the other processes by the root.
    arma::mat data_all_mat, random_all_mat;
    arma::colvec data_all_weights, random_all_weights;
    
    std::string data_filename = CLI::GetParam<std::string>("data");
    
    arma::mat data_in;
    data_in.load(data_filename, arma::raw_ascii);
    
    // THIS IS BAD: do it better
    if (data_in.n_rows > data_in.n_cols) {
      data_all_mat = arma::trans(data_in);
    }
    else {
      data_all_mat = data_in;
    }
    data_in.reset();
    
    data_all_weights.set_size(data_all_mat.n_cols);
    data_all_weights.fill(1.0);
  
    helper.Init(data_all_mat);
    
    //mlpack::Log::Info << "Processing randoms on root process.\n";
    
    if (CLI::HasParam("random")) {
      
      std::string random_filename = CLI::GetParam<std::string>("random");
      
      arma::mat random_in;
      random_in.load(random_filename, arma::raw_ascii);
      
      // THIS IS BAD: do it better
      if (random_in.n_rows > random_in.n_cols) {
        random_all_mat = arma::trans(random_in);
      }
      else {
        random_all_mat = random_in;
      }
      random_in.reset();
      
      random_all_weights.set_size(random_all_mat.n_cols);
      random_all_weights.fill(1.0);
    
    } // is the random set input as a file?
    else {
      
      // we need to generate our own random set
      int num_random = CLI::GetParam<int>("num_random");
      
      random_all_mat.set_size(3, num_random);
      
      if (num_random > 0) {
        GenerateRandomProblem generator(0.1, 0.5, 0.1, 0.5,
                                        num_random, num_random+1,
                                        helper);
        generator.GenerateRandomSet(random_all_mat);
        
        random_all_weights.set_size(num_random);
        random_all_weights.fill(1.0);
        
      }
      
    }
    
    std::vector<arma::mat> proc_data(num_procs);
    std::vector<arma::mat> proc_randoms(num_procs);
    
    std::vector<arma::colvec> proc_data_weights(num_procs);
    std::vector<arma::colvec> proc_random_weights(num_procs);
    
    int num_data = data_all_mat.n_cols;
    int num_randoms = random_all_mat.n_cols;
    
    int data_counter = 0;
    int random_counter = 0;
    
    //mlpack::Log::Info << "Splitting data among processes.\n";
    
    // split the data
    for (int i = 0; i < num_procs; i++)
    {
      
      // make sure the last process gets whatever's left over
      int num_data_here = i < num_procs - 1 ? num_data / num_procs : num_data - data_counter;
      mlpack::Log::Info << "Num data on " << i << ": " << num_data_here << "\n";
      
      // Need the -1 because span is inclusive unlike everything else in c++
      proc_data[i] = data_all_mat(arma::span::all,
                                  arma::span(data_counter,
                                             data_counter + num_data_here - 1));
      
      proc_data_weights[i] = data_all_weights(arma::span(data_counter,
                                                         data_counter + num_data_here - 1), 0);
      
      data_counter += num_data_here;
      
      //mlpack::Log::Info << "Processing randoms.\n";
      
      if (num_randoms > 0) {
        
        int num_randoms_here = i < num_procs - 1 ? num_randoms / num_procs : num_randoms - random_counter;
        mlpack::Log::Info << "Num randoms on " << i << ": " << num_randoms_here << "\n";
        
        proc_randoms[i] = random_all_mat(arma::span::all,
                                         arma::span(random_counter,
                                                    random_counter + num_randoms_here - 1));
        
        proc_random_weights[i] = random_all_weights(arma::span(random_counter,
                                                               random_counter + num_randoms_here - 1), 0);
        
        random_counter += num_randoms_here;
        
      }
      
    } // loop over processes
    
    //mlpack::Log::Info << "Root process sending data and randoms.\n";
    
    boost::mpi::scatter(world, proc_data, data_mat, 0);
    boost::mpi::scatter(world, proc_data_weights, data_weights, 0);
    
    boost::mpi::scatter(world, proc_randoms, random_mat, 0);
    boost::mpi::scatter(world, proc_random_weights, random_weights, 0);

    boost::mpi::broadcast(world, helper, 0);

  } // root process sending data
  else
  {
    
    //mlpack::Log::Info << "Process " << world.rank() << " receiving data.\n";
    
    // other processes need to receive the data
    boost::mpi::scatter(world, data_mat, 0);
    boost::mpi::scatter(world, data_weights, 0);
    
    boost::mpi::scatter(world, random_mat, 0);
    boost::mpi::scatter(world, random_weights, 0);
    
    boost::mpi::broadcast(world, helper, 0);
    
  } // other proceses receiving data
  
  // For now, I'm just splitting processes along one dimension for simplicity
  
  // Only doing one type for now, need to have flags for this later
  // See how this is done in MLPACK
  
  world.barrier();
  
  ////////////// Doing algorithms ///////////////////////////////
  std::string resampler_str = CLI::GetParam<std::string>("resampler");
  std::string kernel_str = CLI::GetParam<std::string>("kernel");
  
  // doing naive resampling
/*
  if (0 == resampler_str.compare("naive")) {
    
    // doing general matcher
    if (0 == kernel_str.compare("general")) {
      
      Log::Info << "Doing Naive Resampling, General Single Matcher.\n\n";
      
      Timer::Start("compute_time");
      
      DistributedNptDriver<NaiveResamplingDriver<SingleMatcher, GenericNptAlg<SingleMatcher>,
      NptNode, SingleResults>, SingleResults>
      driver(world,
             data_mat,
             data_weights,
             random_mat,
             random_weights,
             matcher_args,
             num_x_procs,
             num_y_procs,
             num_z_procs,
             num_x_regions,
             num_y_regions,
             num_z_regions,
             box_x_length,
             box_y_length,
             box_z_length,
             tuple_size,
             leaf_size);
      
      driver.Compute();
      
      world.barrier();
      mlpack::Timer::Stop("compute_time");
      if (world.rank() == 0) {
        
        SingleResults result = driver.results();
        result.PrintResults();
        
      }
      
    } // end of general matcher
      // doing efficient kernel matcher (CPU)
    else if (0 == kernel_str.compare("efficient")) {
      
      Log::Info << "Doing Naive Resampling, Efficient CPU Matcher.\n\n";
      
      Timer::Start("compute_time");
      
      matcher_args.set_template_type(MatcherArguments::TEMPLATE_EFFICIENT_MATCHER);
      
      DistributedNptDriver<NaiveResamplingDriver<Efficient4ptMatcher, GenericNptAlg<Efficient4ptMatcher>,
      NptNode, SingleResults>, SingleResults>
      driver(world,
             data_mat,
             data_weights,
             random_mat,
             random_weights,
             matcher_args,
             num_x_procs,
             num_y_procs,
             num_z_procs,
             num_x_regions,
             num_y_regions,
             num_z_regions,
             box_x_length,
             box_y_length,
             box_z_length,
             tuple_size,
             leaf_size);
      
      driver.Compute();
      
      world.barrier();
      mlpack::Timer::Stop("compute_time");
      if (world.rank() == 0) {
        
        SingleResults result = driver.results();
        result.PrintResults();
        
      }
      
    } // end of efficient matcher
    else {
      Log::Fatal << "Option " << kernel_str << " not a valid kernel type.\n";
      return 1;
    }
    
  } // end of naive resampling
  else 
 */
  if (0 == resampler_str.compare("efficient")) {
    // doing efficient resampling
    
    // doing general matcher
    if (0 == kernel_str.compare("general")) {
      
      Log::Info << "Doing Efficient Resampling, General Single Matcher.\n\n";
      
      Timer::Start("compute_time");
      
      matcher_args.set_template_type(MatcherArguments::TEMPLATE_SINGLE_MATCHER);
      
      DistributedNptDriver<EfficientResamplingDriver<SingleMatcher, GenericNptAlg<SingleMatcher>,
      NptNode, SingleResults>, SingleResults>
      driver(world,
             data_mat,
             data_weights,
             random_mat,
             random_weights,
             matcher_args,
             helper,
             num_x_procs,
             num_y_procs,
             num_z_procs,
             num_x_regions,
             num_y_regions,
             num_z_regions,
             tuple_size,
             leaf_size);
      
      driver.Compute();
      
      world.barrier();
      mlpack::Timer::Stop("compute_time");
      if (world.rank() == 0) {
        
        SingleResults result = driver.results();
        result.PrintResults();
        
      }
      
    } // end of general matcher
      // doing efficient kernel matcher (CPU)
    else if (0 == kernel_str.compare("efficient")) {
      
      Log::Info << "Doing Efficient Resampling, Multi-Tree Traversal, Efficient CPU Matcher.\n\n";
      
      Timer::Start("compute_time");
      
      matcher_args.set_template_type(MatcherArguments::TEMPLATE_EFFICIENT_MATCHER);
      
      DistributedNptDriver<EfficientResamplingDriver<Efficient4ptMatcher, GenericNptAlg<Efficient4ptMatcher>,
      NptNode, SingleResults>, SingleResults>
      driver(world,
             data_mat,
             data_weights,
             random_mat,
             random_weights,
             matcher_args,
             helper,
             num_x_procs,
             num_y_procs,
             num_z_procs,
             num_x_regions,
             num_y_regions,
             num_z_regions,
             tuple_size,
             leaf_size);
      
      driver.Compute();
      
      world.barrier();
      mlpack::Timer::Stop("compute_time");
      if (world.rank() == 0) {
        
        SingleResults result = driver.results();
        result.PrintResults();
        
      }
      
    } // end of efficient matcher
    else {
      Log::Fatal << "Option " << kernel_str << " not a valid kernel type.\n";
      return 1;
    }
    
  } // end of efficient resampling
  else {
    Log::Fatal << "Option " << resampler_str << " not a valid resampling type.\n";
    return 1;
  }
  
  return 0;
  
}


