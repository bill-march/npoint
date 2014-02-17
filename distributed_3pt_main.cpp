/**
 * distributed_3_main.cpp
 * @author Bill March (march@gatech.edu)
 *
 * Does distributed 3-point computation for a single matcher.
 */


#include "distributed/distributed_npt_driver.hpp"
#include "infrastructure/generate_random_problem.hpp"
#include "resampling_classes/efficient_resampling_driver.hpp"
#include "resampling_classes/naive_resampling_driver.hpp"
#include "infrastructure/generic_npt_alg.hpp"
#include "infrastructure/pairwise_npt_traversal.hpp"
#include "matchers/single_matcher.hpp"
#include "matchers/efficient_cpu_matcher.hpp"
#include "matchers/3_point_single_matcher.hpp"
#include "results/single_results.hpp"
#include "infrastructure/resampling_helper.hpp"

PROGRAM_INFO("Distributed 3-point correlation estimation.", "Does 3pcf correlation counts for a single matcher on many nodes with optional jackknife resampling.");
PARAM_STRING_REQ("data", "Point coordinates.", "d");
PARAM_STRING("random", "Optional Poisson set coordinates.", "r", "fake");
PARAM_INT("num_random", "If random isn't specified, this will generate the given number of points", "R", 0);
PARAM_STRING_REQ("matcher_lower_bounds", "The lower bound distances for the matcher, stored in a symmetric 3x3 matrix..", "l")
PARAM_STRING_REQ("matcher_upper_bounds", "The upper bound distances for the matcher, stored in a symmetric 3x3 matrix..", "u")
PARAM_INT("leaf_size", "Maximum number of points in a leaf node.", "i", 16);
PARAM_INT("num_x_regions", "Number of jackknife regions to divide the input into along the x coordinate.",
          "x", 1);
PARAM_INT("num_y_regions", "Number of jackknife regions to divide the input into along the y coordinate.",
          "y", 1);
PARAM_INT("num_z_regions", "Number of jackknife regions to divide the input into along the z coordinate.",
          "z", 1);
PARAM_INT("num_x_procs", "Number of processes along the x dimension.", "e", 1);
PARAM_INT("num_y_procs", "Number of processes along the y dimension.", "f", 1);
PARAM_INT("num_z_procs", "Number of processes along the z dimension.", "g", 1);

PARAM_STRING("resampler", "Select the type of resampling to use.  Only does \"efficient\" for now.", "s", "efficient");
PARAM_STRING("traversal", "Select the type of tree traversal to use.  Options are \"multi\" and \"pairwise\".", "t", "pairwise");
PARAM_STRING("kernel", "Select the type of base case computation.  Options are \"general\" (the old version for general n), \"simple\" for a specialized 3-point case, and \"efficient\" for the hardware optimized version.", "k", "efficient");


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
  
  if (tuple_size != 3) {
    
    mlpack::Log::Fatal << "Matchers need to be 3 x 3 matrices!\n";
    return 1;
    
  }
  
  if (tuple_size != matcher_upper_bounds.n_cols ||
      tuple_size != matcher_lower_bounds.n_rows) {
    
    mlpack::Log::Fatal << "Upper and lower matchers need to both be 3 x 3!\n";
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
    
    std::cout << "x_dims: " << helper.x_min() << ", " << helper.x_max() << "\n";
    std::cout << "y_dims: " << helper.y_min() << ", " << helper.y_max() << "\n";
    std::cout << "z_dims: " << helper.z_min() << ", " << helper.z_max() << "\n";
    
    
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
      
    } // filling in randoms
    
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
  std::string traversal_str = CLI::GetParam<std::string>("traversal");
  std::string kernel_str = CLI::GetParam<std::string>("kernel");
  
  // doing naive resampling
  /*
  if (0 == resampler_str.compare("naive")) {
    
    // doing multi-traversal
    if (0 == traversal_str.compare("multi")) {
      
      // doing general matcher
      if (0 == kernel_str.compare("general")) {
        
        Log::Info << "Doing Naive Resampling, Multi-Tree Traversal, General Single Matcher.\n\n";
        
        Timer::Start("compute_time");
        
        matcher_args.set_template_type(MatcherArguments::TEMPLATE_SINGLE_MATCHER);
        
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
        
        Log::Info << "Doing Naive Resampling, Multi-Tree Traversal, Efficient CPU Matcher.\n\n";
        
        Timer::Start("compute_time");
        
        matcher_args.set_template_type(MatcherArguments::TEMPLATE_EFFICIENT_MATCHER);
        
        DistributedNptDriver<NaiveResamplingDriver<EfficientCpuMatcher, GenericNptAlg<EfficientCpuMatcher>,
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
      else if (0 == kernel_str.compare("simple")) {
        
        Log::Info << "Doing Naive Resampling, Multi-Tree Traversal, Simple 3-Point Matcher.\n\n";
        
        Timer::Start("compute_time");
        
        matcher_args.set_template_type(MatcherArguments::TEMPLATE_SINGLE_MATCHER);
        
        DistributedNptDriver<NaiveResamplingDriver<ThreePointSingleMatcher, GenericNptAlg<ThreePointSingleMatcher>,
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
        
      }
      else {
        Log::Fatal << "Option " << kernel_str << " not a valid kernel type.\n";
        return 1;
      }
      
    }// end of multi-traversal
    else if (0 == traversal_str.compare("pairwise")) {
      // doing pairwise traversal
      
      // doing general matcher
      if (0 == kernel_str.compare("general")) {
        
        Log::Info << "Doing Naive Resampling, Pairwise Traversal, General Single Matcher.\n\n";
        
        Timer::Start("compute_time");
        
        matcher_args.set_template_type(MatcherArguments::TEMPLATE_SINGLE_MATCHER);
        
        DistributedNptDriver<NaiveResamplingDriver<SingleMatcher, PairwiseNptTraversal<SingleMatcher>,
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
        
        Log::Info << "Doing Naive Resampling, Pairwise Traversal, Efficient CPU Matcher.\n\n";
        
        Timer::Start("compute_time");
        
        matcher_args.set_template_type(MatcherArguments::TEMPLATE_EFFICIENT_MATCHER);
        
        DistributedNptDriver<NaiveResamplingDriver<EfficientCpuMatcher, PairwiseNptTraversal<EfficientCpuMatcher>,
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
      
      else if (0 == kernel_str.compare("simple")) {
        
        Log::Info << "Doing Naive Resampling, Multi-Tree Traversal, Simple 3-Point Matcher.\n\n";
        
        Timer::Start("compute_time");
        
        matcher_args.set_template_type(MatcherArguments::TEMPLATE_SINGLE_MATCHER);
        
        DistributedNptDriver<NaiveResamplingDriver<ThreePointSingleMatcher, PairwiseNptTraversal<ThreePointSingleMatcher>,
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
        
      } // end of simple 3-point matcher
      
      else {
        Log::Fatal << "Option " << kernel_str << " not a valid kernel type.\n";
        return 1;
      }
      
    } // end of pairwise traversal
    else {
      Log::Fatal << "Option " << traversal_str << " not a valid traversal type.\n";
      return 1;
    }
    
  } // end of naive resampling
  else 
   */
  if (0 == resampler_str.compare("efficient")) {
    // doing efficient resampling
    
    // doing multi-traversal
    if (0 == traversal_str.compare("multi")) {
      
      // doing general matcher
      if (0 == kernel_str.compare("general")) {
        
        Log::Info << "Doing Efficient Resampling, Multi-Tree Traversal, General Single Matcher.\n\n";
        
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
        
        DistributedNptDriver<EfficientResamplingDriver<EfficientCpuMatcher, GenericNptAlg<EfficientCpuMatcher>,
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
      else if (0 == kernel_str.compare("simple")) {
        
        Log::Info << "Doing Naive Resampling, Multi-Tree Traversal, Simple 3-Point Matcher.\n\n";
        
        Timer::Start("compute_time");
        
        matcher_args.set_template_type(MatcherArguments::TEMPLATE_SINGLE_MATCHER);
        
        DistributedNptDriver<NaiveResamplingDriver<ThreePointSingleMatcher, GenericNptAlg<ThreePointSingleMatcher>,
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
        
      }
      
      else {
        Log::Fatal << "Option " << kernel_str << " not a valid kernel type.\n";
        return 1;
      }
      
    }// end of multi-traversal
    else if (0 == traversal_str.compare("pairwise")) {
      // doing pairwise traversal
      
      // doing general matcher
      if (0 == kernel_str.compare("general")) {
        
        Log::Info << "Doing Efficient Resampling, Pairwise Traversal, General Single Matcher.\n\n";
        
        Timer::Start("compute_time");
        
        matcher_args.set_template_type(MatcherArguments::TEMPLATE_SINGLE_MATCHER);
        
        DistributedNptDriver<EfficientResamplingDriver<SingleMatcher, PairwiseNptTraversal<SingleMatcher>,
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
        
        Log::Info << "Doing Efficient Resampling, Pairwise Traversal, Efficient CPU Matcher.\n\n";
        
        Timer::Start("compute_time");
        
        matcher_args.set_template_type(MatcherArguments::TEMPLATE_EFFICIENT_MATCHER);
        
        DistributedNptDriver<EfficientResamplingDriver<EfficientCpuMatcher, PairwiseNptTraversal<EfficientCpuMatcher>,
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
      else if (0 == kernel_str.compare("simple")) {
        
        Log::Info << "Doing Efficient Resampling, Multi-Tree Traversal, Simple 3-Point Matcher.\n\n";
        
        Timer::Start("compute_time");
        
        matcher_args.set_template_type(MatcherArguments::TEMPLATE_SINGLE_MATCHER);
        
        DistributedNptDriver<EfficientResamplingDriver<ThreePointSingleMatcher,
        PairwiseNptTraversal<ThreePointSingleMatcher>,
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
        
      }
      
      else {
        Log::Fatal << "Option " << kernel_str << " not a valid kernel type.\n";
        return 1;
      }
      
    } // end of pairwise traversal
    else {
      Log::Fatal << "Option " << traversal_str << " not a valid traversal type.\n";
      return 1;
    }
    
  } // end of efficient resampling
  else {
    Log::Fatal << "Option " << resampler_str << " not a valid resampling type.\n";
    return 1;
  }

  //mlpack::Log::Info << "Process " << world.rank() << " finished.\n";
  
  return 0;
  
} // main
