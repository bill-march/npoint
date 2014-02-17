//
//  distributed_npoint_test.cpp
//  contrib_march
//
//  Created by William March on 9/24/12.
//
//

#include "../distributed/distributed_npt_driver.hpp"
#include "../resampling_classes/efficient_resampling_driver.hpp"
#include "../infrastructure/generic_npt_alg.hpp"
#include "../matchers/efficient_cpu_matcher.hpp"
#include "../results/single_results.hpp"
#include "../infrastructure/resampling_helper.hpp"

///////////////// IMPORTANT: need to manually make sure the number of
// openmpi processes matches the product of num_x_procs, etc.

// Always run this with 8 processes

using namespace mlpack;
using namespace npoint_mlpack;

int main(int argc, char* argv[])
{
  
  CLI::ParseCommandLine(argc, argv);
  
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
  
  
  omp_set_nested(1);
  omp_set_num_threads(1);
  
  int num_procs = world.size();
  
  int leaf_size = 5;
  
  arma::mat matcher_lower_bounds(3,3);
  arma::mat matcher_upper_bounds(3,3);
  
  matcher_lower_bounds << 0.0 << 0.09 << 0.09 << arma::endr
                       << 0.09 << 0.0 << 0.09 << arma::endr
                       << 0.09 << 0.09 << 0.0 << arma::endr;
  
  matcher_upper_bounds << 0.0 << 0.11 << 0.11 << arma::endr
                       << 0.11 << 0.0 << 0.11 << arma::endr
                       << 0.11 << 0.11 << 0.0 << arma::endr;
  
  
  MatcherArguments matcher_args(matcher_lower_bounds,
                                matcher_upper_bounds);
  
  unsigned int tuple_size = 3;
  
  int num_x_regions = 3;
  int num_y_regions = 3;
  int num_z_regions = 3;
  
  
  ////////////// Doing algorithms ///////////////////////////////
  
  // The total data and random inputs.  These will be split up and scattered
  // to the other processes by the root.
  arma::mat data_all_mat, random_all_mat;
  arma::colvec data_all_weights, random_all_weights;
  
  // These are the data that this process is responsible for passing to the
  // distributed driver.
  arma::mat data_mat, random_mat;
  arma::colvec data_weights, random_weights;
  
  SingleResults serial_result;
  
  ResamplingHelper helper;
  
  // root process reads in data, sends out chunks
  if (world.rank() == 0) {
    
    //mlpack::Log::Info << "Processing data on root process\n";
    
    std::string data_filename("test_data_1000.csv");
    
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
    
    std::string random_filename("test_random_1000.csv");
    
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
      
    std::vector<arma::mat> proc_data(num_procs);
    std::vector<arma::mat> proc_randoms(num_procs);
    
    std::vector<arma::colvec> proc_data_weights(num_procs);
    std::vector<arma::colvec> proc_random_weights(num_procs);
    
    int num_data = data_all_mat.n_cols;
    int num_randoms = random_all_mat.n_cols;
    
    int data_counter = 0;
    int random_counter = 0;
    
    // split the data
    for (int i = 0; i < num_procs; i++)
    {
      
      // make sure the last process gets whatever's left over
      int num_data_here = i < num_procs - 1 ? num_data / num_procs : num_data - data_counter;
      //mlpack::Log::Info << "Num data on " << i << ": " << num_data_here << "\n";
      
      // Need the -1 because span is inclusive unlike everything else in c++
      proc_data[i] = data_all_mat(arma::span::all,
                                  arma::span(data_counter,
                                             data_counter + num_data_here - 1));
      
      proc_data_weights[i] = data_all_weights(arma::span(data_counter,
                                                         data_counter + num_data_here - 1), 0);
      
      data_counter += num_data_here;
      
      if (num_randoms > 0) {
        
        int num_randoms_here = i < num_procs - 1 ? num_randoms / num_procs : num_randoms - random_counter;
        //mlpack::Log::Info << "Num randoms on " << i << ": " << num_randoms_here << "\n";
        
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
    
    // Now, run the single result
    
    EfficientResamplingDriver<EfficientCpuMatcher,
    GenericNptAlg<EfficientCpuMatcher>,
    NptNode, SingleResults> serial_alg(data_all_mat, data_all_weights,
                                       random_all_mat, random_all_weights,
                                       matcher_args,
                                       num_x_regions, num_y_regions,
                                       num_z_regions,
                                       helper,
                                       tuple_size,
                                       leaf_size);
    
    Log::Info << "Running serial algorithm.\n";

    serial_alg.Compute();
    
    serial_result = serial_alg.results();
    
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
  
  int num_x_procs = 2;
  int num_y_procs = 2;
  int num_z_procs = 2;
  
  Log::Info << "Process " << world.rank() << " starting computation.\n";
  DistributedNptDriver<EfficientResamplingDriver<EfficientCpuMatcher,
                          GenericNptAlg<EfficientCpuMatcher>,
                          NptNode, SingleResults>,
                       SingleResults>
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
  
  if (world.rank() == 0) {
  
    SingleResults parallel_result = driver.results();

    if (serial_result != parallel_result)
    {
      Log::Fatal << "Results don't match!\n";
      Log::Fatal << "Serial results: \n";
      serial_result.PrintResults();
      Log::Fatal << "\nParallel results: \n";
      parallel_result.PrintResults();
      return 1;
    }
    else {
      Log::Info << "All results match.\n";
    }
  
  } // root process checks the results
  
  return 0;
  
} // main
