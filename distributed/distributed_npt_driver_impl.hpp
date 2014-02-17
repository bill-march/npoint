/**
 * distributed_npt_driver_impl.hpp
 * @author Bill March (march@gatech.edu)
 *
 * Definitions for distributed driver.
 */

template<class TResamplingDriver, class TResults>
npoint_mlpack::DistributedNptDriver<TResamplingDriver, TResults>::
DistributedNptDriver(communicator& comm,
                     arma::mat& data_in,
                     arma::colvec& weights_in,
                     arma::mat& randoms_in,
                     arma::colvec& rweights_in,
                     MatcherArguments& matcher_args,
                     ResamplingHelper& resampling_helper,
                     int num_x_processes,
                     int num_y_processes,
                     int num_z_processes,
                     int num_x_regions,
                     int num_y_regions,
                     int num_z_regions,
                     int tuple_size,
                     int leaf_size)
:
comm_(comm),
// These should all start empty
data_all_mat_(),
data_all_weights_(),
random_mat_(),
random_weights_(),
//point_splitter_(),
//ghost_splitter_(),
num_resampling_regions_(num_x_regions * num_y_regions * num_z_regions),
do_random_(randoms_in.n_cols > 0),
tuple_size_(tuple_size),
leaf_size_(leaf_size),
num_x_regions_(num_x_regions),
num_y_regions_(num_y_regions),
num_z_regions_(num_z_regions),
x_step_(resampling_helper.x_size() / (double)num_x_regions_),
y_step_(resampling_helper.y_size() / (double)num_y_regions_),
z_step_(resampling_helper.z_size() / (double)num_z_regions_),
num_x_processes_(num_x_processes),
num_y_processes_(num_y_processes),
num_z_processes_(num_z_processes),
x_proc_step_(resampling_helper.x_size() / (double)num_x_processes_),
y_proc_step_(resampling_helper.y_size() / (double)num_y_processes_),
z_proc_step_(resampling_helper.z_size() / (double)num_z_processes_),
matcher_args_(matcher_args),
// note: this doesn't need to be squared here
max_matcher_(matcher_args_.max_matcher()),
resampling_helper_(resampling_helper),
results_(matcher_args_, num_resampling_regions_)
{
  
  // Go through data and randoms, figure out what will need which
  
  // There are two paritionings of the input space here:
  // 1) Split into resampling regions
  // 2) Parition among different processes
  
  // I also need to distinguish between points I actually own and ghosted points
  
  ///////////////////////
  
  // We identify what processes should own what points, and shuffle accordingly
  FindPointsOwners_(data_in, weights_in, randoms_in, rweights_in);
  
  //mlpack::Log::Info << "Found Points Owners.\n";
  num_points_ = data_all_mat_.n_cols;
  num_randoms_ = random_mat_.n_cols;
  
  
  // Get the ghost points we need
  // or, give out ghost points we know other processes need
  
  FindGhostPointsOwners_(data_all_mat_, data_all_weights_,
                         random_mat_, random_weights_);
  
  num_ghost_points_ = ghost_data_all_mat_.n_cols;
  num_ghost_randoms_ = ghost_random_mat_.n_cols;
  
  mlpack::Log::Info << "Process " << comm_.rank() << " has ";
  mlpack::Log::Info << num_points_ << " points, " << num_randoms_ << " randoms ";
  mlpack::Log::Info << num_ghost_points_ << " ghost points and " << num_ghost_randoms_;
  mlpack::Log::Info << " ghost randoms.\n";
  
  // Partition points and ghost points among resampling regions
  
  point_splitter_ = new ResamplingSplitter(data_all_mat_,
                                           data_all_weights_,
                                           num_x_regions_,
                                           num_y_regions_,
                                           num_z_regions_,
                                           resampling_helper_);
  
  data_mats_ = point_splitter_->data_mats();
  data_weights_ = point_splitter_->data_weights();
  
  ghost_splitter_ = new ResamplingSplitter(ghost_data_all_mat_,
                                           ghost_data_all_weights_,
                                           num_x_regions_,
                                           num_y_regions_,
                                           num_z_regions_,
                                           resampling_helper_);
  
  ghost_data_mats_ = ghost_splitter_->data_mats();
  ghost_data_weights_ = ghost_splitter_->data_weights();
  
} // constructor

template<class TResamplingDriver, class TResults>
npoint_mlpack::DistributedNptDriver<TResamplingDriver, TResults>::
~DistributedNptDriver()
{
  
  delete point_splitter_;
  delete ghost_splitter_;
  
}


template<class TResamplingDriver, class TResults>
int npoint_mlpack::DistributedNptDriver<TResamplingDriver, TResults>::
FindProcess_(arma::colvec& col)
{
  
  int x_ind = floor((col(0) - resampling_helper_.x_min()) / (x_proc_step_ + resampling_helper_.epsilon()));
  int y_ind = floor((col(1) - resampling_helper_.y_min()) / (y_proc_step_ + resampling_helper_.epsilon()));
  int z_ind = floor((col(2) - resampling_helper_.z_min()) / (z_proc_step_ + resampling_helper_.epsilon()));
  
  int proc_ind = (x_ind + num_x_processes_ * y_ind
                  + num_x_processes_ * num_y_processes_ * z_ind);
  
  /*
  std::cout << col << x_ind << ", " << y_ind << ", " << z_ind << "\n";
  std::cout << resampling_helper_.x_min() << ", ";
  std::cout << resampling_helper_.y_min() << ", ";
  std::cout << resampling_helper_.z_min() << "\n";
  std::cout << resampling_helper_.x_max() << ", ";
  std::cout << resampling_helper_.y_max() << ", ";
  std::cout << resampling_helper_.z_max() << "\n";
  std::cout << x_proc_step_ << ", ";
  std::cout << y_proc_step_ << ", ";
  std::cout << z_proc_step_ << "\n\n";
  */
  
  // Don't do this here, the ghost point logic assumes we'll find some bad
  // indices and takes care of it later
  //assert(proc_ind >= 0);
  //assert(proc_ind < comm_.size());
  
  return proc_ind;
  
}

template<class TResamplingDriver, class TResults>
void npoint_mlpack::DistributedNptDriver<TResamplingDriver, TResults>::
FindPointsOwners_(arma::mat& data_in,
                  arma::colvec& /*weights_in*/,
                  arma::mat& randoms_in,
                  arma::colvec& /*rweights_in*/)
{
  
  
  std::vector<arma::mat> process_data_mats(comm_.size());
  std::vector<arma::colvec> process_data_weights(comm_.size());
  
  std::vector<arma::mat> process_random_mats(comm_.size());
  std::vector<arma::colvec> process_random_weights(comm_.size());
  
  // The number of my points that each process will get
  std::vector<size_t> process_num_points(comm_.size(), 0);
  std::vector<size_t> process_num_random(comm_.size(), 0);
  
  // The indices of my points that each process will get
  std::vector<std::vector<size_t> > points_in_process_id(comm_.size());
  std::vector<std::vector<size_t> > randoms_in_process_id(comm_.size());
  
  // loop over the points, determine which process each belongs to
  //#pragma omp parallel for num_threads(1)
  for (size_t i = 0; i < data_in.n_cols; i++)
  {
    
    arma::colvec col_i;
    col_i = data_in.col(i);
    
    int process_id = FindProcess_(col_i);
    
    //#pragma omp critical
    {
    process_num_points[process_id]++;
    points_in_process_id[process_id].push_back(i);
    }
    
  } // loop over data
  
  // now, do the same for the randoms
  //#pragma omp parallel for num_threads(1)
  for (size_t i = 0; i < randoms_in.n_cols; i++)
  {
    
    arma::colvec col_i;
    col_i = randoms_in.col(i);
    
    int process_id = FindProcess_(col_i);
    
    //#pragma omp critical
    {
    process_num_random[process_id]++;
    randoms_in_process_id[process_id].push_back(i);
    }
    
  } // loop over randoms
  
  // make arrays of points to send to each process
  
  //#pragma omp parallel for num_threads(1)
  for (int i = 0; i < comm_.size(); i++)
  {
    
    //process_data_mats[i] = new arma::mat(3, process_num_points[i]);
    //process_data_weights[i] = new arma::colvec(process_num_points[i]);
    
    //process_random_mats[i] = new arma::mat(3, process_num_random[i]);
    //process_random_weights[i] = new arma::mat(process_num_random[i]);
    
    //mlpack::Log::Info << "Process " << i << " num points: " << process_num_points[i];
    //mlpack::Log::Info << ", num randoms: " << process_num_random[i] << "\n";
    
    process_data_mats[i].resize(3, process_num_points[i]);
    process_data_weights[i].resize(process_num_points[i]);
    
    process_random_mats[i].resize(3, process_num_random[i]);
    process_random_weights[i].resize(process_num_random[i]);
    
    // now, go through and copy the points over
    
    for (size_t point_ind = 0; point_ind < process_num_points[i]; point_ind++)
    {
      
      process_data_mats[i].col(point_ind) = data_in.col(points_in_process_id[i][point_ind]);
      
    }
    
    for (size_t rand_ind = 0; rand_ind < process_num_random[i]; rand_ind++) {
      
      process_random_mats[i].col(rand_ind) = randoms_in.col(randoms_in_process_id[i][rand_ind]);
      
    }
    
  } // loop over processes
  
  // IMPORTANT: I've found which points go where, but the arrays are currently
  // local
  
  // now, send and receive points
  
  ShufflePoints_(process_num_points, process_data_mats, process_data_weights,
                 data_all_mat_, data_all_weights_);
  
  ShufflePoints_(process_num_random, process_random_mats,
                 process_random_weights, random_mat_, random_weights_);
  
} // FindPointsOwners

template<class TResamplingDriver, class TResults>
void npoint_mlpack::DistributedNptDriver<TResamplingDriver, TResults>::
ShufflePoints_(std::vector<size_t>& points_per_process,
               std::vector<arma::mat>& points_to_send,
               std::vector<arma::colvec>& weights_to_send,
               arma::mat& destination_mat,
               arma::colvec& destination_weights)
{
  
  std::vector<size_t> num_points_to_receive;
  boost::mpi::all_to_all(comm_, points_per_process, num_points_to_receive);
  
  // points_to_receive[i] is now the number of points I'll be getting from
  // process i
  
  // this is the number of points we'll own
  //size_t total_points_owned = std::accumulate(num_points_to_receive.begin(),
  //                                            num_points_to_receive.end(),
  //                                            0);
  
  std::vector<arma::mat> points_to_receive;
  
  boost::mpi::all_to_all(comm_, points_to_send, points_to_receive);
  
  std::vector<arma::colvec> weights_to_receive;
  
  boost::mpi::all_to_all(comm_, weights_to_send, weights_to_receive);
  
  // now, gather the matrices together
  destination_mat = points_to_receive[0];
  destination_weights = weights_to_receive[0];
  for (int i = 1; i < comm_.size(); i++)
  {
    
    destination_mat.insert_cols(destination_mat.n_cols, points_to_receive[i]);
    destination_weights.insert_rows(destination_weights.n_rows,
                                    weights_to_receive[i]);
    
  }
  
  // now, destination mat and weights should be correct
  
} // ShufflePoints()

// I may be able to reuse code from the normal points above
template<class TResamplingDriver, class TResults>
void npoint_mlpack::DistributedNptDriver<TResamplingDriver, TResults>::
FindGhostPointsOwners_(arma::mat& data_in,
                       arma::colvec& /*weights_in*/,
                       arma::mat& randoms_in,
                       arma::colvec& /*rweights_in*/)
{
  
  // Now, I have the points (and randoms) that I own
  // I need to figure out what ghost points other processes will need and send
  // them
  
  // Note: a point may be a ghost for several processors (i.e. at a corner)
  
  // IMPORTANT: in order to avoid overcounting, we only pass ghosts to processes
  // with a rank lower than ours.  This way, any tuple that crosses a boundary
  // between processes will only be counted by the process with the lowest rank
  // In other words, we make sure that no processes exchange ghosts.
  
  std::vector<arma::mat> process_data_mats(comm_.size());
  std::vector<arma::colvec> process_data_weights(comm_.size());
  
  std::vector<arma::mat> process_random_mats(comm_.size());
  std::vector<arma::colvec> process_random_weights(comm_.size());
  
  // The number of my points that each process will get
  std::vector<size_t> process_num_points(comm_.size(), 0);
  std::vector<size_t> process_num_random(comm_.size(), 0);
  
  // The indices of my points that each process will get
  std::vector<std::vector<size_t> > points_in_process_id(comm_.size());
  std::vector<std::vector<size_t> > randoms_in_process_id(comm_.size());
  
  // loop over the points, determine which process each belongs to
  //#pragma omp parallel for num_threads(1)
  for (size_t i = 0; i < num_points_; i++)
  {
    
    arma::colvec col_i;
    col_i = data_in.col(i);
    
    
    // This is one part thats different from above
    // Need to find all processes that the point is a ghost for
    std::set<int> processes;
    FindGhostProcess_(comm_.rank(), col_i, processes);
    
    for (std::set<int>::iterator process_id = processes.begin();
         process_id != processes.end(); process_id++)
    {
      
      //#pragma omp critical
      {
      process_num_points[*process_id]++;
      points_in_process_id[*process_id].push_back(i);
      }
      
    }
    
  } // loop over data
  
  
  // now, do the same for the randoms
  //#pragma omp parallel for num_threads(1)
  for (size_t i = 0; i < random_mat_.n_cols; i++)
  {
    
    arma::colvec col_i;
    col_i = randoms_in.col(i);
    
    std::set<int> processes;
    FindGhostProcess_(comm_.rank(), col_i, processes);
    
    for (std::set<int>::iterator process_id = processes.begin();
         process_id != processes.end(); process_id++)
    {
      //#pragma omp critical
      {
      process_num_random[*process_id]++;
      randoms_in_process_id[*process_id].push_back(i);
      }
    }
    
  } // loop over randoms
  
  // make arrays of points to send to each process
  //#pragma omp parallel for num_threads(1)
  for (int i = 0; i < comm_.size(); i++)
  {
    
    //process_data_mats[i] = new arma::mat(3, process_num_points[i]);
    //process_data_weights[i] = new arma::colvec(process_num_points[i]);
    
    //process_random_mats[i] = new arma::mat(3, process_num_random[i]);
    //process_random_weights[i] = new arma::mat(process_num_random[i]);
    
    process_data_mats[i].resize(3, process_num_points[i]);
    process_data_weights[i].resize(process_num_points[i]);
    
    process_random_mats[i].resize(3, process_num_random[i]);
    process_random_weights[i].resize(process_num_random[i]);
    
    // now, go through and copy the points over
    
    for (size_t point_ind = 0; point_ind < process_num_points[i]; point_ind++)
    {
      
      process_data_mats[i].col(point_ind) = data_in.col(points_in_process_id[i][point_ind]);
      
    }
    
    for (size_t rand_ind = 0; rand_ind < process_num_random[i]; rand_ind++) {
      
      process_random_mats[i].col(rand_ind) = randoms_in.col(randoms_in_process_id[i][rand_ind]);
      
    }
    
  } // loop over processes
  
  
  // IMPORTANT: I've found which points go where, but the arrays are currently
  // local
  
  // now, send and receive points
  
  ShufflePoints_(process_num_points, process_data_mats, process_data_weights,
                 ghost_data_all_mat_, ghost_data_all_weights_);
  
  ShufflePoints_(process_num_random, process_random_mats,
                 process_random_weights, ghost_random_mat_,
                 ghost_random_weights_);
  
  
} // FindGhostPointsOwners_


template<class TResamplingDriver, class TResults>
void npoint_mlpack::DistributedNptDriver<TResamplingDriver, TResults>::
FindGhostProcess_(int my_rank,
                  arma::colvec& col,
                  std::set<int>& processes)
{
  
  // need to find all processes that the point may be a ghost for and put
  // them in the argument vector
  
  // we already know which process owns the point - the one calling this
  // function
  
  // ghost points will be within distance (max matcher) of the boundary
  // note that the minimum distance doesn't matter, because a point right on
  // the boundary may still be far from an interior point I own
  
  // Idea 1: shift the grid, then call the ordinary find process with the
  // shifted grid - will need to shift it in each combination of directions
  // Don't shift the grid, shift the point
  
  arma::colvec shift_vec(3);
  shift_vec(0) = 0.0;
  shift_vec(1) = 0.0;
  shift_vec(2) = 0.0;
  
  // there will be 27 - 1 combinations
  // three choices for each entry (-1, 0, +1) minus the 0,0,0 case
  
  for (int i = 0; i <= 27; i++) {
    
    // the 0,0,0 case corresponds to (1,1,1) in base 3, which is 13
    if (i == 13)
      continue;
    
    // x val goes 0, 1, 2 (which corresponds to 0, -1, +1)
    int x_val = i % 3;
    // this will give me -1, 0, +1
    x_val -= 1;
    
    int y_val = i / 3;
    y_val = y_val % 3;
    y_val -= 1;
    
    int z_val = i / 9;
    z_val = z_val % 3;
    z_val -= 1;
    
    //mlpack::Log::Info << "x_val: " << x_val << ", y_val: " << y_val;
    //mlpack::Log::Info << ", z_val: " << z_val << "\n";
    
    shift_vec(0) = (double)x_val;
    shift_vec(1) = (double)y_val;
    shift_vec(2) = (double)z_val;
    
    // this will include a few too many ghost points, but avoids computing
    // distances
    //std::cout << "max_matcher: " << max_matcher_ << "\n";
    shift_vec *= max_matcher_;
    
    // need to do this because of armadillo weirdness
    arma::colvec this_vec = col + shift_vec;
    //std::cout << "Doing ghost\n";
    int proc_ind = FindProcess_(this_vec);
    //std::cout << "proc_ind " << proc_ind << "\n\n";
    
    // make sure we got a legit process that isn't our own and that has an
    // index smaller than ours
    if (proc_ind != comm_.rank() && proc_ind >= 0 && proc_ind < comm_.size()
        && proc_ind < my_rank) {
      // problem: this may insert duplicates
      // solutions: use set, have a vector of bools of length comm_.size()
      // and just set things to true
      processes.insert(proc_ind);
    }
    
  } // loop over all shifts
  
} // FindGhostProcess_()


template<class TResamplingDriver, class TResults>
void npoint_mlpack::DistributedNptDriver<TResamplingDriver, TResults>::
GatherResults_(TResults& result)
{
  
  //mlpack::Log::Info << "Calling reduce.\n";
  reduce(comm_, result, results_, std::plus<TResults>(), 0);

  /*
  if (comm_.rank() == 0)
  {
    
    results_.PrintResults();
    
    // now, write it to a file or something, if needed
    
  }
   */
  
} // GatherResults_



template<class TResamplingDriver, class TResults>
void npoint_mlpack::DistributedNptDriver<TResamplingDriver, TResults>::Compute()
{
  
  //mlpack::Log::Info << "Building resampling on proc " << comm_.rank() << "\n";
  
  // do the computation on my points and ghosts
  TResamplingDriver resampler(data_mats_, data_weights_,
                              ghost_data_mats_, ghost_data_weights_,
                              random_mat_, random_weights_,
                              ghost_random_mat_, ghost_random_weights_,
                              matcher_args_,
                              tuple_size_, leaf_size_);
  
  //mlpack::Log::Info << "Computing on proc " << comm_.rank() << "\n";
  
  resampler.Compute();
  
  TResults my_result = resampler.results();
  
  //mlpack::Log::Info << "Partial result on proc " << comm_.rank() << "\n";
  //my_result.PrintResults();
  
  // Wait for all the computations to finish before gathering results
  comm_.barrier();
  
  //mlpack::Log::Info << "past barrier\n";
  
  //my_result.PrintResults();
  
  GatherResults_(my_result);
  
  
} // Compute

template<class TResamplingDriver, class TResults>
TResults& npoint_mlpack::DistributedNptDriver<TResamplingDriver, TResults>::
results()
{
  return results_;
}




