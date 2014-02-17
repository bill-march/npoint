//
//  naive_resampling_driver_impl.hpp
//  contrib_march
//
//  Created by William March on 6/13/12.
//  Copyright (c) 2012 Georgia Institute of Technology. All rights reserved.
//

#ifndef __MLPACK_METHODS_NPOINT_RESAMPLING_CLASSES_NAIVE_RESAMPLING_DRIVER_IMPL_HPP
#define __MLPACK_METHODS_NPOINT_RESAMPLING_CLASSES_NAIVE_RESAMPLING_DRIVER_IMPL_HPP


// Need these to handle the different tree type contructor arguments
// Octrees need to know the size of the box that contains the data 
// they could just figure this out later
template <class TMatcher, class TTraversal, class TTree, class TResults>
void npoint_mlpack::NaiveResamplingDriver<TMatcher, TTraversal, TTree, TResults>::BuildTree_(TTree** tree, arma::mat& data)
{
  
  *tree = new TTree(data, leaf_size_);
  
}

// Passing in ghosts for now, but don't do them for naive
template <class TMatcher, class TTraversal, class TTree, class TResults>
npoint_mlpack::NaiveResamplingDriver<TMatcher, TTraversal, TTree, TResults>::
NaiveResamplingDriver(std::vector<arma::mat*>& data_mats_in,
                      std::vector<arma::colvec*>& data_weights_in,
                      std::vector<arma::mat* >& data_ghost_mats_in,
                      std::vector<arma::colvec*>& data_ghost_weights_in,
                      arma::mat& random, arma::colvec& rweights,
                      arma::mat& random_ghost, arma::colvec& rweights_ghost,
                      MatcherArguments& matcher_args,
                      int tuple_size,
                      int leaf_size)
:
num_resampling_regions_(data_mats_in.size()),
random_mat_(random),
random_weights_(rweights),
data_mats_(data_mats_in),
data_weights_(data_weights_in),
thread_data_mats_(omp_get_max_threads(), NULL),
thread_data_weights_(omp_get_max_threads(), NULL),
thread_data_trees_(omp_get_max_threads(), NULL),
thread_region_id_(omp_get_max_threads(), -1),
thread_random_mats_(omp_get_max_threads(), NULL),
thread_random_weights_(omp_get_max_threads(), NULL),
thread_random_trees_(omp_get_max_threads(), NULL),
previous_region_(-1),
previous_num_random_(-1),
matcher_args_(matcher_args),
owns_data_(false),
results_(matcher_args, num_resampling_regions_),
tuple_size_(tuple_size),
leaf_size_(leaf_size),
do_random_(random_mat_.n_cols > 0),
num_x_partitions_(-1),
num_y_partitions_(-1),
num_z_partitions_(-1),
resampling_helper_(),
x_step_(-1.0),
y_step_(-1.0),
z_step_(-1.0),
num_points_(-1),
total_computations_(total_num_computations()),
num_prunes_(0),
num_base_cases_(0),
num_pairs_considered_(0),
num_triples_considered_(0)
{

  
  if (do_random_) {
    //random_tree_ = new TTree(random_mat_, leaf_size_);
    BuildTree_(&random_tree_, random_mat_);
  }
  else {
    random_tree_ = NULL;
  }
  
  // I don't think we want to call this here
  //GetNextData_();
  
  // Isolating the random matrices per thread
  if (do_random_) {
    for (int i = 0; i < omp_get_max_threads(); i++) {
      
      thread_random_mats_[i] = new arma::mat(random_mat_);
      //*thread_random_mats_[i] = random_mat_;
      
      thread_random_weights_[i] = new arma::colvec(random_weights_);
      //*thread_random_weights_[i] = random_weights_;
      
      BuildTree_(&thread_random_trees_[i], *thread_random_mats_[i]);
      
    }
  }
  
  
} // constructor (pre-split)

// Passing in ghosts for now, but don't do them for naive
template <class TMatcher, class TTraversal, class TTree, class TResults>
npoint_mlpack::NaiveResamplingDriver<TMatcher, TTraversal, TTree, TResults>::
NaiveResamplingDriver(std::vector<arma::mat*>& data_mats_in,
                      std::vector<arma::colvec*>& data_weights_in,
                      arma::mat& random, arma::colvec& rweights,
                      MatcherArguments& matcher_args,
                      int tuple_size,
                      int leaf_size)
:
num_resampling_regions_(data_mats_in.size()),
random_mat_(random),
random_weights_(rweights),
data_mats_(data_mats_in),
data_weights_(data_weights_in),
thread_data_mats_(omp_get_max_threads(), NULL),
thread_data_weights_(omp_get_max_threads(), NULL),
thread_data_trees_(omp_get_max_threads(), NULL),
thread_region_id_(omp_get_max_threads(), -1),
thread_random_mats_(omp_get_max_threads(), NULL),
thread_random_weights_(omp_get_max_threads(), NULL),
thread_random_trees_(omp_get_max_threads(), NULL),
previous_region_(-1),
previous_num_random_(-1),
matcher_args_(matcher_args),
owns_data_(false),
results_(matcher_args, num_resampling_regions_),
tuple_size_(tuple_size),
leaf_size_(leaf_size),
do_random_(random_mat_.n_cols > 0),
num_x_partitions_(-1),
num_y_partitions_(-1),
num_z_partitions_(-1),
resampling_helper_(),
x_step_(-1.0),
y_step_(-1.0),
z_step_(-1.0),
num_points_(-1),
total_computations_(total_num_computations()),
num_prunes_(0),
num_base_cases_(0),
num_pairs_considered_(0),
num_triples_considered_(0)
{
  
  
  if (do_random_) {
    //random_tree_ = new TTree(random_mat_, leaf_size_);
    BuildTree_(&random_tree_, random_mat_);
  }
  else {
    random_tree_ = NULL;
  }
  
  // I don't think we want to call this here
  //GetNextData_();
  
  // Isolating the random matrices per thread
  if (do_random_) {
    for (int i = 0; i < omp_get_max_threads(); i++) {
      
      thread_random_mats_[i] = new arma::mat(random_mat_);
      //*thread_random_mats_[i] = random_mat_;
      
      thread_random_weights_[i] = new arma::colvec(random_weights_);
      //*thread_random_weights_[i] = random_weights_;
      
      BuildTree_(&thread_random_trees_[i], *thread_random_mats_[i]);
      
    }
  }
  
  
} // constructor (pre-split), no ghosts

template <class TMatcher, class TTraversal, class TTree, class TResults>
npoint_mlpack::NaiveResamplingDriver<TMatcher, TTraversal, TTree, TResults>::NaiveResamplingDriver(arma::mat& data, 
                                                                             arma::colvec& weights,
                                                                             arma::mat& random, arma::colvec& rweights,
                                                                             MatcherArguments& matcher_args,
                                                                                                   int num_x_regions, int num_y_regions, int num_z_regions,
                                                                                                   ResamplingHelper& helper,
                                                                                                   int tuple_size,
                                                                                                   int leaf_size) :
num_resampling_regions_(num_x_regions * num_y_regions * num_z_regions),
random_mat_(random),
random_weights_(rweights),
data_mats_(num_resampling_regions_),
data_weights_(num_resampling_regions_),
thread_data_mats_(omp_get_max_threads(), NULL),
thread_data_weights_(omp_get_max_threads(), NULL),
thread_data_trees_(omp_get_max_threads(), NULL),
thread_region_id_(omp_get_max_threads(), -1),
thread_random_mats_(omp_get_max_threads(), NULL),
thread_random_weights_(omp_get_max_threads(), NULL),
thread_random_trees_(omp_get_max_threads(), NULL),
previous_region_(-1),
previous_num_random_(-1),
matcher_args_(matcher_args),
owns_data_(true),
results_(matcher_args,
         num_x_regions * num_y_regions * num_z_regions),
tuple_size_(tuple_size),
leaf_size_(leaf_size),
do_random_(random_mat_.n_cols > 0),
num_x_partitions_(num_x_regions),
num_y_partitions_(num_y_regions),
num_z_partitions_(num_z_regions),
resampling_helper_(helper),
num_points_(data.n_cols),
total_computations_(total_num_computations()),
num_prunes_(0),
num_base_cases_(0),
num_pairs_considered_(0),
num_triples_considered_(0)
{
  
  ResamplingSplitter data_splitter(data, weights,
                                   num_x_partitions_, num_y_partitions_, num_z_partitions_,
                                   resampling_helper_);

  std::vector<arma::mat*> temp_data_mats = data_splitter.data_mats();
  std::vector<arma::colvec*> temp_data_weights = data_splitter.data_weights();
  
  // I think I still need to do this
  // They will hopefully still exist outside, right?
  for (size_t i = 0; i < num_resampling_regions_; i++) {
    
    data_mats_[i] = new arma::mat;
    *(data_mats_[i]) = *(temp_data_mats[i]);

    data_weights_[i] = new arma::colvec;
    *(data_weights_[i]) = *(temp_data_weights[i]);
    
  } // for i
  
  // now, find the step sizes
  x_step_ = data_splitter.x_step();
  y_step_ = data_splitter.y_step();
  z_step_ = data_splitter.z_step();
  
  if (do_random_) {
    //random_tree_ = new TTree(random_mat_, leaf_size_);
    BuildTree_(&random_tree_, random_mat_);
  }
  else {
    random_tree_ = NULL;
  }
  
  // I don't think we want to call this here
  //GetNextData_();
  
  // Isolating the random matrices per thread
  if (do_random_) {
    for (int i = 0; i < omp_get_max_threads(); i++) {
      
      thread_random_mats_[i] = new arma::mat(random_mat_);
      //*thread_random_mats_[i] = random_mat_;
      
      thread_random_weights_[i] = new arma::colvec(random_weights_);
      //*thread_random_weights_[i] = random_weights_;
      
      BuildTree_(&thread_random_trees_[i], *thread_random_mats_[i]);
      
    }
  }
  
} // constructor


template <class TMatcher, class TTraversal, class TTree, class TResults>
int npoint_mlpack::NaiveResamplingDriver<TMatcher, TTraversal, TTree, TResults>::FindRegion_(arma::colvec& col) {
  
  //std::cout << col << "\n";
  
  int x_ind = floor((col(0) - resampling_helper_.x_min()) / (x_step_ + resampling_helper_.epsilon()));
  int y_ind = floor((col(1) - resampling_helper_.y_min()) / (y_step_ + resampling_helper_.epsilon()));
  int z_ind = floor((col(2) - resampling_helper_.z_min()) / (z_step_ + resampling_helper_.epsilon()));
  
  int region_ind = (x_ind + num_x_partitions_ * y_ind
          + num_x_partitions_ * num_y_partitions_ * z_ind);
  
  assert(region_ind >= 0);
  assert(region_ind < (int)num_resampling_regions_);
  
  return region_ind;
  
} // FindRegion


/*
template <class TMatcher, class TTraversal, class TTree, class TResults>
void npoint_mlpack::NaiveResamplingDriver<TMatcher, TTraversal, TTree, TResults>::SplitData_(arma::mat& data,
                                                                                             arma::colvec& weights) {
  
  std::vector<size_t> num_points_per_region(num_resampling_regions_);
  std::vector<std::vector<size_t> > points_in_region_id(num_resampling_regions_);
  
  // loop over the points, determine which region each belongs to
  
  // trying to put this out here because of weird arma errors
  arma::colvec col_i;
  for (size_t i = 0; i < num_points_; i++) {
    
    
    col_i = data.col(i);
    
    // This won't work because of stupid armadillo crap
    //int region_id = FindRegion_(data_all_mat_.col(i));
    
    int region_id = FindRegion_(col_i);
    
    num_points_per_region[region_id]++;
    points_in_region_id[region_id].push_back(i);
  }
  
  // now, loop over regions and insert all the points in them
  // allocate space for them first
  for (size_t region_ind = 0; region_ind < num_resampling_regions_; 
       region_ind++) {
    
    data_mats_[region_ind]->resize(3, num_points_per_region[region_ind]);
    data_weights_[region_ind]->resize(num_points_per_region[region_ind]);
    
    for (size_t point_ind = 0; point_ind < num_points_per_region[region_ind]; 
         point_ind++) {
      
      data_mats_[region_ind]->col(point_ind) = data.col(points_in_region_id[region_ind][point_ind]);
      data_weights_[region_ind]->at(point_ind) = weights(point_ind);
      
    }
    
  }
  
} // SplitData
*/

template<class TMatcher, class TTraversal, class TTree, class TResults>
npoint_mlpack::NaiveResamplingDriver<TMatcher, TTraversal, TTree, TResults>::~NaiveResamplingDriver()
{
  
  if (random_tree_) {
    delete random_tree_;
  }
  
  // only free these if we're supposed to
  if (owns_data_)
  {
    for (size_t i = 0; i < num_resampling_regions_; i++) {
      
      delete data_mats_[i];
      delete data_weights_[i];
      
    }
  }
    
  for (int i = 0; i < omp_get_max_threads(); i++) {
    
    // problem: with single region, I'm just setting this equal to data_mats_[0]
    if (num_resampling_regions_ > 1) {
      if (thread_data_mats_[i]) {
        delete thread_data_mats_[i];
        thread_data_mats_[i] = NULL;
      }
      if (thread_data_weights_[i]) {
        delete thread_data_weights_[i];
        thread_data_weights_[i] = NULL;
      }
      if (thread_data_trees_[i]) {
        delete thread_data_trees_[i];
        thread_data_trees_[i] = NULL;
      }
    }
    
  } // loop over threads
  
} // destructor

// Fills current_data_mat_, current_data_weights_, current_tree
template <class TMatcher, class TTraversal, class TTree, class TResults>
void npoint_mlpack::NaiveResamplingDriver<TMatcher, TTraversal, TTree, TResults>::GetNextData_(int thread_id,
                                                                                               size_t region_needed) 
{
  
  // Do we already have the data we need?
  if (thread_region_id_[thread_id] == region_needed) {
    return;
  }

  // otherwise, we need to check if we can free the data we have and if
  // another thread has the data we need
  
  bool free_data = true;
  //bool data_found = false;
  
  for (int i = 0; i < omp_get_max_threads(); i++){
    
    if (i == thread_id) {
      continue;
    }
    
    // is our data also being used by another thread?
    if (thread_data_mats_[thread_id] == thread_data_mats_[i]) {
      
      free_data = false;
      
    }
    
    
  } // loop over threads
  
  if (free_data) {
    if (thread_data_mats_[thread_id]) {
      delete thread_data_mats_[thread_id];
    }
    if (thread_data_weights_[thread_id]) {
      delete thread_data_weights_[thread_id];
    }
    if (thread_data_trees_[thread_id]) {
      delete thread_data_trees_[thread_id];
    }
  }
  
  thread_data_mats_[thread_id] = NULL;
  thread_data_weights_[thread_id] = NULL;
  thread_data_trees_[thread_id] = NULL;
  thread_region_id_[thread_id] = -1;
  
  for (int i = 0; i < omp_get_max_threads(); i++) {
  
    if (thread_region_id_[i] == region_needed) {
    
      //data_found = true;
      
      thread_data_mats_[thread_id] = thread_data_mats_[i];
      thread_data_weights_[thread_id] = thread_data_weights_[i];
      thread_data_trees_[thread_id] = thread_data_trees_[i];
      thread_region_id_[thread_id] = region_needed;
      
      // now, we're done
      return;
      
    }
  
  } // loop over threads
  
  // need to stick all but the ith data mat together
  if (num_resampling_regions_ > 1) {
    
    // not sure if I need this
    /*
    if (thread_data_mats_[thread_id]) {
      mlpack::Log::Fatal << "Previous data wasn't freed in GetNextDataThreadSafe\n";
    }
     */
    
    thread_data_mats_[thread_id] = new arma::mat();
    thread_data_weights_[thread_id] = new arma::colvec();
    
    for (size_t j = 0; j < num_resampling_regions_; j++) {
      
      if (j != region_needed && data_mats_[j]->n_cols > 0) {
        // this copies data
        thread_data_mats_[thread_id]->insert_cols(thread_data_mats_[thread_id]->n_cols, 
                                                  *(data_mats_[j]));
        thread_data_weights_[thread_id]->insert_rows(thread_data_weights_[thread_id]->n_rows,
                                                     *(data_weights_[j]));
      }
      
    } // for j
    
  }
  else {
    // only one region
    // IMPORTANT: I'm not sure this is thread safe, a later thread will come
    // along and build another tree
    // No it won't, it should just find the one that this thread already built
    
    // This doesn't copy data
    //thread_data_mats_[thread_id] = data_mats_[0];
    //thread_data_weights_[thread_id] = data_weights_[0];
    
    // this does copy data
    thread_data_mats_[thread_id] = new arma::mat(*data_mats_[0]);
    thread_data_weights_[thread_id] = new arma::colvec(*data_weights_[0]);
        
  }
  
  thread_region_id_[thread_id] = region_needed;
  BuildTree_(&(thread_data_trees_[thread_id]), *thread_data_mats_[thread_id]);
  
} // getNextProblem_


// Note: this isn't really thread safe, needs to be called inside critical
template <class TMatcher, class TTraversal, class TTree, class TResults>
void npoint_mlpack::NaiveResamplingDriver<TMatcher, TTraversal, TTree, TResults>::Next_(
                                                      std::vector<arma::mat*>& this_comp_mats,
                     std::vector<arma::colvec*>& this_comp_weights,
                     std::vector<TTree*>& this_comp_trees,
                     std::vector<int>& this_region,
                     int& this_num_randoms)
{
  
  // try having each thread get it's own data
  
  int thread_id = omp_get_thread_num();
  
  size_t region_needed;
  size_t num_random_needed;
  
  num_random_needed = previous_num_random_ + 1;
  
  //std::cout << "previous region: " << previous_region_ << "\n";
  //std::cout << "previous_num_random: " << previous_num_random_ << "\n";
  
  // the first call
  if (previous_region_ > num_resampling_regions_ 
      && previous_num_random_ > tuple_size_) {
    
    //std::cout << "first call to Next";
    region_needed = 0;
    num_random_needed = 0;
    
  }
  // if we aren't doing randoms, then we always move to the next region
  else if (!do_random_) {
    region_needed = previous_region_ + 1;
  }
  else if (num_random_needed >= tuple_size_) {
    
    if (previous_region_ == num_resampling_regions_ - 1) {
      
      // we're doing the all random computation
      num_random_needed = tuple_size_;
      // need this to avoid a compiler warning
      region_needed = -1;
      
    }
    else {
      
      // move to the next region
      region_needed = previous_region_ + 1;
      num_random_needed = 0;
      
    }
    
  }
  else {
    
    region_needed = previous_region_;
    
  }
  
  if (num_random_needed < tuple_size_) {
    GetNextData_(thread_id, region_needed);
  }
  thread_region_id_[thread_id] = region_needed;
  
  for (size_t i = 0; i < num_random_needed && do_random_; i++) {
    
    this_comp_mats[i] = thread_random_mats_[thread_id];
    this_comp_weights[i] = thread_random_weights_[thread_id];
    this_comp_trees[i] = thread_random_trees_[thread_id];
    this_region[i] = thread_region_id_[thread_id];
    
  }
  for (size_t i = num_random_needed; i < tuple_size_; i++) {
    
    // Important, this assumes that the array has the correct matrix
    this_comp_mats[i] = thread_data_mats_[thread_id];
    this_comp_weights[i] = thread_data_weights_[thread_id];
    this_comp_trees[i] = thread_data_trees_[thread_id];
    this_region[i] = thread_region_id_[thread_id];
    
  }
  this_num_randoms = num_random_needed;
  
  previous_region_ = region_needed;
  previous_num_random_ = num_random_needed;
  
} // Next_()


template<class TMatcher, class TTraversal, class TTree, class TResults>
int npoint_mlpack::NaiveResamplingDriver<TMatcher, TTraversal, TTree, TResults>::total_num_computations()
{
  
  int num_comps;
  if (do_random_) {
    // need n per region plus one for the randoms
    num_comps = tuple_size_ * num_resampling_regions_ + 1;
  }
  else {
    // we do 1 per resampling_region
    num_comps = num_resampling_regions_;
  }
  
  //std::cout << "num-comps in naive " << num_comps << "\n";
  
  return num_comps;
  
}


template<class TMatcher, class TTraversal, class TTree, class TResults>
TResults& npoint_mlpack::NaiveResamplingDriver<TMatcher, TTraversal, TTree, TResults>::results() {
  return results_;
}

template<class TMatcher, class TTraversal, class TTree, class TResults>
arma::mat* npoint_mlpack::NaiveResamplingDriver<TMatcher, TTraversal, TTree, TResults>::random_mat() {
  return &random_mat_;
}

template<class TMatcher, class TTraversal, class TTree, class TResults>
arma::colvec* npoint_mlpack::NaiveResamplingDriver<TMatcher, TTraversal, TTree, TResults>::random_weights() {
  return &random_weights_;
}

template<class TMatcher, class TTraversal, class TTree, class TResults>
TTree* npoint_mlpack::NaiveResamplingDriver<TMatcher, TTraversal, TTree, TResults>::random_tree() {
  return random_tree_;
}

// a flag so the Drivers know how to handle results
template<class TMatcher, class TTraversal, class TTree, class TResults>
bool npoint_mlpack::NaiveResamplingDriver<TMatcher, TTraversal, TTree, TResults>::is_efficient() {
  return false;
}

template<class TMatcher, class TTraversal, class TTree, class TResults>
void npoint_mlpack::NaiveResamplingDriver<TMatcher, TTraversal, TTree, TResults>::PrintResults() {
  results_.PrintResults();
}

// Do the actual computation, this is what is called by the main
template<class TMatcher, class TTraversal, class TTree, class TResults>
void npoint_mlpack::NaiveResamplingDriver<TMatcher, TTraversal, TTree, TResults>::Compute() {
    
  // TODO: try different scheduling methods
  // loop over the resampling regions
  
  //#pragma omp threadprivate(this_comp_mats, this_comp_weights, this_comp_trees, this_region, this_num_random)
  //std::cout << "running naive resampling.\n";

  
  //#pragma omp parallel for schedule(dynamic, 1)
  for (int num_computations = 0; num_computations < total_computations_; 
       num_computations++) {
    
    //mlpack::Log::Info << "Running " << omp_get_num_threads() << " threads.\n";
    
    int thread_id = omp_get_thread_num();
    
    //long long int num_leaves_0;

    // These should all be private
    std::vector<arma::mat*> this_comp_mats(tuple_size_);
    std::vector<arma::colvec*> this_comp_weights(tuple_size_);
    std::vector<TTree*> this_comp_trees(tuple_size_);
    std::vector<int> this_region(tuple_size_);
    int this_num_random;
    
    //int my_num_points = mlpack::math::RandInt(1, 100);
    
    //arma::mat my_mat(3, my_num_points);
    //my_mat.randu();

    //TTree* my_tree;
    //BuildTree_(&my_tree, my_mat);
    
    //#pragma omp critical
    {
      //mlpack::Log::Info << "Getting next comp on thread " << omp_get_thread_num() << "\n";
      
      Next_(this_comp_mats, this_comp_weights,
            this_comp_trees, this_region,
           this_num_random);
      
      
      
      mlpack::Log::Info << "Thread " << thread_id << " doing computation on ";
      for (size_t i = 0; i < tuple_size_; i++)
      {
        
        //this_comp_mats[i] = &my_mat;
        //this_comp_trees[i] = my_tree;
        //this_region[i] = 0;
        //this_num_random = omp_get_thread_num();
        
        mlpack::Log::Info << this_comp_mats[i]->n_cols << ", ";
        
      } // loop over tuple size
      mlpack::Log::Info << "points.\n";
      
      /*
       mlpack::Log::Info << "Doing computation " << num_computations;
       mlpack::Log::Info << " with " << this_num_random << " randoms on thread ";
       mlpack::Log::Info << thread_id << ".\n";
       mlpack::Log::Info << "Problem size: " << this_comp_mats[0]->n_cols;
       mlpack::Log::Info << ", " << this_comp_mats[1]->n_cols;
       mlpack::Log::Info << ", " << this_comp_mats[2]->n_cols << "\n";
       //mlpack::Log::Info << "last comp " << local_stop << "\n";
       
       long long int num_leaves;
       int max_depth;
       int min_size;
       int max_size;
       double avg_size;
       double sum_sizes_sq;
       double std_dev;
       
       GetTreeStats(this_comp_trees[0], num_leaves_0, max_depth, min_size,
       max_size, avg_size, sum_sizes_sq);
       
       mlpack::Log::Info << "\nTree Stats 0: \nnum_leaves: " << num_leaves_0;
       mlpack::Log::Info << ", max_depth: " << max_depth;
       mlpack::Log::Info << "\nmin_size: " << min_size;
       mlpack::Log::Info << ", max_size: " << max_size;
       mlpack::Log::Info << ", avg_size: " << avg_size;
       std_dev = (num_leaves_0 / (double)(num_leaves_0 - 1)) * sqrt((sum_sizes_sq / (double)num_leaves_0) - avg_size * avg_size);
       mlpack::Log::Info << ", std. dev: " << std_dev << "\n";
       
       mlpack::Log::Info << "\n\n";
       
       
       GetTreeStats(this_comp_trees[1], num_leaves, max_depth, min_size,
       max_size, avg_size, sum_sizes_sq);
       
       mlpack::Log::Info << "\nTree Stats 1: \nnum_leaves: " << num_leaves;
       mlpack::Log::Info << ", max_depth: " << max_depth;
       mlpack::Log::Info << "\nmin_size: " << min_size;
       mlpack::Log::Info << ", max_size: " << max_size;
       mlpack::Log::Info << ", avg_size: " << avg_size;
       std_dev = (num_leaves / (double)(num_leaves - 1)) * sqrt((sum_sizes_sq / (double)num_leaves) - avg_size * avg_size);
       mlpack::Log::Info << ", std. dev: " << std_dev << "\n";
       
       mlpack::Log::Info << "\n\n";
       
       GetTreeStats(this_comp_trees[2], num_leaves, max_depth, min_size,
       max_size, avg_size, sum_sizes_sq);
       
       mlpack::Log::Info << "\nTree Stats 2: \nnum_leaves: " << num_leaves;
       mlpack::Log::Info << ", max_depth: " << max_depth;
       mlpack::Log::Info << "\nmin_size: " << min_size;
       mlpack::Log::Info << ", max_size: " << max_size;
       mlpack::Log::Info << ", avg_size: " << avg_size;
       std_dev = (num_leaves / (double)(num_leaves - 1)) * sqrt((sum_sizes_sq / (double)num_leaves) - avg_size * avg_size);
       mlpack::Log::Info << ", std. dev: " << std_dev << "\n";
       
       mlpack::Log::Info << "\n\n";
       */
      mlpack::Log::Info << "\n";
    } // critical
    
    
    if ((matcher_args_.arg_type() == MatcherArguments::ANGLE_MATCHER 
         || matcher_args_.arg_type() == MatcherArguments::MULTI_MATCHER
         || matcher_args_.arg_type() == MatcherArguments::UNORDERED_MULTI_MATCHER)
        && (matcher_args_.template_type() == MatcherArguments::TEMPLATE_SINGLE_MATCHER
            || matcher_args_.template_type() == MatcherArguments::TEMPLATE_EFFICIENT_MATCHER)) {
          
          //mlpack::Log::Info << "Doing multiple traversal calls.\n";
          
          //#pragma omp parallel for //private(this_comp_mats, this_comp_weights, this_comp_trees)
          for (int i = 0; i < matcher_args_.total_matchers(); i++)
          {
            
            std::vector<arma::mat*> my_comp_mats = this_comp_mats;
            std::vector<arma::colvec*> my_comp_weights = this_comp_weights;
            std::vector<TTree*> my_comp_trees = this_comp_trees;
            std::vector<int> my_this_region = this_region;
            
            // IMPORTANT: this needs to be thread safe or put in a critical
            MatcherArguments this_args = matcher_args_.GenerateMatcher(i);
            
            TMatcher matcher(my_comp_mats, my_comp_weights,
                             this_args);
            
            matcher.set_matcher_ind(i);

            TTraversal alg(my_comp_trees, matcher);
            
            alg.Compute();
            
            //#pragma omp critical
            {
              
              num_prunes_ += alg.num_prunes();
              num_base_cases_ += alg.num_base_cases();
              num_pairs_considered_ += alg.num_pairwise_distances_computed();
              num_triples_considered_ += alg.num_point_tuples_considered();
              
              //std::cout << "processing results \n";
              
              results_.ProcessResults(my_this_region, this_num_random,
                                      is_efficient(),
                                      matcher);
              
            } // critical (writing results for single matchers)
            
          } // parallel for 
          
        } // if we need to loop over matchers
    else {

      //mlpack::Log::Info << "Doing a single traversal call.\n";
      
      //printf("creating matcher\n");
      // create matcher class
      TMatcher matcher(this_comp_mats, this_comp_weights, 
                       matcher_args_);
      
      //printf("creating alg\n");
      // create alg class
      
      //TTraversal alg(this_comp_trees, matcher, num_leaves_0);
      TTraversal alg(this_comp_trees, matcher);
      
      //std::cout << "Computing for " << this_num_random << " randoms.\n";
      // run alg class
      //printf("doing tree traversal \n");
      alg.Compute();
      
      //alg.PrintStats();
      
      //mlpack::Log::Info << "Num pairs in matcher: " << matcher.num_pairs_considered() << "\n";
      
      
      // process and store results from the matcher
      //#pragma omp critical
      {
        
        //mlpack::Log::Info << "Recording results for thread " << omp_get_thread_num();
        //mlpack::Log::Info << " region: " << this_region[0] << ", randoms: " << this_num_random << "\n";
        
        num_prunes_ += alg.num_prunes();
        num_base_cases_ += alg.num_base_cases();
        num_pairs_considered_ += alg.num_pairwise_distances_computed();
        num_triples_considered_ += alg.num_point_tuples_considered();
        
        results_.ProcessResults(this_region, this_num_random,
                                is_efficient(),
                                matcher);
      } // critical
    
    } // else (no need to loop over matchers)
  
  } // parallel loop over computations to do
  
  //PrintStats();
  
} // Compute


template <class TMatcher, class TTraversal, class TTree, class TResults>
void npoint_mlpack::NaiveResamplingDriver<TMatcher, TTraversal, TTree, TResults>
::PrintRegionSizes() const
{
  
  std::vector<long long int> region_counts(num_resampling_regions_, 0);
  
  for (size_t i = 0; i < num_resampling_regions_;
       i++)
  {
    
    for (size_t j = 0; j < num_resampling_regions_; j++)
    {
      if (i != j)
        region_counts[j] += data_mats_[i]->n_cols;
    }
    
  }

  for (size_t i = 0; i < num_resampling_regions_; i++)
  {
    std::cout << "Region[" << i << "]: " << region_counts[i] << "\n";
  }
  
  std::cout << "Randoms: " << random_mat_.n_cols << "\n";
  
} // PrintRegionSizes()





#endif
