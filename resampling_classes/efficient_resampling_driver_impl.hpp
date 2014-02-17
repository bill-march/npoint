//
//  efficient_resampling_driver_impl.hpp
//  contrib_march
//
//  Created by William March on 7/9/12.
//  Copyright (c) 2012 Georgia Institute of Technology. All rights reserved.
//

#ifndef __MLPACK_NPOINT_RESAMPLING_CLASSES_EFFICIENT_RESAMPLING_DRIVER_IMPL_HPP_
#define __MLPACK_NPOINT_RESAMPLING_CLASSES_EFFICIENT_RESAMPLING_DRIVER_IMPL_HPP_


template <class TMatcher, class TTraversal, class TTree, class TResults>
npoint_mlpack::
EfficientResamplingDriver<TMatcher, TTraversal, TTree, TResults>::
EfficientResamplingDriver(std::vector<arma::mat*>& data_mats_in,
                          std::vector<arma::colvec*>& data_weights_in,
                          std::vector<arma::mat* >& data_ghost_mats_in,
                          std::vector<arma::colvec*>& data_ghost_weights_in,
                          arma::mat& random, arma::colvec& rweights,
                          arma::mat& random_ghost, arma::colvec& rweights_ghost,
                          MatcherArguments& matcher_args,
                          int tuple_size,
                          int leaf_size)
:
// leave this alone for now, shouldn't need it here
data_all_mat_(),
data_all_weights_(),
random_mat_(random),
random_weights_(rweights),
num_resampling_regions_(data_mats_in.size()),
do_random_(random_mat_.n_cols > 0),
data_trees_(num_resampling_regions_),
data_mats_(data_mats_in),
data_weights_(data_weights_in),
have_ghosts_(true),
ghost_data_trees_(num_resampling_regions_),
ghost_data_mats_(data_ghost_mats_in),
ghost_data_weights_(data_ghost_weights_in),
ghost_random_tree_(NULL),
ghost_random_mat_(random_ghost),
ghost_random_weights_(rweights_ghost),
tuple_size_(tuple_size),
leaf_size_(leaf_size),
num_x_partitions_(-1),
num_y_partitions_(-1),
num_z_partitions_(-1),
resampling_helper_(),
num_points_(-1),
results_(matcher_args,
         num_resampling_regions_),
current_tuple_sets_(tuple_size_, 0),
matcher_args_(matcher_args),
// For now, we're assuming that something outside the class owns these pointers
// and will free them when necessary
owns_data_(false),
total_computations_(total_num_computations()),
num_prunes_(0),
num_base_cases_(0),
num_pairs_considered_(0),
num_triples_considered_(0)
{
 
  // TODO: check if we actually have any ghosts here
  
  // data are already split, so we should just need to build the trees
  BuildTrees_();
  
} // constructor (pre-split data)

template <class TMatcher, class TTraversal, class TTree, class TResults>
npoint_mlpack::
EfficientResamplingDriver<TMatcher, TTraversal, TTree, TResults>::
EfficientResamplingDriver(std::vector<arma::mat*>& data_mats_in,
                          std::vector<arma::colvec*>& data_weights_in,
                          arma::mat& random, arma::colvec& rweights,
                          MatcherArguments& matcher_args,
                          int tuple_size,
                          int leaf_size)
:
// leave this alone for now, shouldn't need it here
data_all_mat_(),
data_all_weights_(),
random_mat_(random),
random_weights_(rweights),
num_resampling_regions_(data_mats_in.size()),
do_random_(random_mat_.n_cols > 0),
data_trees_(num_resampling_regions_),
data_mats_(data_mats_in),
data_weights_(data_weights_in),
have_ghosts_(false),
ghost_data_trees_(num_resampling_regions_),
ghost_data_mats_(num_resampling_regions_),
ghost_data_weights_(num_resampling_regions_),
ghost_random_tree_(NULL),
// we don't use these here, but we still need to create them
ghost_random_mat_(0,0),
// I think this is how it's default constructed anyway
// to overcome the overloading problem, just cast to an int
//ghost_random_weights_(0),
tuple_size_(tuple_size),
leaf_size_(leaf_size),
num_x_partitions_(-1),
num_y_partitions_(-1),
num_z_partitions_(-1),
resampling_helper_(),
num_points_(-1),
results_(matcher_args,
         num_resampling_regions_),
current_tuple_sets_(tuple_size_, 0),
matcher_args_(matcher_args),
// For now, we're assuming that something outside the class owns these pointers
// and will free them when necessary
owns_data_(false),
total_computations_(total_num_computations()),
num_prunes_(0),
num_base_cases_(0),
num_pairs_considered_(0),
num_triples_considered_(0)
{
  
  for (size_t i = 0; i < num_resampling_regions_; i++)
  {
    
    // need to build these even though we won't use them
    ghost_data_mats_[i] = new arma::mat();
    ghost_data_weights_[i] = new arma::colvec();
    
  }
  
  // data are already split, so we should just need to build the trees
  BuildTrees_();
  
} // constructor (pre-split data, no ghosts)




template <class TMatcher, class TTraversal, class TTree, class TResults>
npoint_mlpack::
EfficientResamplingDriver<TMatcher, TTraversal, TTree, TResults>::
EfficientResamplingDriver(arma::mat& data, arma::colvec& weights,
                          arma::mat& random, arma::colvec& rweights,
                          MatcherArguments& matcher_args,
                          int num_x_regions, int num_y_regions, 
                          int num_z_regions,
                          ResamplingHelper& helper,
                          int tuple_size,
                          int leaf_size)
:
data_all_mat_(data.memptr(), data.n_rows, data.n_cols, false), 
data_all_weights_(weights),
//random_mat_(random.memptr(), random.n_rows, random.n_cols, false), 
// TODO: make this the more efficient copy too?
random_mat_(random),
random_weights_(rweights),
num_resampling_regions_(num_x_regions * num_y_regions * num_z_regions),
do_random_(random_mat_.n_cols > 0),
data_trees_(num_resampling_regions_),
data_mats_(num_resampling_regions_), 
data_weights_(num_resampling_regions_),
have_ghosts_(false),
ghost_data_trees_(num_resampling_regions_),
ghost_data_mats_(num_resampling_regions_),
ghost_data_weights_(num_resampling_regions_),
ghost_random_tree_(NULL),
// we don't use these here, but we still need to create them
ghost_random_mat_(0,0),
// I think this is how it's default constructed anyway
// to overcome the overloading problem, just cast to an int
//ghost_random_weights_(0),
tuple_size_(tuple_size),
leaf_size_(leaf_size),
num_x_partitions_(num_x_regions),
num_y_partitions_(num_y_regions),
num_z_partitions_(num_z_regions),
resampling_helper_(helper),
num_points_(data.n_cols),
results_(matcher_args,
         num_x_regions * num_y_regions * num_z_regions),
current_tuple_sets_(tuple_size_, 0),
matcher_args_(matcher_args),
// for this constructor, we'll create our own subsamples, so we need to free
// them
owns_data_(true),
total_computations_(total_num_computations()),
num_prunes_(0),
num_base_cases_(0),
num_pairs_considered_(0),
num_triples_considered_(0)
{
  
  //mlpack::Log::Info << "do random: " << do_random_ << "\n";
  
  mlpack::Timer::Start("efficient_resampling_time");
  
  ResamplingSplitter data_splitter(data_all_mat_, data_all_weights_,
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
  
  //SplitData_();
  
  BuildTrees_();
  
  mlpack::Timer::Stop("efficient_resampling_time");

}

template <class TMatcher, class TTraversal, class TTree, class TResults>
npoint_mlpack::
EfficientResamplingDriver<TMatcher, TTraversal, TTree, TResults>::
~EfficientResamplingDriver()
{
  
  // go through and free the data mats we created and trees we built
  for (size_t i = 0; i < num_resampling_regions_; i++)
  {
    
    // we need to avoid freeing the data if we're not responsible for them
    if (owns_data_)
    {
      if (data_mats_[i])
      {
        delete data_mats_[i];
        data_mats_[i] = NULL;
      }
      if (data_weights_[i])
      {
        delete data_weights_[i];
        data_weights_[i] = NULL;
      }
    }
    if (data_trees_[i])
    {
      delete data_trees_[i];
      data_trees_[i] = NULL;
    }
    if (ghost_data_trees_[i])
    {
      delete ghost_data_trees_[i];
      ghost_data_trees_[i] = NULL;
    }
    
  } // loop over resampling regions
  
  if (random_tree_)
  {
    delete random_tree_;
    random_tree_ = NULL;
  }
  if (ghost_random_tree_)
  {
    delete ghost_random_tree_;
    ghost_random_tree_ = NULL;
  }
  
}

// TODO: should these still be public (or even exist)?

template <class TMatcher, class TTraversal, class TTree, class TResults>
arma::mat* npoint_mlpack::EfficientResamplingDriver<TMatcher, TTraversal, TTree, TResults>::random_mat()
{
  return random_mat_;
}

template <class TMatcher, class TTraversal, class TTree, class TResults>
arma::colvec* npoint_mlpack::EfficientResamplingDriver<TMatcher, TTraversal, TTree, TResults>::random_weights()
{
  return random_weights_;
}

template <class TMatcher, class TTraversal, class TTree, class TResults>
TTree* npoint_mlpack::EfficientResamplingDriver<TMatcher, TTraversal, TTree, TResults>::random_tree()
{
  return random_tree_;
}

// a flag so the Drivers know how to handle results
template <class TMatcher, class TTraversal, class TTree, class TResults>
bool npoint_mlpack::EfficientResamplingDriver<TMatcher, TTraversal, TTree, TResults>::is_efficient()
{
  return true;
}

template <class TMatcher, class TTraversal, class TTree, class TResults>
int npoint_mlpack::EfficientResamplingDriver<TMatcher, TTraversal, TTree, TResults>::total_num_computations()
{
 
  // We're counting with replacement, not counting distinct permutations
  // Therefore, for J resampling regions and tuple size n, we want the number
  // of n-tuples of resampling regions (plus a random set) of size n
  // This is equal to (J + 1 + n - 1) choose n
  
  int num_comps;
  
  // This is the old code, which works for no ghost points
  int num_fac = num_resampling_regions_ + tuple_size_;
  int num_tot = 1;
  int den_tot = 1;
  
  int den_fac = 1;
  
  // its ((num_resampling_regions_ + tuple_size_) choose tuple_size_)
  for (int i = 0; i < tuple_size_; i++) {
    
    num_tot *= num_fac;
    num_fac--;
    
    den_tot *= den_fac;
    den_fac++;
    
  }
  
  num_comps = num_tot / den_tot;
  
  if (!do_random_) { 
    // we overcounted 
    num_comps /= num_resampling_regions_ + tuple_size_;
  }
  
  //mlpack::Log::Info << "num resampling comps before ghosts: " << num_comps << "\n";
  
    
  if (have_ghosts_)
  {
   
    // With ghosts, we now have 2 * J + 2 regions to choose from, a ghost and
    // regular random and two for each resampling region.
    // However, we don't want to count tuples only containing ghosts.  The
    // number of these is equal to the result found above, so we subtract it
    // at the end.
    
    int num_no_ghost_comps = num_comps;
    
    num_fac = (2 * num_resampling_regions_) + tuple_size_ + 1;
    num_tot = 1;
    den_tot = 1;
    
    den_fac = 1;
    
    // its ((num_resampling_regions_ + tuple_size_) choose tuple_size_)
    for (int i = 0; i < tuple_size_; i++) {
      
      num_tot *= num_fac;
      num_fac--;
      
      den_tot *= den_fac;
      den_fac++;
      
    }
    
    num_comps = num_tot / den_tot;
    
    if (!do_random_) {
      // we overcounted
      num_comps /= (2 * num_resampling_regions_) + tuple_size_;
    }
    
    // This should handle the overcounting
    num_comps = num_comps - num_no_ghost_comps;

  }
  
  //mlpack::Log::Info << "Efficient resampling num comps: " << num_comps << "\n";
  
  return num_comps;

}

// actually does the computation
template <class TMatcher, class TTraversal, class TTree, class TResults>
void npoint_mlpack::EfficientResamplingDriver<TMatcher, TTraversal, TTree, TResults>::Compute()
{
  
  //#pragma omp parallel for schedule(dynamic, 1) num_threads(1)
  for (int num_computations = 0; num_computations < total_computations_; 
       num_computations++) {
    
    //mlpack::Log::Info << "Running " << omp_get_num_threads() << " threads.\n";
    
    int thread_id = omp_get_thread_num();
    
    // These should all be private
    std::vector<arma::mat*> this_comp_mats(tuple_size_);
    std::vector<arma::colvec*> this_comp_weights(tuple_size_);
    std::vector<TTree*> this_comp_trees(tuple_size_);
    std::vector<int> this_region(tuple_size_);
    int this_num_random;
    
    //#pragma omp critical
    {
      //mlpack::Log::Info << "Getting next comp on thread " << omp_get_thread_num() << "\n";
      
      // This isn't thread safe because it updates current_tuple_sets_
      Next_(this_comp_mats, this_comp_weights,
            this_comp_trees, this_region,
            this_num_random);
      
      mlpack::Log::Info << "Thread " << thread_id << " doing computation on ";
      for (int i = 0; i < tuple_size_; i++) 
      {
        
        mlpack::Log::Info << this_comp_mats[i]->n_cols << ", ";
        
      } // loop over tuple size
      mlpack::Log::Info << "points.\n\n";
      
    } // critical
    
    // we need to determine whether the matcher arguments and matcher require 
    // us to loop here
    // if the arguments call for multiple matchers but the template is a single
    // matcher (or efficient matcher), then we need to loop over matchers here
    
    
    // need to be careful with parallel loops here, will mean multiple levels
    // of nesting and possibly sharing pointers to matrices above
    
    // if we have specified multiple matchers in the arguments but are 
    // using a single matcher class for the template arguments to this class
    if ((matcher_args_.arg_type() == MatcherArguments::ANGLE_MATCHER 
        || matcher_args_.arg_type() == MatcherArguments::MULTI_MATCHER
         || matcher_args_.arg_type() == MatcherArguments::UNORDERED_MULTI_MATCHER)
        && (matcher_args_.template_type() == MatcherArguments::TEMPLATE_SINGLE_MATCHER
            || matcher_args_.template_type() == MatcherArguments::TEMPLATE_EFFICIENT_MATCHER)) {
      
          //#pragma omp parallel for num_threads(1)
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
              
              results_.ProcessResults(this_region, this_num_random,
                                      is_efficient(),
                                      matcher);
              
            } // critical (writing results for single matchers)
            
          } // parallel for 
      
    }
    else {
      // we don't need to loop over matchers, so we'll just do one and exit
      
      // create matcher class
      TMatcher matcher(this_comp_mats, this_comp_weights, 
                       matcher_args_);
      
      //std::cout << "matcher built, building algorithm.\n";
      // create alg class
      TTraversal alg(this_comp_trees, matcher);
      
      //std::cout << "running computation.\n";
      // run alg class
      alg.Compute();
      
      //std::cout << "finished computation\n";
      //alg.PrintStats();
      
      // process and store results from the matcher
      //#pragma omp critical
      {
        
        //mlpack::Log::Info << "Recording results for thread " << omp_get_thread_num();
        //mlpack::Log::Info << " region: " << this_region[0] << ", randoms: " << this_num_random << "\n";
        
        num_prunes_ += alg.num_prunes();
        num_base_cases_ += alg.num_base_cases();
        num_pairs_considered_ += alg.num_pairwise_distances_computed();
        num_triples_considered_ += alg.num_point_tuples_considered();
        
        //std::cout << "processing results \n";
        
        results_.ProcessResults(this_region, this_num_random,
                                is_efficient(),
                                matcher);
      } // critical
      
    } // don't need to loop over matchers
    
  } // parallel loop over computations to do
  
  //PrintStats();
  
}

template <class TMatcher, class TTraversal, class TTree, class TResults>
void npoint_mlpack::EfficientResamplingDriver<TMatcher, TTraversal, TTree, TResults>::PrintResults()
{
  
  results_.PrintResults();
  
}

template <class TMatcher, class TTraversal, class TTree, class TResults>
TResults& npoint_mlpack::EfficientResamplingDriver<TMatcher, TTraversal, TTree, TResults>::results()
{
  return results_;
}



template <class TMatcher, class TTraversal, class TTree, class TResults>
void npoint_mlpack::EfficientResamplingDriver<TMatcher, TTraversal, TTree, TResults>::BuildTree_(TTree** tree, arma::mat& data)
{
  
  *tree = new TTree(data, leaf_size_);
  
}


template <class TMatcher, class TTraversal, class TTree, class TResults>
int npoint_mlpack::EfficientResamplingDriver<TMatcher, TTraversal, TTree, TResults>::FindRegion_(arma::colvec& col) {
  
  int x_ind = floor((col(0) - resampling_helper_.x_min()) / (x_step_ + resampling_helper_.epsilon()));
  int y_ind = floor((col(1) - resampling_helper_.y_min()) / (y_step_ + resampling_helper_.epsilon()));
  int z_ind = floor((col(2) - resampling_helper_.z_min()) / (z_step_ + resampling_helper_.epsilon()));
  
  int region_ind = (x_ind + num_x_partitions_ * y_ind 
          + num_x_partitions_ * num_y_partitions_ * z_ind);
  
  //std::cout << col << "\n";
  
  assert(region_ind >= 0);
  assert(region_ind < (int)num_resampling_regions_);
  
  return region_ind;
  
} // FindRegion

/*
template <class TMatcher, class TTraversal, class TTree, class TResults>
void npoint_mlpack::EfficientResamplingDriver<TMatcher, TTraversal, TTree, TResults>::SplitData_() {
  
  std::vector<size_t> num_points_per_region(num_resampling_regions_);
  std::vector<std::vector<size_t> > points_in_region_id(num_resampling_regions_);
  
  // loop over the points, determine which region each belongs to
  for (int i = 0; i < num_points_; i++) {
    
    arma::colvec col_i;
    col_i = data_all_mat_.col(i);
    
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
    
    for (size_t point_ind = 0; point_ind < num_points_per_region[region_ind]; 
         point_ind++) {
      
      data_mats_[region_ind]->col(point_ind) = data_all_mat_.col(points_in_region_id[region_ind][point_ind]);
      
    }
    
  }

} // SplitData
*/

template <class TMatcher, class TTraversal, class TTree, class TResults>
void npoint_mlpack::EfficientResamplingDriver<TMatcher, TTraversal, TTree, TResults>::BuildTrees_() {
  
  //mlpack::Timer::Start("efficient_tree_building");
  
  if (do_random_) {
    //random_tree_ = new TTree(random_mat_, leaf_size_);
    BuildTree_(&random_tree_, random_mat_);
    
    if (have_ghosts_ && ghost_random_mat_.n_cols > 0)
    {
      BuildTree_(&ghost_random_tree_, ghost_random_mat_);
    }
    else
    {
      ghost_random_tree_ = NULL;
    }
    
  }
  else {
    random_tree_ = NULL;
    ghost_random_tree_ = NULL;
  }
  // TODO: add old_from_new vectors for weight permutations
  for (size_t i = 0; i < num_resampling_regions_; i++)
  {
    
    // some regions might have no points (especially for small tests)
    if (data_mats_[i]->n_cols > 0)
    {
      //data_trees_[i] = new TTree(*data_mats_[i], leaf_size_);
      BuildTree_(&data_trees_[i], *data_mats_[i]);
    }
    else {
      data_trees_[i] = NULL;
    }
    
    if (have_ghosts_ && ghost_data_mats_[i]->n_cols > 0)
    {
      BuildTree_(&ghost_data_trees_[i], *ghost_data_mats_[i]);
    }
    else
    {
      ghost_data_trees_[i] = NULL;
    }
    
  } // for i
  
  //mlpack::Timer::Stop("efficient_tree_building");
  
} // BuildTrees_


template <class TMatcher, class TTraversal, class TTree, class TResults>
void npoint_mlpack::EfficientResamplingDriver<TMatcher, TTraversal, TTree, TResults>::Next_(std::vector<arma::mat*>& this_comp_mats,
                                           std::vector<arma::colvec*>& this_comp_weights,
                                           std::vector<TTree*>& this_comp_trees,
                                           std::vector<int>& this_region,
                                           int& this_num_randoms)
{
  
  // need to iterate through all unique n-tuples of sets, with replacement
  
  this_num_randoms = 0;
  
  for (int i = 0; i < tuple_size_; i++)
  {
    
    // we're adding the ghost random set
    if (current_tuple_sets_[i] == 2 * num_resampling_regions_ + 1)
    {
      this_comp_mats[i] = &ghost_random_mat_;
      this_comp_weights[i] = &ghost_random_weights_;
      this_comp_trees[i] = ghost_random_tree_;
      this_num_randoms++;
      // mark it as a random set
      this_region[i] = num_resampling_regions_;
    }
    // we're adding a ghost data set
    else if (current_tuple_sets_[i] > num_resampling_regions_)
    {
      
      // Is this rights?
      size_t ghost_ind = current_tuple_sets_[i] - num_resampling_regions_ - 1;
      
      this_comp_mats[i] = ghost_data_mats_[ghost_ind];
      this_comp_weights[i] = ghost_data_weights_[ghost_ind];
      this_comp_trees[i] = ghost_data_trees_[ghost_ind];
      
      // this should be the index of the resampling region we're handling
      // ghosts from
      this_region[i] = ghost_ind;
      
    }
    // we're adding the real random set
    else if (current_tuple_sets_[i] == num_resampling_regions_) {
      this_comp_mats[i] = &random_mat_;
      this_comp_weights[i] = &random_weights_;
      this_comp_trees[i] = random_tree_;
      this_num_randoms++;
      
      this_region[i] = current_tuple_sets_[i];
      
    }
    // we're adding a real data set
    else {
      this_comp_mats[i] = data_mats_[current_tuple_sets_[i]];
      this_comp_weights[i] = data_weights_[current_tuple_sets_[i]];
      this_comp_trees[i] = data_trees_[current_tuple_sets_[i]];
      this_region[i] = current_tuple_sets_[i];
    }
    
  } // for i
  
  /*
  mlpack::Log::Info << "current_tuple_sets: ";
  for (int i = 0; i < tuple_size_; i++)
  {
    mlpack::Log::Info << current_tuple_sets_[i] << ", ";
  }
  mlpack::Log::Info << "\n\n";
  */
  
  //std::cout << "Incrementing tuple sets\n";
  IncrementCurrentTupleSets_();
  
} // Next()

// increment the ith place
template <class TMatcher, class TTraversal, class TTree, class TResults>
void npoint_mlpack::EfficientResamplingDriver<TMatcher, TTraversal, TTree, TResults>::IncrementCurrentTupleSetsHelper_(int i)
{
  
  current_tuple_sets_[i]++;
  
  // If we're computing with a random set, then we need to include it in the tuple
  // otherwise, we need to stop one earlier
  // TODO: make this a class variable, don't need to keep recomputing it
  size_t cutoff;
  
  // we don't use ghosts for the first position
  if (i == 0 || !have_ghosts_)
  {
    cutoff = do_random_ ? num_resampling_regions_ : num_resampling_regions_ - 1;
  }
  else
  {
    // I'm not sure we'll ever have ghosts with no randoms, but I'll cover it
    // just to be safe
    cutoff = do_random_ ? 2 * num_resampling_regions_ + 1:
                          2 * (num_resampling_regions_) - 1;
  }

  // do we need to change any of the other numbers?
  if (current_tuple_sets_[i] > cutoff) {
    
    if (i > 0) {
      IncrementCurrentTupleSetsHelper_(i-1);
      // now, the i-1th place is correct
      current_tuple_sets_[i] = current_tuple_sets_[i-1];
    }
    else {
      // now, we're done
      current_tuple_sets_[i] = -1;
    }
  }
  
  
} // IncrementHelper

template <class TMatcher, class TTraversal, class TTree, class TResults>
void npoint_mlpack::EfficientResamplingDriver<TMatcher, TTraversal, TTree, TResults>::IncrementCurrentTupleSets_() 
{
  
  IncrementCurrentTupleSetsHelper_(tuple_size_ - 1);
  
}

template <class TMatcher, class TTraversal, class TTree, class TResults>
void npoint_mlpack::EfficientResamplingDriver<TMatcher, TTraversal, TTree, TResults>
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
    mlpack::Log::Info << "Region[" << i << "]: " << region_counts[i] << "\n";
  }
  
  mlpack::Log::Info << "Randoms: " << random_mat_.n_cols << "\n";
  
} // PrintRegionSizes()




#endif
