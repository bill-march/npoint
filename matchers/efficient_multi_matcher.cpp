/*
 *  efficient_multi_matcher.cpp
 *
 *
 *  Copyright 2011 Georgia Institute of Technology. All rights reserved.
 *
 */

#include "efficient_multi_matcher.hpp"

npoint_mlpack::EfficientMultiMatcher::PartialResult::PartialResult(const std::vector<int>& results_size)
:
num_results_(results_size[0]),
results_(num_results_, 0)
{}

npoint_mlpack::EfficientMultiMatcher::PartialResult::~PartialResult() {}

void npoint_mlpack::EfficientMultiMatcher::PartialResult::Reset()
{
  results_.assign(num_results_, 0);
}

int npoint_mlpack::EfficientMultiMatcher::PartialResult::num_results() const
{
  return num_results_;
}

const std::vector<long long int>& npoint_mlpack::EfficientMultiMatcher::PartialResult::results() const
{
  return results_;
}

std::vector<long long int>& npoint_mlpack::EfficientMultiMatcher::PartialResult::results()
{
  return results_;
}

npoint_mlpack::EfficientMultiMatcher::PartialResult&
npoint_mlpack::EfficientMultiMatcher::PartialResult::operator=(const PartialResult& other)
{
  
  if (this != &other) {
    
    num_results_ = other.num_results();
    results_ = other.results();
    
  }
  
  return *this;
  
}

npoint_mlpack::EfficientMultiMatcher::PartialResult&
npoint_mlpack::EfficientMultiMatcher::PartialResult::operator+=(const PartialResult& other)
{
  
  if (this != &other)
  {
    
    if (num_results_ != other.num_results()) {
      mlpack::Log::Fatal << "Using operator+= on mismatched EfficientMultiMatcher PartialResults\n";
    }
    
    for (int i = 0; i < num_results_; i++)
    {
      results_[i] += other.results()[i];
    }
    
  }
  
  return *this;
  
}

const npoint_mlpack::EfficientMultiMatcher::PartialResult
npoint_mlpack::EfficientMultiMatcher::PartialResult::operator+(const PartialResult &other) const
{
  PartialResult result = *this;
  result += other;
  return result;
}


npoint_mlpack::EfficientMultiMatcher::EfficientMultiMatcher(const std::vector<arma::mat*>& data_in,
                                          const std::vector<arma::colvec*>& weights_in,
                                          MatcherArguments& args)
:
data_mat_list_(data_in),
data_weights_list_(weights_in),
total_matchers_(1),
tuple_size_(args.tuple_size()),
tuple_size_choose_2_(args.num_bands().size()),
min_bands_(args.min_bands()),
max_bands_(args.max_bands()),
num_bands_(args.num_bands()),
band_steps_(tuple_size_choose_2_),
sorted_upper_bounds_sq_(tuple_size_choose_2_),
sorted_lower_bounds_sq_(tuple_size_choose_2_),
perms_(args.tuple_size()),
num_permutations_(perms_.num_permutations()),
matcher_ind_(-1),
results_size_(1),
do_off_diagonal_(args.do_off_diagonal())
{
  
  if (tuple_size_ != 3) {
    mlpack::Log::Fatal << "EfficientMultiMatcher only supports 3-point computations.\n";
  }
  
  // don't think this actually gets used in the new version of the code
  //FillThreePointCorrelationLUT(satisfiability);
  if (do_off_diagonal_) {
    for (int i = 0; i < tuple_size_choose_2_; i++) {
      
      double band_step = (max_bands_[i] - min_bands_[i]) / (double)num_bands_[i];
      band_steps_[i] = band_step;
      
      std::vector<double> bands_i(num_bands_[i]);
      
      std::vector<double> lower_i(num_bands_[i]);
      std::vector<double> upper_i(num_bands_[i]);
      
      if (num_bands_[i] > 1) {
        
        for (int j = 0; j < num_bands_[i]; j++) {
          
          bands_i[j] = min_bands_[i] + (double)j * band_step;
          
          lower_i[j] = (min_bands_[i] + (double)j * band_step)
          * (min_bands_[i] + (double)j * band_step);
          upper_i[j] = (min_bands_[i] + (double)(j+1) * band_step)
          * (min_bands_[i] + (double)(j+1) * band_step);
          
        } // for j
        
      } // more than one bandwidth
      else {
        
        bands_i[0] = min_bands_[i];
        lower_i[0] = min_bands_[i] * min_bands_[i];
        upper_i[0] = max_bands_[i] * max_bands_[i];
        
      }
      
      matcher_lower_bounds_sq_.push_back(lower_i);
      matcher_upper_bounds_sq_.push_back(upper_i);
      
      // no longer need half band
      sorted_lower_bounds_sq_[i] = min_bands_[i] * min_bands_[i];
      sorted_upper_bounds_sq_[i] = max_bands_[i] * max_bands_[i];
      
      total_matchers_ *= num_bands_[i];
      
    } // for i
    
    // Fill in the pointers of matcher distances
    MultiMatcherGenerator generator;
    generator.Init(min_bands_, max_bands_, num_bands_, tuple_size_);
    
    lower_bounds_sqr_ptr_ = new double*[total_matchers_];
    upper_bounds_sqr_ptr_ = new double*[total_matchers_];
    
    for (int i = 0; i < total_matchers_; i++)
    {
      lower_bounds_sqr_ptr_[i] = new double[3];
      upper_bounds_sqr_ptr_[i] = new double[3];
      
      arma::mat& lower_bounds = generator.lower_matcher(i);
      arma::mat& upper_bounds = generator.upper_matcher(i);
      
      lower_bounds_sqr_ptr_[i][0] = lower_bounds(0,1) * lower_bounds(0,1);
      lower_bounds_sqr_ptr_[i][1] = lower_bounds(0,2) * lower_bounds(0,2);
      lower_bounds_sqr_ptr_[i][2] = lower_bounds(1,2) * lower_bounds(1,2);

      upper_bounds_sqr_ptr_[i][0] = upper_bounds(0,1) * upper_bounds(0,1);
      upper_bounds_sqr_ptr_[i][1] = upper_bounds(0,2) * upper_bounds(0,2);
      upper_bounds_sqr_ptr_[i][2] = upper_bounds(1,2) * upper_bounds(1,2);
      
    }
  } // doing off diagonal
  else { // just doing equilateral
    
    total_matchers_ = num_bands_[0];
    
    for (int i = 0; i < tuple_size_choose_2_; i++) {
      
      // no longer need half band
      sorted_lower_bounds_sq_[i] = min_bands_[0] * min_bands_[0];
      sorted_upper_bounds_sq_[i] = max_bands_[0] * max_bands_[0];
      
    } // for i
    
    lower_bounds_sqr_ptr_ = new double*[total_matchers_];
    upper_bounds_sqr_ptr_ = new double*[total_matchers_];
    
    for (int i = 0; i < total_matchers_; i++)
    {
      lower_bounds_sqr_ptr_[i] = new double[3];
      upper_bounds_sqr_ptr_[i] = new double[3];
      
      std::vector<double> lower(total_matchers_);
      std::vector<double> upper(total_matchers_);
      
      if (total_matchers_ > 1) {
        
        double band_step = (max_bands_[0] - min_bands_[0]) / (double)num_bands_[0];
        
        for (int j = 0; j < total_matchers_; j++) {
          
          lower[j] = (min_bands_[0] + (double)j * band_step)
          * (min_bands_[0] + (double)j * band_step);
          upper[j] = (min_bands_[0] + (double)(j+1) * band_step)
          * (min_bands_[0] + (double)(j+1) * band_step);
          
        } // for j
        
      } // more than one bandwidth
      else {
        
        lower[0] = min_bands_[0] * min_bands_[0];
        upper[0] = max_bands_[0] * max_bands_[0];
        
      }

      lower_bounds_sqr_ptr_[i][0] = lower[i];
      lower_bounds_sqr_ptr_[i][1] = lower[i];
      lower_bounds_sqr_ptr_[i][2] = lower[i];
      
      upper_bounds_sqr_ptr_[i][0] = upper[i];
      upper_bounds_sqr_ptr_[i][1] = upper[i];
      upper_bounds_sqr_ptr_[i][2] = upper[i];
      
    }

  }
  /*
  std::cout << "multi_matcher bounds ptr\n";
  for (int i = 0; i < total_matchers_; i++)
  {
    
    std::cout << "lower " << i << "\n";
    std::cout << lower_bounds_sqr_ptr_[i][0] << ", " << lower_bounds_sqr_ptr_[i][1] << ", " << lower_bounds_sqr_ptr_[i][2] << "\n";

    std::cout << "upper " << i << "\n";
    std::cout << upper_bounds_sqr_ptr_[i][0] << ", " << upper_bounds_sqr_ptr_[i][1] << ", " << upper_bounds_sqr_ptr_[i][2] << "\n";

  }
  */
  std::sort(sorted_lower_bounds_sq_.begin(), sorted_lower_bounds_sq_.end());
  std::sort(sorted_upper_bounds_sq_.begin(), sorted_upper_bounds_sq_.end());
  
  results_.resize(total_matchers_);
  weighted_results_.resize(total_matchers_);
  
  results_size_[0] = total_matchers_;
  
  //  mlpack::Log::Info << "results inside multi matcher\n";
  for (int i = 0; i < total_matchers_; i++) {
    //mlpack::Log::Info << "results[" << i << "]: " << results_[i] << "\n";
    results_[i] = 0;
    weighted_results_[i] = 0.0;
    
  }
  
} // constructor

npoint_mlpack::EfficientMultiMatcher::~EfficientMultiMatcher()
{
  
  for (int i = 0; i < total_matchers_; i++)
  {
    delete lower_bounds_sqr_ptr_[i];
    delete upper_bounds_sqr_ptr_[i];
  }

  delete lower_bounds_sqr_ptr_;
  delete upper_bounds_sqr_ptr_;

}

bool npoint_mlpack::EfficientMultiMatcher::TestNodeTuple(NodeTuple& nodes) {
  
  std::vector<double> node_lower;
  std::vector<double> node_upper;
  
  double d01_lo = nodes.node_list(0)->Bound().MinDistance(nodes.node_list(1)->Bound());
  double d01_hi = nodes.node_list(0)->Bound().MaxDistance(nodes.node_list(1)->Bound());
  node_lower.push_back(d01_lo);
  node_upper.push_back(d01_hi);
  
  double d02_lo = nodes.node_list(0)->Bound().MinDistance(nodes.node_list(2)->Bound());
  double d02_hi = nodes.node_list(0)->Bound().MaxDistance(nodes.node_list(2)->Bound());
  node_lower.push_back(d02_lo);
  node_upper.push_back(d02_hi);

  double d12_lo = nodes.node_list(1)->Bound().MinDistance(nodes.node_list(2)->Bound());
  double d12_hi = nodes.node_list(1)->Bound().MaxDistance(nodes.node_list(2)->Bound());
  node_lower.push_back(d12_lo);
  node_upper.push_back(d12_hi);
  
  return TestNodeTuple(node_lower, node_upper);
  
} // TestNodeTuple


bool npoint_mlpack::EfficientMultiMatcher::TestNodeTuple(std::vector<double>& min_dists_sqr,
                                                std::vector<double>& max_dists_sqr)
{
  
  bool possibly_valid = true;
  
  std::sort(min_dists_sqr.begin(), min_dists_sqr.end());
  std::sort(max_dists_sqr.begin(), max_dists_sqr.end());
  
  if (min_dists_sqr[0] > sorted_upper_bounds_sq_[0]
      || max_dists_sqr[0] < sorted_lower_bounds_sq_[0])
  {
    possibly_valid = false;
  }
  else if (min_dists_sqr[1] > sorted_upper_bounds_sq_[1]
      || max_dists_sqr[1] < sorted_lower_bounds_sq_[1])
  {
    possibly_valid = false;
  }
  else if (min_dists_sqr[2] > sorted_upper_bounds_sq_[2]
           || max_dists_sqr[2] < sorted_lower_bounds_sq_[2])
  {
    possibly_valid = false;
  }
  
  return possibly_valid;
  
}

void npoint_mlpack::EfficientMultiMatcher::ComputeBaseCase(NodeTuple& nodes) {
  
  PartialResult this_result(results_size_);
  ComputeBaseCase(nodes, this_result);
  AddResult(this_result);
  
} // ComputeBaseCase

void npoint_mlpack::EfficientMultiMatcher::ComputeBaseCase(NodeTuple& nodes,
                                                  PartialResult& result)
{
  
  NptNode* nodeA = nodes.node_list(0);
  NptNode* nodeB = nodes.node_list(1);
  NptNode* nodeC = nodes.node_list(2);
  
  int numA = nodeA->Count();
  int numB = nodeB->Count();
  int numC = nodeC->Count();
  
  if (numA > 64 || numB > 64 || numC > 64)
  {
    mlpack::Log::Fatal << "Calling base case with large nodes in EfficientMultiMatcher.\n";
  }
  
  const double3* pointsA = (double3*)data_mat_list_[0]->colptr(nodeA->Begin());
  const double3* pointsB = (double3*)data_mat_list_[1]->colptr(nodeB->Begin());
  const double3* pointsC = (double3*)data_mat_list_[2]->colptr(nodeC->Begin());
  
  NptRuntimes runtime;
  
  //std::cout << "allocating results in base case\n";
  uint64_t** kernel_results = new uint64_t*[total_matchers_];
  for (int i = 0; i < total_matchers_; i++) {
    kernel_results[i] = new uint64_t[4];
  }
  
  // TODO: prune out some of the matchers here
  ComputeThreePointCorrelationCountsMultiCPU(kernel_results, runtime,
                                           pointsA, numA,
                                           pointsB, numB,
                                           pointsC, numC,
                                           lower_bounds_sqr_ptr_,
                                           upper_bounds_sqr_ptr_,
                                           total_matchers_,
                                           &satisfiability[0]);
  
  // Need to handle the overcounting we may do
  // This assumes that nodes are either identical or don't overlap
  int num_same_nodes = 0;
  // not sure if this works
  if (nodeA == nodeB) {
    num_same_nodes++;
  }
  if (nodeB == nodeC) {
    num_same_nodes++;
  }
  // note that A cant equal C unless both equal B by symmetry prunes
  
  // it's impossible for 0 and 2 to be equal but not be equal to 1
  int overcounting_factor;
  if (num_same_nodes == 0) {
    overcounting_factor = 1;
  }
  else if (num_same_nodes == 1) {
    overcounting_factor = 2;
  }
  // all three same
  else {
    overcounting_factor = 6;
  }
  
  for (int i = 0; i < total_matchers_; i++)
  {
    result.results()[i] += (kernel_results[i][3] / overcounting_factor);
  }
  
  for (int i = 0; i < total_matchers_; i++) {
    delete kernel_results[i];
  }
  delete kernel_results;
  
}

void npoint_mlpack::EfficientMultiMatcher::ComputeBaseCase(NptNode* nodeA,
                                                  NptNode* nodeB,
                                                  std::vector<NptNode*>& nodeC_list,
                                                  PartialResult& result)
{
  
  // IMPORTANT: several possibilities for overcounting here
  // A == B,
  // A == C for some (1) C
  // B == C for some (1) C
  // A == B == C for some (1) C
  
  int countA = nodeA->Count();
  int countB = nodeB->Count();
  
  int countC = 0;
  for (size_t i = 0; i < nodeC_list.size(); i++) {
    
    NptNode* nodeC = nodeC_list[i];
    
    if (!(nodeC == nodeA || nodeC == nodeB)) {
      countC += nodeC->Count();
    }
    
  }
  
  // Don't think I really need this
  // can't check if countC is zero here, because I need to look at
  // the ones where C is equal to one of the other two
  if (countA <= 0 || countB <= 0) {
    return;
  }
  
  double3* pointsA = (double3*)data_mat_list_[0]->colptr(nodeA->Begin());
  double3* pointsB = (double3*)data_mat_list_[1]->colptr(nodeB->Begin());
  double3* pointsC = new double3[countC * sizeof(double3)];
  
  int copyC = 0;
  for (size_t i = 0; i < nodeC_list.size(); i++) {
    
    NptNode* nodeC = nodeC_list[i];
    
    // special case where some of the nodes are equal
    if (nodeC == nodeA || nodeC == nodeB) {
      
      std::vector<NptNode*> this_node_list(3);
      this_node_list[0] = nodeA;
      this_node_list[1] = nodeB;
      this_node_list[2] = nodeC;
      
      //std::cout << "Doing symmetry base case.\n";
      NodeTuple this_tuple(this_node_list);
      ComputeBaseCase(this_tuple, result);
      
    }
    else {
      
      std::memcpy(pointsC+copyC, data_mat_list_[2]->colptr(nodeC->Begin()),
                  nodeC->Count() * sizeof(double3));
      copyC += nodeC->Count();
      
    } // none of the nodes are the same
    
  } // loop over C
  
  // this gets initialized in the kernel
  uint64_t** kernel_results = new uint64_t*[total_matchers_];
  for (int i = 0; i < total_matchers_; i++) {
    kernel_results[i] = new uint64_t[4];
  }
  
  uint64_t* this_result = new uint64_t[total_matchers_];
  memset(this_result, 0, total_matchers_ * sizeof(uint64_t));
  
  NptRuntimes runtime;
  
  // do we have any work to do
  if (countC > 0) {
    
    int elements_remaining_A = countA;
    int startA = 0;
    while (elements_remaining_A > 0) {
      
      int numA = std::min(64, elements_remaining_A);
      
      int elements_remaining_B = countB;
      int startB = 0;
      while (elements_remaining_B > 0) {
        
        int numB = std::min(64, elements_remaining_B);
        
        int elements_remaining_C = countC;
        int startC = 0;
        while (elements_remaining_C > 0) {
          
          int numC = std::min(64, elements_remaining_C);
          
          // do the computation
          ComputeThreePointCorrelationCountsMultiCPU(kernel_results, runtime,
                                                     pointsA + startA, numA,
                                                     pointsB + startB, numB,
                                                     pointsC + startC, numC,
                                                     lower_bounds_sqr_ptr_,
                                                     upper_bounds_sqr_ptr_,
                                                     total_matchers_,
                                                     &satisfiability[0]);
          
          // have to add the results in here because the CPU kernel call
          // zeros out the array
          //result.result += kernel_results[3] / overcounting_factor;
          for (int i = 0; i < total_matchers_; i++)
          {
            this_result[i] += kernel_results[i][3];
          }
          
          // update counter
          elements_remaining_C -= numC;
          startC += numC;
          
        } // loop over count2
        
        elements_remaining_B -= numB;
        startB += numB;
        
      } // loop over count1
      
      elements_remaining_A -= numA;
      startA += numA;
      
    } // loop over count0
    
  } // if there are any points in C
  
  // we don't need to worry about overcounting with C because we've already
  // handled the cases where C == A or C == B
  int overcounting_factor = 1;
  if (nodeA == nodeB) {
    overcounting_factor = 2;
  }
  
  // process and store the results
  for (int i = 0; i < total_matchers_; i++)
  {
    result.results()[i] += this_result[i] / overcounting_factor;
  }
  
  // free memory
  for (int i = 0; i < total_matchers_; i++) {
    delete kernel_results[i];
  }
  delete kernel_results;
  delete this_result;
  delete[] pointsC;
  
}

void npoint_mlpack::EfficientMultiMatcher::AddResult(PartialResult& result)
{
  for (int i = 0; i < total_matchers_; i++)
  {
    results_[i] += result.results()[i];
  }
}


std::vector<long long int>& npoint_mlpack::EfficientMultiMatcher::results() {
  return results_;
}

std::vector<double>& npoint_mlpack::EfficientMultiMatcher::weighted_results() {
  return weighted_results_;
}


int npoint_mlpack::EfficientMultiMatcher::num_permutations() {
  return perms_.num_permutations();
}

int npoint_mlpack::EfficientMultiMatcher::matcher_ind() const
{
  return matcher_ind_;
}

void npoint_mlpack::EfficientMultiMatcher::set_matcher_ind(int new_ind)
{
  matcher_ind_ = new_ind;
}

std::vector<int>& npoint_mlpack::EfficientMultiMatcher::results_size()
{
  return results_size_;
}

double npoint_mlpack::EfficientMultiMatcher::min_dist_sq() const
{
  return sorted_lower_bounds_sq_.front();
}

double npoint_mlpack::EfficientMultiMatcher::max_dist_sq() const
{
  return sorted_upper_bounds_sq_.back();
}




