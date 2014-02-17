//
//  efficient_cpu_matcher.cpp
//  
//
//  Created by William March on 3/9/12.
//  Copyright (c) 2012 Georgia Institute of Technology. All rights reserved.
//

#include "efficient_cpu_matcher.hpp"

npoint_mlpack::EfficientCpuMatcher::PartialResult::PartialResult(std::vector<int>& /*results_size*/)
:
results_(0)
{}

npoint_mlpack::EfficientCpuMatcher::PartialResult::~PartialResult() {}


void npoint_mlpack::EfficientCpuMatcher::PartialResult::Reset()
{
  results_ = 0;
}

long long int npoint_mlpack::EfficientCpuMatcher::PartialResult::results() const
{
  return results_;
}

npoint_mlpack::EfficientCpuMatcher::PartialResult&
npoint_mlpack::EfficientCpuMatcher::PartialResult::operator=(const PartialResult& other)
{
  
  if (this != &other)
  {
    results_ = other.results();
  }
  
  return *this;
  
}

npoint_mlpack::EfficientCpuMatcher::PartialResult&
npoint_mlpack::EfficientCpuMatcher::PartialResult::operator+=(const PartialResult& other)
{
  
  if (this != &other)
  {
    results_ += other.results();
  }
  
  return *this;
  
}

const npoint_mlpack::EfficientCpuMatcher::PartialResult
npoint_mlpack::EfficientCpuMatcher::PartialResult::operator+(const PartialResult &other) const
{
  PartialResult result = *this;
  result += other;
  return result;
}

void npoint_mlpack::EfficientCpuMatcher::PartialResult::AddResult(long long int result)
{
  results_ += result;
}

npoint_mlpack::EfficientCpuMatcher::EfficientCpuMatcher(const std::vector<arma::mat*>& data_in,
                                              const std::vector<arma::colvec*>& weights_in,
                                              const MatcherArguments& matcher_args)
:
data_mat_list_(data_in),
data_weights_list_(weights_in),
perms_(matcher_args.lower_matcher().n_cols),
lower_bounds_sqr_(matcher_args.lower_matcher() % matcher_args.lower_matcher()),
upper_bounds_sqr_(matcher_args.upper_matcher() % matcher_args.upper_matcher()),
results_(0),
weighted_results_(0.0),
// For now, the kernels only work for 3pt
tuple_size_(3),
num_base_cases_(0),
num_permutations_(perms_.num_permutations()),
matcher_ind_(-1)
{
  
  // Defined in the efficient kernel code, just initializes the lookup table
  FillThreePointCorrelationLUT(satisfiability);
  
  // These are the valid combinations
  // We represent a triple of points with 9 bits.  The first three bits 
  // indicate if the first pair of points satisfy each of the three distance
  // constraints.  The second and third pairs correspond to the second and 
  // third sets of bits.  A tuple is valid if and only if it ANDed with one 
  // of these masks is non-zero.
  valid_masks_[0] = 0x111; // 100 010 001
  valid_masks_[1] = 0x10A; // 100 001 010
  valid_masks_[2] = 0x0A1; // 010 100 001
  valid_masks_[3] = 0x08C; // 010 001 100
  valid_masks_[4] = 0x062; // 001 100 010
  valid_masks_[5] = 0x054; // 001 010 100
  
  // Fill in the arrays for the kernels
  lower_bounds_sqr_ptr_[0] = lower_bounds_sqr_(0, 1);
  lower_bounds_sqr_ptr_[1] = lower_bounds_sqr_(0, 2);
  lower_bounds_sqr_ptr_[2] = lower_bounds_sqr_(1, 2);
  
  upper_bounds_sqr_ptr_[0] = upper_bounds_sqr_(0, 1);
  upper_bounds_sqr_ptr_[1] = upper_bounds_sqr_(0, 2);
  upper_bounds_sqr_ptr_[2] = upper_bounds_sqr_(1, 2);
  
} // Constructor


npoint_mlpack::EfficientCpuMatcher::EfficientCpuMatcher(const EfficientCpuMatcher& other,
                                              bool is_copy)
:
data_mat_list_(other.data_mat_list()),
data_weights_list_(other.data_weights_list()),
perms_(other.lower_bounds_sqr().n_cols),
lower_bounds_sqr_(other.lower_bounds_sqr()),
upper_bounds_sqr_(other.upper_bounds_sqr()),
tuple_size_(other.tuple_size()),
num_base_cases_(other.num_base_cases()),
num_permutations_(perms_.num_permutations()),
matcher_ind_(-1)
{
  
  // Initialize the lookup table
  FillThreePointCorrelationLUT(satisfiability);
  
  if (is_copy) 
  {
    results_ = other.results();
    weighted_results_ = other.weighted_results();
  }
  else 
  {
    results_ = 0;
    weighted_results_ = 0.0;
  }
  // Fill in the arrays for the kernels
  lower_bounds_sqr_ptr_[0] = lower_bounds_sqr_(0, 1);
  lower_bounds_sqr_ptr_[1] = lower_bounds_sqr_(0, 2);
  lower_bounds_sqr_ptr_[2] = lower_bounds_sqr_(1, 2);
  
  upper_bounds_sqr_ptr_[0] = upper_bounds_sqr_(0, 1);
  upper_bounds_sqr_ptr_[1] = upper_bounds_sqr_(0, 2);
  upper_bounds_sqr_ptr_[2] = upper_bounds_sqr_(1, 2);
  
} // copy constructor

npoint_mlpack::EfficientCpuMatcher::~EfficientCpuMatcher()
{
  
}

void npoint_mlpack::EfficientCpuMatcher::SumResults(const EfficientCpuMatcher& left_matcher,
                                          const EfficientCpuMatcher& right_matcher)
{
  
  // IMPORTANT: I don't think this needs to be +=, but I'm not sure
  results_ = left_matcher.results() + right_matcher.results();
  weighted_results_ = left_matcher.weighted_results() 
                      + right_matcher.weighted_results();
  
} 


bool npoint_mlpack::EfficientCpuMatcher::TestNodeTuple(std::vector<double>& min_dists_sqr_,
                   std::vector<double>& max_dists_sqr_)
{
  
  bool can_satisfy;
  
  double d01_min_sq = min_dists_sqr_[0];
  double d01_max_sq = max_dists_sqr_[0];
  
  double d02_min_sq = min_dists_sqr_[1];
  double d02_max_sq = max_dists_sqr_[1];

  double d12_min_sq = min_dists_sqr_[2];
  double d12_max_sq = max_dists_sqr_[2];

  unsigned char sat01 = 0;
  if (!(d01_min_sq > upper_bounds_sqr_ptr_[0] 
        || d01_max_sq < lower_bounds_sqr_ptr_[0])) 
    sat01 |= 0x1;

  if (!(d01_min_sq > upper_bounds_sqr_ptr_[1] 
        || d01_max_sq < lower_bounds_sqr_ptr_[1])) 
    sat01 |= 0x2;
  
  if (!(d01_min_sq > upper_bounds_sqr_ptr_[2] 
        || d01_max_sq < lower_bounds_sqr_ptr_[2])) 
    sat01 |= 0x4;

  
  if (sat01) 
  {
    
    unsigned char sat02 = 0;
    // if dist_0q range overlaps the matcher range at all, then the matcher 
    // may be satisfied
    if (!(d02_min_sq > upper_bounds_sqr_ptr_[0] 
          || d02_max_sq < lower_bounds_sqr_ptr_[0])) {
      sat02 |= 0x1;
    }
    if (!(d02_min_sq > upper_bounds_sqr_ptr_[1] 
          || d02_max_sq < lower_bounds_sqr_ptr_[1])) {
      sat02 |= 0x2;
    }
    if (!(d02_min_sq > upper_bounds_sqr_ptr_[2] 
          || d02_max_sq < lower_bounds_sqr_ptr_[2])) {
      sat02 |= 0x4;
    }
    
    unsigned char sat12 = 0;
    // if dist_01 range overlaps the matcher range at all, then the matcher 
    // may be satisfied
    if (!(d12_min_sq > upper_bounds_sqr_ptr_[0] 
          || d12_max_sq < lower_bounds_sqr_ptr_[0])) {
      sat12 |= 0x1;
    }
    if (!(d12_min_sq > upper_bounds_sqr_ptr_[1] 
          || d12_max_sq < lower_bounds_sqr_ptr_[1])) {
      sat12 |= 0x2;
    }
    if (!(d12_min_sq > upper_bounds_sqr_ptr_[2] 
          || d12_max_sq < lower_bounds_sqr_ptr_[2])) {
      sat12 |= 0x4;
    }
    
    // now, check the three satisfiability things
    unsigned short sat_bits = (sat01 << 6) | (sat02 << 3) | sat12;
    
    
    can_satisfy = ((sat_bits & valid_masks_[0]) == valid_masks_[0])
    || ((sat_bits & valid_masks_[1]) == valid_masks_[1])
    || ((sat_bits & valid_masks_[2]) == valid_masks_[2])
    || ((sat_bits & valid_masks_[3]) == valid_masks_[3])
    || ((sat_bits & valid_masks_[4]) == valid_masks_[4])
    || ((sat_bits & valid_masks_[5]) == valid_masks_[5]);
    
  }
  // if the 01 pair can't satisfy any part of the matcher, then we can prune
  else {
    can_satisfy = false;
  }
  
  return can_satisfy;

}

bool npoint_mlpack::EfficientCpuMatcher::TestNodeTuple(NodeTuple& nodes) 
{
  
  bool can_satisfy;
  
  NptNode* node0 = nodes.node_list(0);
  NptNode* node1 = nodes.node_list(1);
  NptNode* node2 = nodes.node_list(2);

  double d01_max_sq = node0->Bound().MaxDistance(node1->Bound());
  double d01_min_sq = node0->Bound().MinDistance(node1->Bound());

  double d02_max_sq = node0->Bound().MaxDistance(node2->Bound());
  double d02_min_sq = node0->Bound().MinDistance(node2->Bound());

  double d12_max_sq = node1->Bound().MaxDistance(node2->Bound());
  double d12_min_sq = node1->Bound().MinDistance(node2->Bound());
  
  std::vector<double> min_dists_sq(3);
  std::vector<double> max_dists_sq(3);
  
  min_dists_sq[0] = d01_min_sq;
  min_dists_sq[1] = d02_min_sq;
  min_dists_sq[2] = d12_min_sq;
  
  max_dists_sq[0] = d01_max_sq;
  max_dists_sq[1] = d02_max_sq;
  max_dists_sq[2] = d12_max_sq;
  
  can_satisfy = TestNodeTuple(min_dists_sq, max_dists_sq);
  
  return can_satisfy;
  
} // TestNodeTuple

void npoint_mlpack::EfficientCpuMatcher::ComputeBaseCase(NptNode* nodeA,
                                               NptNode* nodeB,
                                               std::vector<NptNode*>& nodeC_list)
{
  
  PartialResult result(results_size_);
  
  ComputeBaseCase(nodeA, nodeB, nodeC_list, result);
  
  results_ += result.results();
  
}

void npoint_mlpack::EfficientCpuMatcher::AddResult(PartialResult& result)
{
  
  results_ += result.results();
  
}

void npoint_mlpack::EfficientCpuMatcher::ComputeBaseCase(NptNode* nodeA,
                                               NptNode* nodeB,
                                               std::vector<NptNode*>& nodeC_list,
                                               PartialResult& result)
{

  // IMPORTANT: several possibilities for overcounting here
  // A == B,
  // A == C for some (1) C
  // B == C for some (1) C
  // A == B == C for some (1) C
  
  //mlpack::Timer::Start("pairwise_efficient_base_case");
  
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
  uint64_t kernel_results[4];
  uint64_t this_result = 0;
  
  NptRuntimes runtime;
  
  // TODO: look at taking out these loops
  
  //std::cout << "Doing main base case on " << countC << " points in C.\n";

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
          ComputeThreePointCorrelationCountsCPU(kernel_results, runtime,
                                                pointsA + startA, numA,
                                                pointsB + startB, numB,
                                                pointsC + startC, numC,
                                                lower_bounds_sqr_ptr_,
                                                upper_bounds_sqr_ptr_,
                                                &satisfiability[0]);
          
          // have to add the results in here because the CPU kernel call
          // zeros out the array
          //result.result += kernel_results[3] / overcounting_factor;
          this_result += kernel_results[3];
          
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
  
  result.AddResult(this_result / overcounting_factor);
  
  delete[] pointsC;
  
  //mlpack::Timer::Stop("pairwise_efficient_base_case");
  
}

void npoint_mlpack::EfficientCpuMatcher::ComputeBaseCase(NodeTuple& nodes)
{
  
  PartialResult result(results_size_);
  ComputeBaseCase(nodes, result);
  results_ += result.results();
  
}


void npoint_mlpack::EfficientCpuMatcher::ComputeBaseCase(NodeTuple& nodes, 
                                               PartialResult& result) 
{
  
  //std::cout << "Computing base case.\n";
  
  //mlpack::Timer::Start("efficient_base_case");
  
  // This does change the result
 /*
  NptNode* nodeA = nodes.node_list(0);
  NptNode* nodeB = nodes.node_list(1);
  NptNode* nodeC = nodes.node_list(2);
  
  std::vector<NptNode*> node_list(1);
  node_list[0] = nodeC;
  
  ComputeBaseCase(nodeA, nodeB, node_list, result);
  
  return;
  */
  
  
  int count0 = nodes.node_list(0)->Count();
  int count1 = nodes.node_list(1)->Count();
  int count2 = nodes.node_list(2)->Count();
  
  //file << count0 << ", " << count1 << ", " << count2 << std::endl;
  //std::cout << "base case on " << count0 << ", " << count1 << ", " << count2 << "\n";
  
  if (count0 > 0 && count1 > 0 && count2 > 0) {
    
  //  if (0) {
    int begin0 = nodes.node_list(0)->Begin();
    int begin1 = nodes.node_list(1)->Begin();
    int begin2 = nodes.node_list(2)->Begin();
    
    //std::vector<unsigned char> satisfiability;
    
    //FillThreePointCorrelationLUT(satisfiability);
    
    // This gets initialized in the function below
    uint64_t kernel_results[4];
    
    uint64_t this_result = 0;
    
    NptRuntimes runtime;
    
    // TODO: take this stuff out, make it do this in the constructor
    // I think I can just get away with casting here, since this is what the memcpy does
    
    // fill the arrays
    //size_t num0_bytes = count0 * sizeof(double3);
    //size_t num1_bytes = count1 * sizeof(double3);
    //size_t num2_bytes = count2 * sizeof(double3);
    
    //double3* points0 = new double3[num0_bytes];
    //double3* points1 = new double3[num1_bytes];
    //double3* points2 = new double3[num2_bytes];
    
    
    //double* points0_in = data_mat_list_[0]->colptr(begin0);
    //double* points1_in = data_mat_list_[1]->colptr(begin1);
    //double* points2_in = data_mat_list_[2]->colptr(begin2);
    
    //std::memcpy(points0, points0_in, num0_bytes);
    //std::memcpy(points1, points1_in, num1_bytes);
    //std::memcpy(points2, points2_in, num2_bytes);
    
    double3* points0 = (double3*)data_mat_list_[0]->colptr(begin0);
    double3* points1 = (double3*)data_mat_list_[1]->colptr(begin1);
    double3* points2 = (double3*)data_mat_list_[2]->colptr(begin2);
    
    // Need to handle the overcounting we may do
    // This assumes that nodes are either identical or don't overlap
    int num_same_nodes = 0;
    // not sure if this works
    if ((data_mat_list_[0] == data_mat_list_[1])
        && (begin0 == begin1)) {
      num_same_nodes++;
    }
    if ((data_mat_list_[1] == data_mat_list_[2])
        && (begin1 == begin2)) {
      num_same_nodes++;
    }
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
    
    
    
    int elements_remaining_0 = count0;
    int start0 = 0;
    while (elements_remaining_0 > 0) {
      
      int num0 = std::min(64, elements_remaining_0);
      
      int elements_remaining_1 = count1;
      int start1 = 0;
      while (elements_remaining_1 > 0) {
        
        int num1 = std::min(64, elements_remaining_1);
        
        int elements_remaining_2 = count2;
        int start2 = 0;
        while (elements_remaining_2 > 0) {
          
          // update stuff for 2
          
          int num2 = std::min(64, elements_remaining_2);
          
          // do the computation
          
          ComputeThreePointCorrelationCountsCPU(kernel_results, runtime,
                                                points0 + start0, num0,
                                                points1 + start1, num1,
                                                points2 + start2, num2,
                                                lower_bounds_sqr_ptr_,
                                                upper_bounds_sqr_ptr_,
                                                &satisfiability[0]);
          
          // have to add the results in here because the CPU kernel call
          // zeros out the array
          //result.result += kernel_results[3] / overcounting_factor;
          this_result += kernel_results[3];
          
          // update counter
          elements_remaining_2 -= num2;
          start2 += num2;
          
        } // loop over count2
        
        elements_remaining_1 -= num1;
        start1 += num1;
        
      } // loop over count1
      
      elements_remaining_0 -= num0;
      start0 += num0;
      
    } // loop over count0
    
    //std::cout << "found " << this_result / overcounting_factor << " tuples.\n";
    //result.result += this_result / overcounting_factor;
    result.AddResult(this_result / overcounting_factor);
    
    //delete[] points0;
    //delete[] points1;
    //delete[] points2;
    
  } // if there is work to do 
  
  //mlpack::Timer::Stop("efficient_base_case");
  
} // BaseCase

// returns the minimum bound distance in the matcher
double npoint_mlpack::EfficientCpuMatcher::min_dist_sq() const {
  
  return std::min(lower_bounds_sqr_ptr_[0], std::min(lower_bounds_sqr_ptr_[1],
                                                     lower_bounds_sqr_ptr_[2]));
  
}

double npoint_mlpack::EfficientCpuMatcher::max_dist_sq() const {
  
  return std::max(upper_bounds_sqr_ptr_[0], std::max(upper_bounds_sqr_ptr_[1],
                                                     upper_bounds_sqr_ptr_[2]));
  
}


void npoint_mlpack::EfficientCpuMatcher::OutputResults()
{
  
  std::string d_string(tuple_size_, 'D');
  std::string r_string(tuple_size_, 'R');
  std::string label_string;
  label_string+=d_string;
  label_string+=r_string;
  
  for (int i = 0; i <= tuple_size_; i++) {
    
    // i is the number of random points in the tuple
    std::string this_string(label_string, i, tuple_size_);
    mlpack::Log::Info << this_string << ": ";
    
    mlpack::Log::Info << results_ << "\n";
    
    //mlpack::Log::Info << "\n\n";
    
  } // for i
  
} // OutputResults

long long int npoint_mlpack::EfficientCpuMatcher::results() const {
  return results_;
}

double  npoint_mlpack::EfficientCpuMatcher::weighted_results() const {
  return weighted_results_;
}

int npoint_mlpack::EfficientCpuMatcher::tuple_size() const {
  return tuple_size_;
}

const std::vector<arma::mat*>& npoint_mlpack::EfficientCpuMatcher::data_mat_list() const
{
  return data_mat_list_;
}

const std::vector<arma::colvec*>& npoint_mlpack::EfficientCpuMatcher::data_weights_list() const
{
  return data_weights_list_;
}
const arma::mat& npoint_mlpack::EfficientCpuMatcher::lower_bounds_sqr() const {
  return lower_bounds_sqr_;
}

const arma::mat& npoint_mlpack::EfficientCpuMatcher::upper_bounds_sqr() const {
  return upper_bounds_sqr_;
}

long long int npoint_mlpack::EfficientCpuMatcher::num_base_cases() const {
  return num_base_cases_;
}

int npoint_mlpack::EfficientCpuMatcher::matcher_ind() const
{  
  return matcher_ind_;
}

void npoint_mlpack::EfficientCpuMatcher::set_matcher_ind(int new_ind)
{  
  matcher_ind_ = new_ind;
}

std::vector<int>& npoint_mlpack::EfficientCpuMatcher::results_size()
{
  return results_size_;
}

