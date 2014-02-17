//
//  efficient_angle_matcher.cpp
//  contrib_march
//
//  Created by William March on 10/12/12.
//
//

#include "efficient_angle_matcher.hpp"

npoint_mlpack::EfficientAngleMatcher::PartialResult::PartialResult(std::vector<int>& sizes)
:
num_r1_(sizes[0]),
num_theta_(sizes[1]),
results_(boost::extents[num_r1_][num_theta_])
{
  std::fill(results_.origin(), results_.origin() + results_.size(), 0);
}

npoint_mlpack::EfficientAngleMatcher::PartialResult::~PartialResult() {}

void npoint_mlpack::EfficientAngleMatcher::PartialResult::Reset()
{
  std::fill(results_.origin(), results_.origin() + results_.size(), 0);
}

int npoint_mlpack::EfficientAngleMatcher::PartialResult::num_r1() const
{
  return num_r1_;
}

int npoint_mlpack::EfficientAngleMatcher::PartialResult::num_theta() const
{
  return num_theta_;
}

const boost::multi_array<long long int, 2>& npoint_mlpack::EfficientAngleMatcher::PartialResult::results() const
{
  return results_;
}

boost::multi_array<long long int, 2>& npoint_mlpack::EfficientAngleMatcher::PartialResult::results()
{
  return results_;
}

npoint_mlpack::EfficientAngleMatcher::PartialResult&
npoint_mlpack::EfficientAngleMatcher::PartialResult::operator=(const PartialResult& other)
{
  
  if (this != &other) {
    
    num_r1_ = other.num_r1();
    num_theta_ = other.num_theta();
    
    results_.resize(boost::extents[num_r1_][num_theta_]);
    results_ = other.results();
    
  }
  
  return *this;
  
}

npoint_mlpack::EfficientAngleMatcher::PartialResult&
npoint_mlpack::EfficientAngleMatcher::PartialResult::operator+=(const PartialResult& other)
{
  
  if (this != &other)
  {
    
    if (num_r1_ != other.num_r1() || num_theta_ != other.num_theta()) {
      mlpack::Log::Fatal << "Using operator+= on mismatched AngleMatcher PartialResults\n";
    }
    
    for (int i = 0; i < num_r1_; i++)
    {
      for (int j = 0; j < num_theta_; j++)
      {
        results_[i][j] += other.results()[i][j];
      }
    }
    
  }
  
  return *this;
  
}

const npoint_mlpack::EfficientAngleMatcher::PartialResult
npoint_mlpack::EfficientAngleMatcher::PartialResult::operator+(const PartialResult &other) const {
  PartialResult result = *this;
  result += other;
  return result;
}


npoint_mlpack::EfficientAngleMatcher::EfficientAngleMatcher(const std::vector<arma::mat*>& data_in,
                                          const std::vector<arma::colvec*>& weights_in,
                                          MatcherArguments& args)
:
data_mat_list_(data_in),
data_weights_list_(weights_in),
short_sides_(args.short_sides()),
long_side_multiplier_(args.long_side()),
bin_thickness_factor_(args.bin_size()),
long_sides_(short_sides_.size()),
thetas_(args.thetas()),
r3_sides_(boost::extents[short_sides_.size()][thetas_.size()]),
r1_lower_sqr_(short_sides_.size()),
r1_upper_sqr_(short_sides_.size()),
r2_lower_sqr_(short_sides_.size()),
r2_upper_sqr_(short_sides_.size()),
r3_lower_sqr_(boost::extents[short_sides_.size()][thetas_.size()]),
r3_upper_sqr_(boost::extents[short_sides_.size()][thetas_.size()]),
tuple_size_(3),
total_num_matchers_(short_sides_.size() * thetas_.size()),
num_base_cases_(0),
num_pairs_considered_(0),
matcher_ind_(-1),
results_size_(2),
results_(boost::extents[short_sides_.size()][thetas_.size()]),
weighted_results_(boost::extents[short_sides_.size()][thetas_.size()])
{
  
  // Defined in the efficient kernel code, just initializes the lookup table
  FillThreePointCorrelationLUT(satisfiability);
  
  lower_bounds_sqr_ptr_ = new double*[total_num_matchers_];
  upper_bounds_sqr_ptr_ = new double*[total_num_matchers_];
  
  for (int i = 0; i < total_num_matchers_; i++)
  {
    lower_bounds_sqr_ptr_[i] = new double[3];
    upper_bounds_sqr_ptr_[i] = new double[3];
  }
  
  std::fill(results_.origin(), results_.origin() + results_.size(), 0);
  std::fill(weighted_results_.origin(),
            weighted_results_.origin() + weighted_results_.size(), 0.0);
  
  double half_thickness = bin_thickness_factor_ / 2.0;
  
  int bounds_ptr_ind = 0;
  
  for (unsigned int i = 0; i < short_sides_.size(); i++) {
    
    long_sides_[i] = long_side_multiplier_ * short_sides_[i];
    
    r1_lower_sqr_[i] = ((1.0 - half_thickness) * short_sides_[i])
    * ((1.0 - half_thickness) * short_sides_[i]);
    r1_upper_sqr_[i] = ((1.0 + half_thickness) * short_sides_[i])
    * ((1.0 + half_thickness) * short_sides_[i]);
    
    r2_lower_sqr_[i] = ((1.0 - half_thickness) * long_sides_[i])
    * ((1.0 - half_thickness) * long_sides_[i]);
    r2_upper_sqr_[i] = ((1.0 + half_thickness) * long_sides_[i])
    * ((1.0 + half_thickness) * long_sides_[i]);
    
    for (unsigned int j = 0; j < thetas_.size(); j++) {
      
      r3_sides_[i][j] = ComputeR3_(short_sides_[i],
                                   long_sides_[i],
                                   thetas_[j]);
      
      r3_lower_sqr_[i][j] = ((1.0 - half_thickness) * r3_sides_[i][j])
      * ((1.0 - half_thickness) * r3_sides_[i][j]);
      r3_upper_sqr_[i][j] = ((1.0 + half_thickness) * r3_sides_[i][j])
      * ((1.0 + half_thickness) * r3_sides_[i][j]);
      
      lower_bounds_sqr_ptr_[bounds_ptr_ind][0] = r1_lower_sqr_[i];
      lower_bounds_sqr_ptr_[bounds_ptr_ind][1] = r2_lower_sqr_[i];
      lower_bounds_sqr_ptr_[bounds_ptr_ind][2] = r3_lower_sqr_[i][j];
      
      upper_bounds_sqr_ptr_[bounds_ptr_ind][0] = r1_upper_sqr_[i];
      upper_bounds_sqr_ptr_[bounds_ptr_ind][1] = r2_upper_sqr_[i];
      upper_bounds_sqr_ptr_[bounds_ptr_ind][2] = r3_upper_sqr_[i][j];
      
      bounds_ptr_ind++;
      
    } // for j
    
  } // for i
  
  results_size_[0] = short_sides_.size();
  results_size_[1] = thetas_.size();
  
  longest_possible_side_sqr_ = std::max(*std::max_element(r1_upper_sqr_.begin(), r1_upper_sqr_.end()),
                                        std::max(*std::max_element(r2_upper_sqr_.begin(), r2_upper_sqr_.end()),
                                                 *std::max_element(r3_upper_sqr_[short_sides_.size() - 1].begin(), r3_upper_sqr_[short_sides_.size() - 1].end())));
  
  shortest_possible_side_sqr_ = std::min(*std::min_element(r1_lower_sqr_.begin(), r1_lower_sqr_.end()),
                                         std::min(*std::min_element(r2_lower_sqr_.begin(), r2_lower_sqr_.end()),
                                                  *std::min_element(r3_lower_sqr_[0].begin(), r3_lower_sqr_[0].end())));
  
} // constructor

npoint_mlpack::EfficientAngleMatcher::~EfficientAngleMatcher()
{
  for (int i = 0; i < total_num_matchers_; i++)
  {
    delete lower_bounds_sqr_ptr_[i];
    delete upper_bounds_sqr_ptr_[i];
  }
  delete lower_bounds_sqr_ptr_;
  delete upper_bounds_sqr_ptr_;
}

// given two edges and the angle between them, compute the length of the
// third size
// TODO: keep things squared?
double npoint_mlpack::EfficientAngleMatcher::ComputeR3_(double r1, double r2, double theta) {
  
  double r3sqr = (r1 * r1) + (r2 * r2) - 2.0 * r1 * r2 * cos(theta);
  
  return sqrt(r3sqr);
  
}


// Naive base case for now
void npoint_mlpack::EfficientAngleMatcher::ComputeBaseCase(NodeTuple& nodes) {
  PartialResult this_result(results_size_);
  ComputeBaseCase(nodes, this_result);
  AddResult(this_result);
} // ComputeBaseCase


// IMPORTANT: this doesn't use the partial result either
// this means it's not thread safe!!!
void npoint_mlpack::EfficientAngleMatcher::ComputeBaseCase(NodeTuple& nodes,
                                                  PartialResult& result)
{
  num_base_cases_++;
  
  NptNode* nodeA = nodes.node_list(0);
  NptNode* nodeB = nodes.node_list(1);
  NptNode* nodeC = nodes.node_list(2);
  
  int numA = nodeA->Count();
  int numB = nodeB->Count();
  int numC = nodeC->Count();
  
  // now, I'm going to prune out base cases
  std::vector<std::vector<int> > valid_indices(short_sides_.size());
  int total_matchers = TestNodeTupleCarefully_(nodeA, nodeB, nodeC, valid_indices);
  
  // do we need to do any work?
  if (total_matchers <= 0) {
    return;
  }
  
  if (numA > 64 || numB > 64 || numC > 64)
  {
    mlpack::Log::Fatal << "Calling base case with large nodes in EfficientAngleMatcher.\n";
  }
  
  const double3* pointsA = (double3*)data_mat_list_[0]->colptr(nodeA->Begin());
  const double3* pointsB = (double3*)data_mat_list_[1]->colptr(nodeB->Begin());
  const double3* pointsC = (double3*)data_mat_list_[2]->colptr(nodeC->Begin());
  
  NptRuntimes runtime;
  
  //std::cout << "allocating results in base case\n";
  uint64_t** kernel_results = new uint64_t*[total_matchers];
  for (int i = 0; i < total_matchers; i++) {
    kernel_results[i] = new uint64_t[4];
  }
  
  double** lower_bounds = new double*[total_matchers];
  double** upper_bounds = new double*[total_matchers];
  int bounds_ind = 0;
  
  for (unsigned int r1_ind = 0; r1_ind < short_sides_.size(); r1_ind++)
  {
    for (unsigned int theta_ind = 0; theta_ind < valid_indices[r1_ind].size();
         theta_ind++)
    {
      lower_bounds[bounds_ind] = new double[3];
      upper_bounds[bounds_ind] = new double[3];
      
      lower_bounds[bounds_ind][0] = r1_lower_sqr_[r1_ind];
      upper_bounds[bounds_ind][0] = r1_upper_sqr_[r1_ind];
      lower_bounds[bounds_ind][1] = r2_lower_sqr_[r1_ind];
      upper_bounds[bounds_ind][1] = r2_upper_sqr_[r1_ind];
      lower_bounds[bounds_ind][2] = r3_lower_sqr_[r1_ind][valid_indices[r1_ind][theta_ind]];
      upper_bounds[bounds_ind][2] = r3_upper_sqr_[r1_ind][valid_indices[r1_ind][theta_ind]];
      
      bounds_ind++;
      
    }
  }
  
  
  ComputeThreePointCorrelationCountsMultiCPU(kernel_results, runtime,
                                             pointsA, numA,
                                             pointsB, numB,
                                             pointsC, numC,
                                             lower_bounds,
                                             upper_bounds,
                                             total_matchers,
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
  
  bounds_ind = 0;
  // process and store results
  for (unsigned int r1_ind = 0; r1_ind < short_sides_.size(); r1_ind++)
  {
    for (unsigned int theta_ind = 0; theta_ind < valid_indices[r1_ind].size();
         theta_ind++)
    {
      result.results()[r1_ind][valid_indices[r1_ind][theta_ind]] += kernel_results[bounds_ind][3] / overcounting_factor;
      bounds_ind++;
    }
  }
  
  for (int i = 0; i < total_matchers; i++) {
    delete kernel_results[i];
    delete lower_bounds[i];
    delete upper_bounds[i];
  }
  delete kernel_results;
  delete lower_bounds;
  delete upper_bounds;
  
}

// IMPORTANT: this doesn't really use the partial result
// i.e. it directly counts the tuples and stores the result in the matcher
// So, not thread safe!
// ALSO: for now, this goes through all of the matchers, which is a big waste
// need to think about how to prune using a list of C nodes
void npoint_mlpack::EfficientAngleMatcher::ComputeBaseCase(NptNode* nodeA,
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
  uint64_t** kernel_results = new uint64_t*[total_num_matchers_];
  for (int i = 0; i < total_num_matchers_; i++) {
    kernel_results[i] = new uint64_t[4];
  }
  
  uint64_t* this_result = new uint64_t[total_num_matchers_];
  memset(this_result, 0, total_num_matchers_ * sizeof(uint64_t));
  
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
          ComputeThreePointCorrelationCountsMultiCPU(kernel_results, runtime,
                                                pointsA + startA, numA,
                                                pointsB + startB, numB,
                                                pointsC + startC, numC,
                                                lower_bounds_sqr_ptr_,
                                                upper_bounds_sqr_ptr_,
                                                total_num_matchers_,
                                                &satisfiability[0]);
          
          // have to add the results in here because the CPU kernel call
          // zeros out the array
          //result.result += kernel_results[3] / overcounting_factor;
          for (int i = 0; i < total_num_matchers_; i++)
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
  int results_ptr_ind = 0;
  for (unsigned int r1_ind = 0; r1_ind < short_sides_.size(); r1_ind++)
  {
    for (unsigned int theta_ind = 0; theta_ind < thetas_.size(); theta_ind++)
    {
      result.results()[r1_ind][theta_ind] += this_result[results_ptr_ind] / overcounting_factor;
      results_ptr_ind++;
    }
  }

  // free memory
  for (int i = 0; i < total_num_matchers_; i++) {
    delete kernel_results[i];
  }
  delete kernel_results;
  delete this_result;
  delete[] pointsC;
  
} // base case (over list of nodes)




// returns the index of the value of r1 that is satisfied by the tuple
// the list contains the indices of thetas_ that are satisfied by the tuple
// assumes that valid_theta_indices is initialized and empty
// Important: it is possible to have a tuple satisfy more than one matcher
// Return -1 if there is no matcher satisfied by this pair
int npoint_mlpack::EfficientAngleMatcher::TestNodeTupleCarefully_(NptNode* node1,
                                                                  NptNode* node2,
                                                                  NptNode* node3,
                                                                   std::vector<std::vector<int> >& valid_indices)
{
  
  /*
  double d12_lower_sqr = node1->Bound().MinDistance(node2->Bound());
  double d13_lower_sqr = node1->Bound().MinDistance(node3->Bound());
  double d23_lower_sqr = node2->Bound().MinDistance(node3->Bound());
  
  double d12_upper_sqr = node1->Bound().MaxDistance(node2->Bound());
  double d13_upper_sqr = node1->Bound().MaxDistance(node3->Bound());
  double d23_upper_sqr = node2->Bound().MaxDistance(node3->Bound());
  */
  
  int total_matchers = 0;
  
  // this won't prune any possible base cases
  for (unsigned int i = 0; i < short_sides_.size(); i++)
  {
    for (unsigned int j = 0; j < thetas_.size(); j++)
    {
      valid_indices[i].push_back(j);
      
      total_matchers++;
    }
  }
  
  return total_matchers;
  
} // TestNodeTupleCarefully

bool npoint_mlpack::EfficientAngleMatcher::TestNodeTuple(NodeTuple& nodes) {
  
  // pruning options: all three distances are shorter than the shortest r1
  // or longer than the longest one
  mlpack::bound::HRectBound<2, false>& box1 = nodes.node_list(0)->Bound();
  mlpack::bound::HRectBound<2, false>& box2 = nodes.node_list(1)->Bound();
  mlpack::bound::HRectBound<2, false>& box3 = nodes.node_list(2)->Bound();
  
  double d12_lower = box1.MinDistance(box2);
  double d12_upper = box1.MaxDistance(box2);
  
  double d13_lower = box1.MinDistance(box3);
  double d13_upper = box1.MaxDistance(box3);
  
  double d23_lower = box2.MinDistance(box3);
  double d23_upper = box2.MaxDistance(box3);
  
  std::vector<double> min_dists(3);
  min_dists[0] = d12_lower;
  min_dists[1] = d13_lower;
  min_dists[2] = d23_lower;
  
  std::vector<double> max_dists(3);
  max_dists[0] = d12_upper;
  max_dists[1] = d13_upper;
  max_dists[2] = d23_upper;
  
  return TestNodeTuple(min_dists, max_dists);
  
} // TestNodeTuple

bool npoint_mlpack::EfficientAngleMatcher::
TestNodeTuple(std::vector<double>& min_dists_sqr,
              std::vector<double>& max_dists_sqr)
{
  
  bool possibly_valid = true;
  
  double min_dist = *std::min_element(min_dists_sqr.begin(),
                                      min_dists_sqr.end());
  
  double max_dist = *std::max_element(max_dists_sqr.begin(),
                                      max_dists_sqr.end());
  
  if (min_dist > longest_possible_side_sqr_
      || max_dist < shortest_possible_side_sqr_) {
    possibly_valid = false;
  }
  
  return possibly_valid;
  
}

void npoint_mlpack::EfficientAngleMatcher::AddResult(PartialResult& result)
{
  
  for (unsigned int i = 0; i < short_sides_.size(); i++)
  {
    for (unsigned int j = 0; j < thetas_.size(); j++)
    {
      results_[i][j] += result.results()[i][j];
    }
  }
  
}



void npoint_mlpack::EfficientAngleMatcher::OutputResults() {
  
  for (size_t j = 0; j < results_.size(); j++) {
    
    for (size_t k = 0; k < results_[j].size(); k++) {
      
      mlpack::Log::Info << "Matcher: ";
      mlpack::Log::Info << "R1: " << short_sides_[j] << ", ";
      mlpack::Log::Info << "R2: " << (short_sides_[j] * long_side_multiplier_) << ", ";
      mlpack::Log::Info << "theta: " << thetas_[k] << ": ";
      
      mlpack::Log::Info << results_[j][k] << "\n";
      
    } // for k
    
  } // for j
  
  mlpack::Log::Info << "\n\n";
  
} // OutputResults


int npoint_mlpack::EfficientAngleMatcher::tuple_size() const {
  return tuple_size_;
}

boost::multi_array<long long int, 2>& npoint_mlpack::EfficientAngleMatcher::results() {
  return results_;
}

boost::multi_array<double, 2>& npoint_mlpack::EfficientAngleMatcher::weighted_results() {
  return weighted_results_;
}

double npoint_mlpack::EfficientAngleMatcher::min_dist_sq() const
{
  return shortest_possible_side_sqr_;
}

double npoint_mlpack::EfficientAngleMatcher::max_dist_sq() const
{
  return longest_possible_side_sqr_;
}

int npoint_mlpack::EfficientAngleMatcher::num_base_cases() const
{
  return num_base_cases_;
}

long long int npoint_mlpack::EfficientAngleMatcher::num_pairs_considered() const
{
  return num_pairs_considered_;
}

int npoint_mlpack::EfficientAngleMatcher::matcher_ind() const
{
  return matcher_ind_;
}

void npoint_mlpack::EfficientAngleMatcher::set_matcher_ind(int new_ind)
{
  matcher_ind_ = new_ind;
}

std::vector<int>& npoint_mlpack::EfficientAngleMatcher::results_size() 
{
  return results_size_;
}