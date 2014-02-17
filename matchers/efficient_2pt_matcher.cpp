//
//  efficient_2pt_matcher.cpp
//
//
//

#include "efficient_2pt_matcher.hpp"

npoint_mlpack::Efficient2ptMatcher::PartialResult::PartialResult(std::vector<int>& /*results_size*/)
:
results_(0)
{}

npoint_mlpack::Efficient2ptMatcher::PartialResult::~PartialResult() {}


void npoint_mlpack::Efficient2ptMatcher::PartialResult::Reset()
{
  results_ = 0;
}

long long int npoint_mlpack::Efficient2ptMatcher::PartialResult::results() const
{
  return results_;
}

npoint_mlpack::Efficient2ptMatcher::PartialResult&
npoint_mlpack::Efficient2ptMatcher::PartialResult::operator=(const PartialResult& other)
{
  
  if (this != &other)
  {
    results_ = other.results();
  }
  
  return *this;
  
}

npoint_mlpack::Efficient2ptMatcher::PartialResult&
npoint_mlpack::Efficient2ptMatcher::PartialResult::operator+=(const PartialResult& other)
{
  
  if (this != &other)
  {
    results_ += other.results();
  }
  
  return *this;
  
}

const npoint_mlpack::Efficient2ptMatcher::PartialResult
npoint_mlpack::Efficient2ptMatcher::PartialResult::operator+(const PartialResult &other) const
{
  PartialResult result = *this;
  result += other;
  return result;
}

void npoint_mlpack::Efficient2ptMatcher::PartialResult::AddResult(long long int result)
{
  results_ += result;
}

npoint_mlpack::Efficient2ptMatcher::Efficient2ptMatcher(const std::vector<arma::mat*>& data_in,
                                                        const std::vector<arma::colvec*>& weights_in,
                                                        const MatcherArguments& matcher_args)
:
data_mat_list_(data_in),
data_weights_list_(weights_in),
lower_bounds_sqr_(matcher_args.lower_matcher() % matcher_args.lower_matcher()),
upper_bounds_sqr_(matcher_args.upper_matcher() % matcher_args.upper_matcher()),
results_(0),
weighted_results_(0.0),
// For now, the kernels only work for 3pt
tuple_size_(2),
num_base_cases_(0),
matcher_ind_(-1)
{
  
  // Fill in the arrays for the kernels
  lower_bounds_sqr_ptr_[0] = lower_bounds_sqr_(0, 1);
  upper_bounds_sqr_ptr_[0] = upper_bounds_sqr_(0, 1);
  
} // Constructor


npoint_mlpack::Efficient2ptMatcher::Efficient2ptMatcher(const Efficient2ptMatcher& other,
                                                        bool is_copy)
:
data_mat_list_(other.data_mat_list()),
data_weights_list_(other.data_weights_list()),
lower_bounds_sqr_(other.lower_bounds_sqr()),
upper_bounds_sqr_(other.upper_bounds_sqr()),
tuple_size_(other.tuple_size()),
num_base_cases_(other.num_base_cases()),
matcher_ind_(-1)
{
  
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
  upper_bounds_sqr_ptr_[0] = upper_bounds_sqr_(0, 1);
  
} // copy constructor

npoint_mlpack::Efficient2ptMatcher::~Efficient2ptMatcher()
{
  
}

void npoint_mlpack::Efficient2ptMatcher::SumResults(const Efficient2ptMatcher& left_matcher,
                                                    const Efficient2ptMatcher& right_matcher)
{
  
  // IMPORTANT: I don't think this needs to be +=, but I'm not sure
  results_ = left_matcher.results() + right_matcher.results();
  weighted_results_ = left_matcher.weighted_results()
  + right_matcher.weighted_results();
  
}


bool npoint_mlpack::Efficient2ptMatcher::TestNodeTuple(std::vector<double>& min_dists_sqr_,
                                                       std::vector<double>& max_dists_sqr_)
{
  
  bool can_satisfy = true;
  
  double d01_min_sq = min_dists_sqr_[0];
  double d01_max_sq = max_dists_sqr_[0];
  
  can_satisfy = !(d01_min_sq > upper_bounds_sqr_ptr_[0]
                  || d01_max_sq < lower_bounds_sqr_ptr_[0]);
  
  return can_satisfy;
  
}

bool npoint_mlpack::Efficient2ptMatcher::TestNodeTuple(NodeTuple& nodes)
{
  
  bool can_satisfy;
  
  NptNode* node0 = nodes.node_list(0);
  NptNode* node1 = nodes.node_list(1);
  
  double d01_max_sq = node0->Bound().MaxDistance(node1->Bound());
  double d01_min_sq = node0->Bound().MinDistance(node1->Bound());
  
  std::vector<double> min_dists_sq(1);
  std::vector<double> max_dists_sq(1);
  
  min_dists_sq[0] = d01_min_sq;
  max_dists_sq[0] = d01_max_sq;
  
  can_satisfy = TestNodeTuple(min_dists_sq, max_dists_sq);
  
  return can_satisfy;
  
} // TestNodeTuple

void npoint_mlpack::Efficient2ptMatcher::ComputeBaseCase(NptNode* nodeA,
                                                         NptNode* nodeB,
                                                         std::vector<NptNode*>& nodeC_list)
{
  
  mlpack::Log::Fatal << "Using pairwise traversal with a 2pt matcher.\n";
  
}

void npoint_mlpack::Efficient2ptMatcher::AddResult(PartialResult& result)
{
  
  results_ += result.results();
  
}

void npoint_mlpack::Efficient2ptMatcher::ComputeBaseCase(NptNode* nodeA,
                                                         NptNode* nodeB,
                                                         std::vector<NptNode*>& nodeC_list,
                                                         PartialResult& result)
{
  
  mlpack::Log::Fatal << "Using pairwise traversal with a 2pt matcher.\n";
  
}

void npoint_mlpack::Efficient2ptMatcher::ComputeBaseCase(NodeTuple& nodes)
{
  
  PartialResult result(results_size_);
  ComputeBaseCase(nodes, result);
  results_ += result.results();
  
}


void npoint_mlpack::Efficient2ptMatcher::ComputeBaseCase(NodeTuple& nodes,
                                                         PartialResult& result)
{
  
  int count0 = nodes.node_list(0)->Count();
  int count1 = nodes.node_list(1)->Count();
  
  if (count0 > 0 && count1 > 0) {
    
    int begin0 = nodes.node_list(0)->Begin();
    int begin1 = nodes.node_list(1)->Begin();
    
    // This gets initialized in the function below
    uint64_t kernel_results[1];
    
    uint64_t this_result = 0;
    
    NptRuntimes runtime;
    
    double3* points0 = (double3*)data_mat_list_[0]->colptr(begin0);
    double3* points1 = (double3*)data_mat_list_[1]->colptr(begin1);
    
    // Need to handle the overcounting we may do
    // This assumes that nodes are either identical or don't overlap
    int num_same_nodes = 0;
    // not sure if this works
    if ((data_mat_list_[0] == data_mat_list_[1])
        && (begin0 == begin1)) {
      num_same_nodes++;
    }
    
    int overcounting_factor;
    if (num_same_nodes == 0) {
      overcounting_factor = 1;
    }
    else if (num_same_nodes == 1) {
      overcounting_factor = 2;
    }
    
    int elements_remaining_0 = count0;
    int start0 = 0;
    while (elements_remaining_0 > 0) {
      
      int num0 = std::min(64, elements_remaining_0);
      
      int elements_remaining_1 = count1;
      int start1 = 0;
      while (elements_remaining_1 > 0) {
        
        int num1 = std::min(64, elements_remaining_1);
        
                  
        ComputeTwoPointCorrelationCountsCPU(kernel_results, runtime,
                                              points0 + start0, num0,
                                              points1 + start1, num1,
                                              lower_bounds_sqr_ptr_,
                                              upper_bounds_sqr_ptr_,
                                              &satisfiability[0]);
      
        // have to add the results in here because the CPU kernel call
        // zeros out the array
        //result.result += kernel_results[3] / overcounting_factor;
        this_result += kernel_results[0];
        
        elements_remaining_1 -= num1;
        start1 += num1;
        
      } // loop over count1
      
      elements_remaining_0 -= num0;
      start0 += num0;
      
    } // loop over count0
    
    //std::cout << "found " << this_result / overcounting_factor << " tuples.\n";
    //result.result += this_result / overcounting_factor;
    result.AddResult(this_result / overcounting_factor);
    
  } // if there is work to do
  
} // BaseCase

// returns the minimum bound distance in the matcher
double npoint_mlpack::Efficient2ptMatcher::min_dist_sq() const {
  
  return lower_bounds_sqr_ptr_[0];
  
}

double npoint_mlpack::Efficient2ptMatcher::max_dist_sq() const {

  return upper_bounds_sqr_ptr_[0];
  
}


void npoint_mlpack::Efficient2ptMatcher::OutputResults()
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

long long int npoint_mlpack::Efficient2ptMatcher::results() const {
  return results_;
}

double  npoint_mlpack::Efficient2ptMatcher::weighted_results() const {
  return weighted_results_;
}

int npoint_mlpack::Efficient2ptMatcher::tuple_size() const {
  return tuple_size_;
}

const std::vector<arma::mat*>& npoint_mlpack::Efficient2ptMatcher::data_mat_list() const
{
  return data_mat_list_;
}

const std::vector<arma::colvec*>& npoint_mlpack::Efficient2ptMatcher::data_weights_list() const
{
  return data_weights_list_;
}
const arma::mat& npoint_mlpack::Efficient2ptMatcher::lower_bounds_sqr() const {
  return lower_bounds_sqr_;
}

const arma::mat& npoint_mlpack::Efficient2ptMatcher::upper_bounds_sqr() const {
  return upper_bounds_sqr_;
}

long long int npoint_mlpack::Efficient2ptMatcher::num_base_cases() const {
  return num_base_cases_;
}

int npoint_mlpack::Efficient2ptMatcher::matcher_ind() const
{
  return matcher_ind_;
}

void npoint_mlpack::Efficient2ptMatcher::set_matcher_ind(int new_ind)
{  
  matcher_ind_ = new_ind;
}

std::vector<int>& npoint_mlpack::Efficient2ptMatcher::results_size()
{
  return results_size_;
}

