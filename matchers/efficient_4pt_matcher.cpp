//
//  efficient_2pt_matcher.cpp
//
//
//

#include "efficient_4pt_matcher.hpp"

npoint_mlpack::Efficient4ptMatcher::PartialResult::PartialResult(std::vector<int>& /*results_size*/)
:
results_(0)
{}

npoint_mlpack::Efficient4ptMatcher::PartialResult::~PartialResult() {}


void npoint_mlpack::Efficient4ptMatcher::PartialResult::Reset()
{
  results_ = 0;
}

long long int npoint_mlpack::Efficient4ptMatcher::PartialResult::results() const
{
  return results_;
}

npoint_mlpack::Efficient4ptMatcher::PartialResult&
npoint_mlpack::Efficient4ptMatcher::PartialResult::operator=(const PartialResult& other)
{
  
  if (this != &other)
  {
    results_ = other.results();
  }
  
  return *this;
  
}

npoint_mlpack::Efficient4ptMatcher::PartialResult&
npoint_mlpack::Efficient4ptMatcher::PartialResult::operator+=(const PartialResult& other)
{
  
  if (this != &other)
  {
    results_ += other.results();
  }
  
  return *this;
  
}

const npoint_mlpack::Efficient4ptMatcher::PartialResult
npoint_mlpack::Efficient4ptMatcher::PartialResult::operator+(const PartialResult &other) const
{
  PartialResult result = *this;
  result += other;
  return result;
}

void npoint_mlpack::Efficient4ptMatcher::PartialResult::AddResult(long long int result)
{
  results_ += result;
}

npoint_mlpack::Efficient4ptMatcher::Efficient4ptMatcher(const std::vector<arma::mat*>& data_in,
                                                        const std::vector<arma::colvec*>& weights_in,
                                                        const MatcherArguments& matcher_args)
:
data_mat_list_(data_in),
data_weights_list_(weights_in),
lower_bounds_sqr_(matcher_args.lower_matcher() % matcher_args.lower_matcher()),
upper_bounds_sqr_(matcher_args.upper_matcher() % matcher_args.upper_matcher()),
results_(0),
weighted_results_(0.0),
tuple_size_(4),
num_base_cases_(0),
matcher_ind_(-1),
node_sets_(4)
{
  
  // Fill in the arrays for the kernels
  lower_bounds_sqr_ptr_[0] = lower_bounds_sqr_(0, 1);
  lower_bounds_sqr_ptr_[1] = lower_bounds_sqr_(0, 2);
  lower_bounds_sqr_ptr_[2] = lower_bounds_sqr_(0, 3);
  lower_bounds_sqr_ptr_[3] = lower_bounds_sqr_(1, 2);
  lower_bounds_sqr_ptr_[4] = lower_bounds_sqr_(1, 3);
  lower_bounds_sqr_ptr_[5] = lower_bounds_sqr_(2, 3);

  upper_bounds_sqr_ptr_[0] = upper_bounds_sqr_(0, 1);
  upper_bounds_sqr_ptr_[1] = upper_bounds_sqr_(0, 2);
  upper_bounds_sqr_ptr_[2] = upper_bounds_sqr_(0, 3);
  upper_bounds_sqr_ptr_[3] = upper_bounds_sqr_(1, 2);
  upper_bounds_sqr_ptr_[4] = upper_bounds_sqr_(1, 3);
  upper_bounds_sqr_ptr_[5] = upper_bounds_sqr_(2, 3);
  
  min_matcher_dist_sq_ = *std::min_element(lower_bounds_sqr_ptr_, lower_bounds_sqr_ptr_ + 6);
  max_matcher_dist_sq_ = *std::max_element(upper_bounds_sqr_ptr_, upper_bounds_sqr_ptr_ + 6);
  
  node_sets_[0].push_back(0);
  if (data_mat_list_[1] == data_mat_list_[0]) {
    node_sets_[0].push_back(1);
  }
  else {
    node_sets_[1].push_back(1);
  }
  
  if (data_mat_list_[2] == data_mat_list_[0]) {
    node_sets_[0].push_back(2);
  }
  else if (data_mat_list_[2] == data_mat_list_[1]) {
    node_sets_[1].push_back(2);
  }
  else {
    node_sets_[2].push_back(2);
  }
  
  if (data_mat_list_[3] == data_mat_list_[0]) {
    node_sets_[0].push_back(3);
  }
  else if (data_mat_list_[3] == data_mat_list_[1]) {
    node_sets_[1].push_back(3);
  }
  else if (data_mat_list_[3] == data_mat_list_[2]) {
    node_sets_[2].push_back(3);
  }
  else {
    node_sets_[3].push_back(3);
  }
  
} // Constructor


npoint_mlpack::Efficient4ptMatcher::Efficient4ptMatcher(const Efficient4ptMatcher& other,
                                                        bool is_copy)
:
data_mat_list_(other.data_mat_list()),
data_weights_list_(other.data_weights_list()),
lower_bounds_sqr_(other.lower_bounds_sqr()),
upper_bounds_sqr_(other.upper_bounds_sqr()),
tuple_size_(other.tuple_size()),
num_base_cases_(other.num_base_cases()),
matcher_ind_(-1),
node_sets_(other.node_sets())
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
  lower_bounds_sqr_ptr_[1] = lower_bounds_sqr_(0, 2);
  lower_bounds_sqr_ptr_[2] = lower_bounds_sqr_(0, 3);
  lower_bounds_sqr_ptr_[3] = lower_bounds_sqr_(1, 2);
  lower_bounds_sqr_ptr_[4] = lower_bounds_sqr_(1, 3);
  lower_bounds_sqr_ptr_[5] = lower_bounds_sqr_(2, 4);
  
  upper_bounds_sqr_ptr_[0] = upper_bounds_sqr_(0, 1);
  upper_bounds_sqr_ptr_[1] = upper_bounds_sqr_(0, 2);
  upper_bounds_sqr_ptr_[2] = upper_bounds_sqr_(0, 3);
  upper_bounds_sqr_ptr_[3] = upper_bounds_sqr_(1, 2);
  upper_bounds_sqr_ptr_[4] = upper_bounds_sqr_(1, 3);
  upper_bounds_sqr_ptr_[5] = upper_bounds_sqr_(2, 3);
  
  min_matcher_dist_sq_ = *std::min_element(lower_bounds_sqr_ptr_, lower_bounds_sqr_ptr_ + 6);
  max_matcher_dist_sq_ = *std::max_element(upper_bounds_sqr_ptr_, upper_bounds_sqr_ptr_ + 6);

} // copy constructor

npoint_mlpack::Efficient4ptMatcher::~Efficient4ptMatcher()
{
  
}

void npoint_mlpack::Efficient4ptMatcher::SumResults(const Efficient4ptMatcher& left_matcher,
                                                    const Efficient4ptMatcher& right_matcher)
{
  
  // IMPORTANT: I don't think this needs to be +=, but I'm not sure
  results_ = left_matcher.results() + right_matcher.results();
  weighted_results_ = left_matcher.weighted_results()
  + right_matcher.weighted_results();
  
}


bool npoint_mlpack::Efficient4ptMatcher::TestNodeTuple(std::vector<double>& min_dists_sqr_,
                                                       std::vector<double>& max_dists_sqr_)
{
  
  bool can_satisfy = true;
  
  double smallest_dist_sq = *std::min_element(min_dists_sqr_.begin(), min_dists_sqr_.end());
  double largest_dist_sq = *std::max_element(max_dists_sqr_.begin(), max_dists_sqr_.end());

  can_satisfy = !(smallest_dist_sq > max_matcher_dist_sq_
                  || largest_dist_sq < min_matcher_dist_sq_);
  
  return can_satisfy;
  
}

bool npoint_mlpack::Efficient4ptMatcher::TestNodeTuple(NodeTuple& nodes)
{
  
  bool can_satisfy;
  
  NptNode* node0 = nodes.node_list(0);
  NptNode* node1 = nodes.node_list(1);
  NptNode* node2 = nodes.node_list(2);
  NptNode* node3 = nodes.node_list(3);
  
  double d01_max_sq = node0->Bound().MaxDistance(node1->Bound());
  double d01_min_sq = node0->Bound().MinDistance(node1->Bound());
  
  double d02_max_sq = node0->Bound().MaxDistance(node2->Bound());
  double d02_min_sq = node0->Bound().MinDistance(node2->Bound());
  
  double d03_max_sq = node0->Bound().MaxDistance(node3->Bound());
  double d03_min_sq = node0->Bound().MinDistance(node3->Bound());
  
  double d12_max_sq = node1->Bound().MaxDistance(node2->Bound());
  double d12_min_sq = node1->Bound().MinDistance(node2->Bound());
  
  double d13_max_sq = node1->Bound().MaxDistance(node3->Bound());
  double d13_min_sq = node1->Bound().MinDistance(node3->Bound());

  double d23_max_sq = node2->Bound().MaxDistance(node3->Bound());
  double d23_min_sq = node2->Bound().MinDistance(node3->Bound());

  std::vector<double> min_dists_sq(6);
  std::vector<double> max_dists_sq(6);
  
  min_dists_sq[0] = d01_min_sq;
  max_dists_sq[0] = d01_max_sq;

  min_dists_sq[1] = d02_min_sq;
  max_dists_sq[1] = d02_max_sq;

  min_dists_sq[2] = d03_min_sq;
  max_dists_sq[2] = d03_max_sq;

  min_dists_sq[3] = d12_min_sq;
  max_dists_sq[3] = d12_max_sq;

  min_dists_sq[4] = d13_min_sq;
  max_dists_sq[4] = d13_max_sq;

  min_dists_sq[5] = d23_min_sq;
  max_dists_sq[5] = d23_max_sq;

  can_satisfy = TestNodeTuple(min_dists_sq, max_dists_sq);
  
  return can_satisfy;
  
} // TestNodeTuple

void npoint_mlpack::Efficient4ptMatcher::ComputeBaseCase(NptNode* nodeA,
                                                         NptNode* nodeB,
                                                         std::vector<NptNode*>& nodeC_list)
{
  
  mlpack::Log::Fatal << "Using pairwise traversal with a 4pt matcher.\n";
  
}

void npoint_mlpack::Efficient4ptMatcher::AddResult(PartialResult& result)
{
  
  results_ += result.results();
  
}

void npoint_mlpack::Efficient4ptMatcher::ComputeBaseCase(NptNode* nodeA,
                                                         NptNode* nodeB,
                                                         std::vector<NptNode*>& nodeC_list,
                                                         PartialResult& result)
{
  
  mlpack::Log::Fatal << "Using pairwise traversal with a 4pt matcher.\n";
  
}

void npoint_mlpack::Efficient4ptMatcher::ComputeBaseCase(NodeTuple& nodes)
{
  
  PartialResult result(results_size_);
  ComputeBaseCase(nodes, result);
  results_ += result.results();
  
}

int npoint_mlpack::Efficient4ptMatcher::OvercountingFactor_(int begin0,
                                                            int begin1,
                                                            int begin2,
                                                            int begin3)
{
  
  std::vector<int> begins(4);
  begins[0] = begin0;
  begins[1] = begin1;
  begins[2] = begin2;
  begins[3] = begin3;
  
  int overcounting_factor = 1;
  
  // loop through entries in node_sets
  for (int i = 0; i < 4; i++) {
    
    int this_overcounting = 1;

    int this_num_same = 0;
    // going to assume the nodes are in order here
    for (int j = 0; j < (int)node_sets_[i].size() - 1; j++) {
      
      if (begins[node_sets_[i][j]] == begins[node_sets_[i][j+1]]) {
        this_num_same++;
      }
      
    } // loop over nodes in this set
    
    if (this_num_same == 1) {
      this_overcounting = 2;
    }
    else if (this_num_same == 2)
    {
      // need to check if there are two pairs the same
      // if its DDRR, then that will get caught above
      // so, we assume it's DDDD or RRRR
      if (i == 0 && node_sets_[i].size() == 4 &&
          (begins[node_sets_[i][0]] != begins[node_sets_[i][2]]
           && begins[node_sets_[i][1]] != begins[node_sets_[i][3]])) {
        this_overcounting = 4;
      }
      else {
        this_overcounting = 6;
      }
    }
    else if (this_num_same == 3) {
      this_overcounting = 24;
    }
    
    overcounting_factor *= this_overcounting;
    
  }
  
  //std::cout << "begins: " << begin0 << ", " << begin1 << ", " << begin2 << ", " << begin3 << "\n";
  
  return overcounting_factor;
  
} // OvercountingFactor


void npoint_mlpack::Efficient4ptMatcher::ComputeBaseCase(NodeTuple& nodes,
                                                         PartialResult& result)
{
  
  int count0 = nodes.node_list(0)->Count();
  int count1 = nodes.node_list(1)->Count();
  int count2 = nodes.node_list(2)->Count();
  int count3 = nodes.node_list(3)->Count();
  
  if (count0 > 0 && count1 > 0 && count2 > 0 && count3 > 0) {
    
    int begin0 = nodes.node_list(0)->Begin();
    int begin1 = nodes.node_list(1)->Begin();
    int begin2 = nodes.node_list(2)->Begin();
    int begin3 = nodes.node_list(3)->Begin();
    
    // This gets initialized in the function below
    uint64_t kernel_results[1];
    
    uint64_t this_result = 0;
    
    NptRuntimes runtime;
    
    double3* points0 = (double3*)data_mat_list_[0]->colptr(begin0);
    double3* points1 = (double3*)data_mat_list_[1]->colptr(begin1);
    double3* points2 = (double3*)data_mat_list_[2]->colptr(begin2);
    double3* points3 = (double3*)data_mat_list_[3]->colptr(begin3);
    
    // Need to handle the overcounting we may do
    // This assumes that nodes are either identical or don't overlap
    
    int overcounting_factor = OvercountingFactor_(begin0,
                                                  begin1,
                                                  begin2,
                                                  begin3);
    
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
          
          int num2 = std::min(64, elements_remaining_2);
          
          int elements_remaining_3 = count3;
          int start3 = 0;
          while (elements_remaining_3 > 0) {
            
            int num3 = std::min(64, elements_remaining_3);
            
            ComputeFourPointCorrelationCountsCPU(kernel_results, runtime,
                                                points0 + start0, num0,
                                                points1 + start1, num1,
                                                points2 + start2, num2,
                                                points3 + start3, num3,
                                                lower_bounds_sqr_ptr_,
                                                upper_bounds_sqr_ptr_,
                                                &satisfiability[0]);
        
            // have to add the results in here because the CPU kernel call
            // zeros out the array
            this_result += kernel_results[0];
     
            elements_remaining_3 -= num3;
            start3 += num3;
          } // loop over count3
          
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
    //std::cout << "this_result " << this_result << "\n";
    //std::cout << "overcounting: " << overcounting_factor << "\n\n";
    
    result.AddResult(this_result / overcounting_factor);
    //result.AddResult(this_result);
    
  } // if there is work to do
  
} // BaseCase

// returns the minimum bound distance in the matcher
double npoint_mlpack::Efficient4ptMatcher::min_dist_sq() const {
  
  return min_matcher_dist_sq_;
  
}

double npoint_mlpack::Efficient4ptMatcher::max_dist_sq() const {
  
  return max_matcher_dist_sq_;
  
}


void npoint_mlpack::Efficient4ptMatcher::OutputResults()
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

long long int npoint_mlpack::Efficient4ptMatcher::results() const {
  return results_;
}

double  npoint_mlpack::Efficient4ptMatcher::weighted_results() const {
  return weighted_results_;
}

int npoint_mlpack::Efficient4ptMatcher::tuple_size() const {
  return tuple_size_;
}

const std::vector<arma::mat*>& npoint_mlpack::Efficient4ptMatcher::data_mat_list() const
{
  return data_mat_list_;
}

const std::vector<arma::colvec*>& npoint_mlpack::Efficient4ptMatcher::data_weights_list() const
{
  return data_weights_list_;
}
const arma::mat& npoint_mlpack::Efficient4ptMatcher::lower_bounds_sqr() const {
  return lower_bounds_sqr_;
}

const arma::mat& npoint_mlpack::Efficient4ptMatcher::upper_bounds_sqr() const {
  return upper_bounds_sqr_;
}

long long int npoint_mlpack::Efficient4ptMatcher::num_base_cases() const {
  return num_base_cases_;
}

int npoint_mlpack::Efficient4ptMatcher::matcher_ind() const
{
  return matcher_ind_;
}

void npoint_mlpack::Efficient4ptMatcher::set_matcher_ind(int new_ind)
{
  matcher_ind_ = new_ind;
}

std::vector<int>& npoint_mlpack::Efficient4ptMatcher::results_size()
{
  return results_size_;
}

const std::vector<std::vector<int> >& npoint_mlpack::Efficient4ptMatcher::node_sets() const
{
  return node_sets_;
}

