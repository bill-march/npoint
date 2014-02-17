//
//  2_point_single_matcher.cpp
//  contrib_march
//
//  Created by William March on 6/25/13.
//
//

#include "2_point_single_matcher.hpp"


namespace npoint_mlpack {

  
  TwoPointSingleMatcher::PartialResult::PartialResult(const std::vector<int>& /*results_size*/)
  :
  results_(0)
  {}
  
  TwoPointSingleMatcher::PartialResult::~PartialResult() {}
  
  long long int TwoPointSingleMatcher::PartialResult::results() const
  {
    return results_;
  }
  
  void TwoPointSingleMatcher::PartialResult::Reset()
  {
    results_ = 0;
  }
  
  void TwoPointSingleMatcher::PartialResult::Increment()
  {
    results_++;
  }
  
  TwoPointSingleMatcher::PartialResult&
  TwoPointSingleMatcher::PartialResult::operator=(const PartialResult& other)
  {
    
    if (this != &other)
    {
      results_ = other.results();
    }
    
    return *this;
    
  }
  
  TwoPointSingleMatcher::PartialResult&
  TwoPointSingleMatcher::PartialResult::operator+=(const PartialResult& other)
  {
    
    if (this != &other)
    {
      results_ += other.results();
    }
    
    return *this;
    
  }
  
  const TwoPointSingleMatcher::PartialResult
  npoint_mlpack::TwoPointSingleMatcher::PartialResult::operator+(const PartialResult &other) const {
    PartialResult result = *this;
    result += other;
    return result;
  }

  
  
  TwoPointSingleMatcher::TwoPointSingleMatcher(const std::vector<arma::mat*>& data_in,
                      const std::vector<arma::colvec*>& weights_in,
                      const MatcherArguments& matcher_args)
  :
  data_mat_list_(data_in),
  data_weights_list_(weights_in),
  tuple_size_(2),
  perms_(tuple_size_),
  lower_bounds_sqr_(matcher_args.lower_matcher() % matcher_args.lower_matcher()),
  upper_bounds_sqr_(matcher_args.upper_matcher() % matcher_args.upper_matcher()),
  min_matcher_dist_sqr_(lower_bounds_sqr_(0,1)),
  max_matcher_dist_sqr_(upper_bounds_sqr_(0,1)),
  results_(0),
  weighted_results_(0.0),
  num_base_cases_(0),
  num_permutations_(1),
  num_pairs_considered_(0),
  // 1 because its a single matcher
  results_size_(1)
  {
    
  }

  // returns true if the tuple might satisfy the matcher
  bool TwoPointSingleMatcher::TestNodeTuple(NodeTuple& nodes)
  {
  
    double min_dist_sqr = nodes.node_list(0)->Bound().MinDistance(nodes.node_list(1)->Bound());
    double max_dist_sqr = nodes.node_list(0)->Bound().MaxDistance(nodes.node_list(1)->Bound());
    
    return (min_dist_sqr <= max_matcher_dist_sqr_
            && max_dist_sqr >= min_matcher_dist_sqr_);
    
  }

  // Ordering doesn't matter here, there's only one distance in each
  bool TwoPointSingleMatcher::TestNodeTuple(std::vector<double>& min_dists_sqr_,
                                            std::vector<double>& max_dists_sqr_)
  {
    
    return (min_dists_sqr_[0] <= max_matcher_dist_sqr_
            && max_dists_sqr_[0] >= min_matcher_dist_sqr_);
    
  }

  void TwoPointSingleMatcher::ComputeBaseCase(NodeTuple& nodes)
  {
  
    PartialResult result(results_size_);
    ComputeBaseCase(nodes, result);
    results_ += result.results();
    
  }

  void TwoPointSingleMatcher::ComputeBaseCase(NodeTuple& nodes, PartialResult& result)
  {
   
    num_base_cases_++;
    
    NptNode* node_i = nodes.node_list(0);
    NptNode* node_j = nodes.node_list(1);
    
    size_t begin_i = node_i->Begin();
    size_t end_i = node_i->End();
    // points in node(0)
    for (size_t i = begin_i; i < end_i; i++) {
      
      arma::colvec vec_i = data_mat_list_[0]->col(i);
      
      size_t begin_j = (node_i == node_j) ? i+1 : node_j->Begin();
      size_t end_j = node_j->End();
      // points in node(1)
      for (size_t j = begin_j; j < end_j; j++) {
        
        arma::colvec vec_j = data_mat_list_[1]->col(j);
        
        double dist_ij = metric_.Evaluate(vec_i, vec_j);
        
        if (min_matcher_dist_sqr_ <= dist_ij &&
            dist_ij <= max_matcher_dist_sqr_) {
          result.Increment();
        }
        
      } // for j
      
    } // for i

    
  }

  void TwoPointSingleMatcher::AddResult(PartialResult& result)
  {
    
    results_ += result.results();
    
  }

const std::vector<int>& TwoPointSingleMatcher::results_size() const
  {
    return results_size_;
  }

long long int TwoPointSingleMatcher::results() const
  {
    return results_;
  }

double TwoPointSingleMatcher::weighted_results() const
  {
    return weighted_results_;
  }

int TwoPointSingleMatcher::tuple_size() const
  {
    return 2;
  }

  int TwoPointSingleMatcher::num_permutations() const
  {
    return 1;
  }

const std::vector<arma::colvec*>& TwoPointSingleMatcher::data_weights_list() const
  {
    return data_weights_list_;
  }

const Permutations& TwoPointSingleMatcher::perms() const
  {
    return perms_;
  }

const arma::mat& TwoPointSingleMatcher::lower_bounds_sqr() const
  {
    return lower_bounds_sqr_;
  }

const arma::mat& TwoPointSingleMatcher::upper_bounds_sqr() const
  {
    return upper_bounds_sqr_;
  }

long long int TwoPointSingleMatcher::num_base_cases() const
  {
    return num_base_cases_;
  }

long long int TwoPointSingleMatcher::num_pairs_considered() const
  {
    return num_pairs_considered_;
  }

double TwoPointSingleMatcher::min_dist_sq() const
  {
    return min_matcher_dist_sqr_;
  }

double TwoPointSingleMatcher::max_dist_sq() const
  {
    return max_matcher_dist_sqr_;
  }

int TwoPointSingleMatcher::matcher_ind() const
  {
    return matcher_ind_;
  }

void TwoPointSingleMatcher::set_matcher_ind(int new_ind)
  {
    matcher_ind_ = new_ind;
  }

void TwoPointSingleMatcher::OutputResults()
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
    
  }

} // namespace

