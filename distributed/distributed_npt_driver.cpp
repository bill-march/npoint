//
//  distributed_npt_driver.cpp
//  contrib_march
//
//  Created by William March on 9/4/12.
//
//

#include "distributed_npt_driver.hpp"


// These declarations tell the boost MPI reduce code that adding two results
// classes is commutative, which should improve performance

namespace boost { namespace mpi {
  
  template<>
  struct is_commutative<std::plus<npoint_mlpack::SingleResults>,
                        npoint_mlpack::SingleResults>
  : mpl::true_ { };
  
  template<>
  struct is_commutative<std::plus<npoint_mlpack::MultiResults>,
                        npoint_mlpack::MultiResults>
  : mpl::true_ { };
  
  template<>
  struct is_commutative<std::plus<npoint_mlpack::AngleResults>,
                        npoint_mlpack::AngleResults>
  : mpl::true_ { };
  
} } // end namespace boost::mpi




