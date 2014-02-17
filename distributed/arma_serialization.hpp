/**
 * @file arma_serialization.hpp
 * @author Bill March (march@gatech.edu)
 *
 * For the distributed code, we need to be able to serialize arma mat and colvec
 */

#ifndef _NPOINT_MLPACK_DISTRIBUTED_ARMA_SERIALIZATION_HPP_
#define _NPOINT_MLPACK_DISTRIBUTED_ARMA_SERIALIZATION_HPP_

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <mlpack/core.hpp>

namespace boost {
  namespace serialization {
    
    /*
     // for arma::mat
     template<class Archive>
     void serialize(Archive& ar, arma::mat& mat, const unsigned int version)
     {
     
     // The const_cast craziness comes from boost serialization
     // See: http://www.boost.org/doc/libs/1_40_0/libs/serialization/doc/serialization.html#const
     ar & const_cast<arma::uword & >(mat.n_rows);
     ar & const_cast<arma::uword & >(mat.n_cols);
     ar & const_cast<arma::uword & >(mat.n_elem);
     
     ar & const_cast<arma::uhword & >(mat.vec_state);
     ar & const_cast<arma::uhword & >(mat.mem_state);
     
     ar & const_cast<double* &>(mat.mem);
     
     }
     */
    
    template<class Archive>
    void save(Archive & ar, const arma::mat& mat,
              unsigned int /*version*/)
    {
      
      ar << const_cast<arma::uword & >(mat.n_rows);
      ar << const_cast<arma::uword & >(mat.n_cols);
      ar << const_cast<arma::uword & >(mat.n_elem);
      
      ar << const_cast<arma::uhword & >(mat.vec_state);
      ar << const_cast<arma::uhword & >(mat.mem_state);
      
      std::vector<double> copy_vec(mat.n_elem);
      std::copy(mat.mem, mat.mem+mat.n_elem, copy_vec.begin());
      
      ar << copy_vec;
      
      //ar << boost::serialization::make_array(mat.mem, mat.n_elem);
      
    }
    
    template<class Archive>
    void load(Archive & ar, arma::mat& mat,
              unsigned int /*version*/)
    {
      
      ar >> const_cast<arma::uword & >(mat.n_rows);
      ar >> const_cast<arma::uword & >(mat.n_cols);
      ar >> const_cast<arma::uword & >(mat.n_elem);
      
      ar >> const_cast<arma::uhword & >(mat.vec_state);
      ar >> const_cast<arma::uhword & >(mat.mem_state);
      
      std::vector<double> out_vec;
      ar >> out_vec;
      
      // is the memory already available
      // I have no idea if this is right, but it does compile
      const_cast<double* &>(mat.mem) = new double[mat.n_elem];
      std::copy(&(out_vec[0]), &(out_vec[0])+mat.n_elem,
                const_cast<double* &>(mat.mem));
      
    }
    
    template <class Archive>
    
    void serialize(Archive & ar, arma::mat& c, const unsigned int version)
    
    {
      
      split_free(ar, c, version);
      
    }
    
    
    
    
    // for arma::colvec
    
    template<class Archive>
    void serialize(Archive& ar, arma::colvec& vec,
                   const unsigned int /*version*/)
    {
      
      //Should be able to get away with this, from same reference above
      ar & base_object<arma::mat>(vec);
      
    }
    
    
    ////// serialization for boost multi arrays
    // TODO: need to add copyright stuff for where I got this
    
    template<class Archive,class T>
    inline void load(
                     Archive & ar,
                     boost::multi_array<T,2> & t,
                     const unsigned int /*file_version*/
                     )
    {
      typedef boost::multi_array<T,2> multi_array_;
      typedef typename multi_array_::size_type size_;
      
      size_ n0;
      ar >> BOOST_SERIALIZATION_NVP(n0);
      size_ n1;
      ar >> BOOST_SERIALIZATION_NVP(n1);
      
      t.resize(boost::extents[n0][n1]);
      ar >> make_array(t.data(), t.num_elements());
    }
    
    template<typename Archive,typename T>
    inline void save(
                     Archive & ar,
                     const boost::multi_array<T,2> & t,
                     const unsigned int /*file_version*/
                     )
    {
    	typedef boost::multi_array<T,2> multi_array_;
      typedef typename multi_array_::size_type size_;
      
      size_ n0 = (t.shape()[0]);
      ar << BOOST_SERIALIZATION_NVP(n0);
      size_ n1 = (t.shape()[1]);
      ar << BOOST_SERIALIZATION_NVP(n1);
      ar << make_array(t.data(), t.num_elements());
    } //save
    
    
    template<class Archive,typename T>
    inline void serialize(
                          Archive & ar,
                          boost::multi_array<T,2>& t,
                          const unsigned int file_version
                          )
    {
      split_free(ar, t, file_version);
    }

    ////////// 3 for multi matcher
    template<class Archive,class T>
    inline void load(
                     Archive & ar,
                     boost::multi_array<T,3> & t,
                     const unsigned int /*file_version*/
                     )
    {
      typedef boost::multi_array<T,3> multi_array_;
      typedef typename multi_array_::size_type size_;
      
      size_ n0;
      ar >> BOOST_SERIALIZATION_NVP(n0);
      size_ n1;
      ar >> BOOST_SERIALIZATION_NVP(n1);
      size_ n2;
      ar >> BOOST_SERIALIZATION_NVP(n2);
      
      t.resize(boost::extents[n0][n1][n2]);
      ar >> make_array(t.data(), t.num_elements());
    }
    
    template<typename Archive,typename T>
    inline void save(
                     Archive & ar,
                     const boost::multi_array<T,3> & t,
                     const unsigned int /*file_version*/
                     )
    {
    	typedef boost::multi_array<T,3> multi_array_;
      typedef typename multi_array_::size_type size_;
      
      size_ n0 = (t.shape()[0]);
      ar << BOOST_SERIALIZATION_NVP(n0);
      size_ n1 = (t.shape()[1]);
      ar << BOOST_SERIALIZATION_NVP(n1);
      size_ n2 = (t.shape()[2]);
      ar << BOOST_SERIALIZATION_NVP(n2);
      
      ar << make_array(t.data(), t.num_elements());
    } //save
    
    
    template<class Archive,typename T>
    inline void serialize(
                          Archive & ar,
                          boost::multi_array<T,3>& t,
                          const unsigned int file_version
                          )
    {
      split_free(ar, t, file_version);
    }

    ////////// 4 for angle matcher //////////////
    template<class Archive,class T>
    inline void load(
                     Archive & ar,
                     boost::multi_array<T,4> & t,
                     const unsigned int /*file_version*/
                     )
    {
      typedef boost::multi_array<T,4> multi_array_;
      typedef typename multi_array_::size_type size_;
      
      size_ n0;
      ar >> BOOST_SERIALIZATION_NVP(n0);
      size_ n1;
      ar >> BOOST_SERIALIZATION_NVP(n1);
      size_ n2;
      ar >> BOOST_SERIALIZATION_NVP(n2);
      size_ n3;
      ar >> BOOST_SERIALIZATION_NVP(n3);
      
      t.resize(boost::extents[n0][n1][n2][n3]);
      ar >> make_array(t.data(), t.num_elements());
    }
    
    template<typename Archive,typename T>
    inline void save(
                     Archive & ar,
                     const boost::multi_array<T,4> & t,
                     const unsigned int /*file_version*/
                     )
    {
    	typedef boost::multi_array<T,4> multi_array_;
      typedef typename multi_array_::size_type size_;
      
      size_ n0 = (t.shape()[0]);
      ar << BOOST_SERIALIZATION_NVP(n0);
      size_ n1 = (t.shape()[1]);
      ar << BOOST_SERIALIZATION_NVP(n1);
      size_ n2 = (t.shape()[2]);
      ar << BOOST_SERIALIZATION_NVP(n2);
      size_ n3 = (t.shape()[3]);
      ar << BOOST_SERIALIZATION_NVP(n3);
      
      ar << make_array(t.data(), t.num_elements());
    } //save
    
    
    template<class Archive,typename T>
    inline void serialize(
                          Archive & ar,
                          boost::multi_array<T,4>& t,
                          const unsigned int file_version
                          )
    {
      split_free(ar, t, file_version);
    }
    

    
  } /// namespace boost::serialization
} /// namespace boost








#endif
