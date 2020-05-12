#ifndef __CILKPLUS_RANGE_
#define __CILKPLUS_RANGE_

#ifdef KOKKOS_ENABLE_EMU
   #include<CilkPlus/Kokkos_CilkEmu_Reduce.hpp>
#else
   #include<CilkPlus/Kokkos_CilkPlus_Reduce.hpp>
#endif
#include <cilk/cilk.h>
#include <memory.h>
//Replace specific Emu headers with the tools header to allow x86 compilation
#include <emu_c_utils/emu_c_utils.h>
//#include <pmanip.h>

//#define KOKKOS_CILK_USE_PARALLEL_FOR
#define MAX_THREAD_COUNT 32
/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/
/* Parallel patterns for Kokkos::Experimental::CilkPlus with RangePolicy */

namespace Kokkos {
namespace Impl {

template< class FunctorType , class ... Traits >
class ParallelFor< FunctorType ,
                   Kokkos::RangePolicy< Traits ... > ,
                   Kokkos::Experimental::CilkPlus
                 >
{
private:

  typedef Kokkos::RangePolicy< Traits ... > Policy ;

  const FunctorType m_functor ;
  const Policy      m_policy ;

  template< class TagType >
  typename std::enable_if< std::is_same< TagType , void >::value >::type 
  inner_exec(int inLen_, int sc_, int iLoop, int start) const {	  
	  if ( sc_ > 1 ) {
	     for (int s = 0; s < sc_; s++) {		  
		     _Cilk_spawn inner_exec<TagType>(inLen_, 0, iLoop * sc_ + s, start );
         }
         cilk_sync;
      } else {
         int offset = iLoop * inLen_;
         for ( int j = 0; j < inLen_; j++) {
		     int ndx = start + offset + j;
		     if ( ndx < m_policy.end() )  {
		        m_functor( ndx );
		     }
         }
      }
  }

  template< class TagType >
  typename std::enable_if< ! std::is_same< TagType , void >::value >::type
  inner_exec(const TagType t, int inLen_, int sc_, int iLoop, int start) const {	  
	  if ( sc_ > 1 ) {
	     for (int s = 0; s < sc_; s++) {	
		     _Cilk_spawn inner_exec<TagType>(t, inLen_, 0, iLoop * sc_ + s, start );
         }
         cilk_sync;
      } else {
         int offset = iLoop * inLen_;
         for ( int j = 0; j < inLen_; j++) {
		     int ndx = start + offset + j;
		     if ( ndx < m_policy.end() )
		        m_functor( t, ndx );
         }      
	  }
  }  

  template< class TagType >
  typename std::enable_if< std::is_same< TagType , void >::value >::type
  exec() const
   {  
      const typename Policy::member_type e = m_policy.end();
      const typename Policy::member_type b = m_policy.begin();
      const typename Policy::member_type len = e-b;
      const typename Policy::member_type par_loop = len > MAX_THREAD_COUNT ? MAX_THREAD_COUNT: len;
      typename Policy::member_type int_loop = 1;
      if ( par_loop > 0 )
          int_loop = (len / par_loop) + ( ( (len % par_loop) == 0) ? 0 : 1 );
      
#ifdef KOKKOS_CILK_USE_PARALLEL_FOR
      cilk_for (typename Policy::member_type i = 0 ; i < par_loop ; ++i ) {
         for ( typename Policy::member_type j = (int_loop * i); j < ( (int_loop * i) + int_loop); j++ ) {
           if (j < e) {
              m_functor( (const typename Policy::member_type)(j+b) );
		   }
         }
       }
       cilk_sync;
#else
      int mz = Kokkos::Experimental::EmuReplicatedSpace::memory_zones();
      int sc_count = par_loop / mz + ( ( (par_loop % mz) == 0) ? 0 : 1 );
      
      if (sc_count == 0) sc_count = 1;
      for (typename Policy::member_type i = 0 ; i < mz; ++i ) {  // This should be the number of nodes...
           _Cilk_spawn inner_exec<TagType>(int_loop, sc_count, i, b);
       }
       cilk_sync;
#endif
   }    

  template< class TagType >
  typename std::enable_if< ! std::is_same< TagType , void >::value >::type
  exec() const
    {
	  const TagType t{} ;
      const typename Policy::member_type e = m_policy.end();
      const typename Policy::member_type b = m_policy.begin();
      const typename Policy::member_type len = e-b;
      const typename Policy::member_type par_loop = len > MAX_THREAD_COUNT ? MAX_THREAD_COUNT : len;
      typename Policy::member_type int_loop = 1;
      if ( par_loop > 0 )
          int_loop = (len / par_loop) + ( ( (len % par_loop) == 0) ? 0 : 1 );
      
#ifdef KOKKOS_CILK_USE_PARALLEL_FOR      
      cilk_for (typename Policy::member_type i = 0 ; i < par_loop ; ++i ) {
         for ( typename Policy::member_type j = (int_loop * i); j < ( (int_loop * i) + int_loop); j++ ) {
           if (j < e) {			   
              m_functor( t, (const typename Policy::member_type)(j+b) );
		   }
         }
       }
       cilk_sync;
#else
      int mz = Kokkos::Experimental::EmuReplicatedSpace::memory_zones();
      int sc_count = ( par_loop / mz ) + ( ( (par_loop % mz ) == 0 ) ? 0 : 1);
      if (sc_count == 0) sc_count = 1;
      for (typename Policy::member_type i = 0 ; i < mz ; ++i ) {
           _Cilk_spawn inner_exec<TagType>(t, int_loop, sc_count, i, b);
       }
       cilk_sync;
#endif
    }

public:

  inline
  void execute() const
    { this-> template exec< typename Policy::work_tag >(); }

  inline
  ParallelFor( const FunctorType & arg_functor
             , const Policy      & arg_policy )
    : m_functor( arg_functor )
    , m_policy(  arg_policy )
    {}
};

template< class FunctorType , class ReducerType , class ... Traits >
class ParallelReduce< FunctorType
                    , Kokkos::RangePolicy< Traits ... >
                    , ReducerType
                    , Kokkos::Experimental::CilkPlus
                    >
{
private:

  typedef Kokkos::RangePolicy< Traits ... > Policy ;
  typedef typename Policy::work_tag                                  WorkTag ;
  typedef Kokkos::Experimental::CilkPlus exec_space;

  typedef Kokkos::Impl::if_c< std::is_same<InvalidType,ReducerType>::value, FunctorType, ReducerType> ReducerConditional;

  typedef typename ReducerConditional::type ReducerTypeFwd;
  typedef typename Kokkos::Impl::if_c< std::is_same<InvalidType,ReducerType>::value, WorkTag, void>::type WorkTagFwd;

  typedef FunctorAnalysis< FunctorPatternInterface::REDUCE , Policy , FunctorType > Analysis ;
  typedef Kokkos::Impl::kokkos_cilk_reducer< ReducerTypeFwd, FunctorType, typename Analysis::value_type, WorkTagFwd > cilk_reducer_wrapper;

  typedef Kokkos::Impl::FunctorValueInit<   ReducerTypeFwd , WorkTagFwd >  ValueInit ;

  typedef typename Analysis::pointer_type    pointer_type ;
  typedef typename Analysis::reference_type    reference_type ;
  
  typedef typename cilk_reducer_wrapper::ReducerTypeFwd::value_type reduction_type;
  typedef typename cilk_reducer_wrapper::local_reducer_type local_reducer_type;
  typedef NodeletReducer< typename cilk_reducer_wrapper::reduce_container > nodelet_reducer_type;
  
  const FunctorType   m_functor ;
  const Policy        m_policy ;
  const typename Policy::member_type m_policy_len;
  const typename Policy::member_type m_policy_par_loop;
  const typename Policy::member_type m_policy_int_loop;
  const typename Policy::member_type m_policy_par_size;
  const ReducerType   m_reducer ;
  const pointer_type  m_result_ptr;
  const size_t m_reduce_size;
  
  const void * global_reducer = NULL;
  const void * local_reducer = NULL;
//  const void * working_ptr = NULL;
//  pointer_type pRef[8] = {nullptr, nullptr, nullptr, nullptr, 
//	                      nullptr, nullptr, nullptr, nullptr};
  cilk_reducer_wrapper* reducerRef[8] = {nullptr, nullptr, nullptr, nullptr, 
	                                     nullptr, nullptr, nullptr, nullptr};
  local_reducer_type* localRef[8] = {nullptr, nullptr, nullptr, nullptr, 
	                                     nullptr, nullptr, nullptr, nullptr};
  
  // length of the range
  const typename Policy::member_type get_policy_len() const {
      const typename Policy::member_type e = m_policy.end();
      const typename Policy::member_type b = m_policy.begin();
      return (e-b);
  }
  
  // number of threads
  const typename Policy::member_type get_policy_par_loop() const {
	  return (m_policy_len > MAX_THREAD_COUNT ? MAX_THREAD_COUNT : m_policy_len);
  }
  
  // internal loop per thread
  const typename Policy::member_type get_policy_int_loop() const {      
      if ( m_policy_par_loop > 0 )
          return ((m_policy_len / m_policy_par_loop) + ( ( (m_policy_len % m_policy_par_loop) == 0) ? 0 : 1 ));
      else 
          return 1;
  }
  
  // depth of working strided memory
  const typename Policy::member_type get_policy_par_size() const { 
      if ( m_policy_par_loop > 0 )
	     return ((m_policy_par_loop/Kokkos::Experimental::EmuReplicatedSpace::memory_zones()) + 
	             ( ( (m_policy_par_loop % Kokkos::Experimental::EmuReplicatedSpace::memory_zones()) == 0) ? 0 : 1) );
	  else
	     return 1;
  }
  
  const size_t get_reduce_size(FunctorType func_, ReducerType red_) {
        return Analysis::value_size( ReducerConditional::select(func_ , red_) );
  }


  template< class TagType >
  inline
  typename std::enable_if< std::is_same< TagType , void >::value >::type
  internal_reduce(const typename Policy::member_type b, const typename Policy::member_type e, 
                  int i, const size_t l_alloc_size, int nl_) const {
	 int array_ndx = i & (nl_-1); 
	 //int reduce_off = (i>>log2(nl_));

	 cilk_reducer_wrapper* pReducer = get_reducer<cilk_reducer_wrapper>(reducerRef, array_ndx);
//	 printf("[%d] reduce ndx = %d, off = %d , %lx, %lx \n", i, array_ndx, reduce_off, 
//	               (unsigned long)pRef[array_ndx], (unsigned long)pReducer);
     int start = (b + (m_policy_int_loop * i));
     int end = start + m_policy_int_loop;
	 //printf("[%d] reduce ndx = %d, %lx, %d - %d\n", i, array_ndx, (unsigned long)pReducer, start, end);
	 //fflush(stdout);
     //reduction_type & lupdate = (reduction_type &)*(&(pRef[array_ndx][reduce_off]));     
     reduction_type lupdate;
     nodelet_reducer_type localReduce(reducerRef, array_ndx, cilk_reducer_wrapper::default_value());
     typename cilk_reducer_wrapper::reduce_container::ViewType * cont = localReduce.view();
     
     for ( typename Policy::member_type j = start; j < end; j++ ) {
        if (j < e) {
		   cont->identity(reducerRef, array_ndx, &lupdate);
           pReducer->f( (const typename Policy::member_type)j , lupdate );
           cont->join( reducerRef, array_ndx, lupdate );
        }
     }
     lupdate = localReduce.get_value();
     pReducer->join(lupdate);
  }
  

  template< class TagType >
  inline
  typename std::enable_if< ! std::is_same< TagType , void >::value >::type
  internal_reduce(const typename Policy::member_type b, const typename Policy::member_type e, 
                  int i, const size_t l_alloc_size, int nl_) const {
	 const TagType t{} ;
	 int array_ndx = i & (nl_-1); 
	 //int reduce_off = (i>>log2(nl_)); 
	 cilk_reducer_wrapper* pReducer = get_reducer<cilk_reducer_wrapper>(reducerRef, array_ndx);
     //reduction_type & lupdate = (reduction_type & )*(&(pRef[array_ndx][reduce_off]));     
     reduction_type lupdate;
     
     for ( typename Policy::member_type j = (b+(m_policy_int_loop * i)); j < (b+( (m_policy_int_loop * i) + m_policy_int_loop)); j++ ) {
        if (j < e) {
		   pReducer->init(lupdate);	   
           pReducer->f( t, (const typename Policy::member_type)j , lupdate );
           pReducer->join( lupdate );
        }
     }
  }
  
  void init_reducer(const size_t l_alloc_bytes, int i) const {
     cilk_reducer_wrapper* pH = get_reducer<cilk_reducer_wrapper>(reducerRef, i);         
	 local_reducer_type* pLocalRed = get_reducer<local_reducer_type>(localRef, i);
	 new (pLocalRed) nodelet_reducer_type(reducerRef, i, cilk_reducer_wrapper::default_value());
     new (pH) cilk_reducer_wrapper(reducerRef, i, ReducerConditional::select(m_functor , m_reducer), l_alloc_bytes, pLocalRed);
  }

  void initialize_cilk_reducer(const size_t l_alloc_bytes) const
  {      
	  //long * refPtr = Kokkos::Experimental::EmuReplicatedSpace::getRefAddr();	      
      for (int i = 0; i < Kokkos::Experimental::EmuReplicatedSpace::memory_zones(); i++) {
         _Cilk_spawn init_reducer(l_alloc_bytes, i);
      }
      cilk_sync;
  }

  template< class TagType >
  inline
  typename std::enable_if< std::is_same< TagType , void >::value >::type
  exec( reference_type update ) const
    {      
	  int nl_ = Kokkos::Experimental::EmuReplicatedSpace::memory_zones();
      
      const typename Policy::member_type e = m_policy.end();
      const typename Policy::member_type b = m_policy.begin();         
               
      for (int i = 0; i < m_policy_par_loop; i++) {
 	     //int node_ = i % nl_;
         _Cilk_spawn this->template internal_reduce<TagType>(b, e, i, m_reduce_size * m_policy_par_size, nl_);
      }
      if (m_policy_par_loop > 0) cilk_sync;
         
      cilk_reducer_wrapper* pReducerHost = get_reducer<cilk_reducer_wrapper>(reducerRef, 0);
      for (int i = 1; i < nl_; i++) {
		  cilk_reducer_wrapper* pReducerNode = get_reducer<cilk_reducer_wrapper>(reducerRef, i);
		  reduction_type lRef;
		  pReducerNode->update_value(lRef);
		  pReducerHost->join(lRef);
	  }
         
      get_reducer<cilk_reducer_wrapper>(reducerRef, 0)->update_value(update);
      //printf("final node value: %d \n", update);
      for (int i = 1; i < nl_; i++) {
         cilk_reducer_wrapper * pWrap = 
		        get_reducer<cilk_reducer_wrapper>(reducerRef, i);
		 pWrap->release_resources();
         pWrap->~cilk_reducer_wrapper();
	  }                 
  }

  template< class TagType >
  inline
  typename std::enable_if< ! std::is_same< TagType , void >::value >::type
  exec( reference_type update ) const
  {      
	  int nl_ = Kokkos::Experimental::EmuReplicatedSpace::memory_zones();
      
      const typename Policy::member_type e = m_policy.end();
      const typename Policy::member_type b = m_policy.begin();

	  for (int i = 0; i < m_policy_par_loop; i++) {
		 //int node_ = i % nl_;
		 _Cilk_spawn this->template internal_reduce<TagType>(b, e, i, m_reduce_size*m_policy_par_size, nl_);
	  }
	  cilk_sync;
	  cilk_reducer_wrapper* pReducerHost = get_reducer<cilk_reducer_wrapper>(reducerRef, 0);
	  for (int i = 1; i < nl_; i++) {
	 	 cilk_reducer_wrapper* pReducerNode = get_reducer<cilk_reducer_wrapper>(reducerRef, i);
		 reduction_type lRef;
		 pReducerNode->update_value(lRef);
		 pReducerHost->join(lRef);
	  }
	 	 
      get_reducer<cilk_reducer_wrapper>(reducerRef, 0)->update_value(update);
      //printf("final node value: %d \n", update);
      for (int i = 1; i < nl_; i++) {
         cilk_reducer_wrapper * pWrap = 
		        get_reducer<cilk_reducer_wrapper>(reducerRef, i);
		 pWrap->release_resources();
         pWrap->~cilk_reducer_wrapper();
	  }        
	   
   }

public:

  inline
  void execute() const
    {
      KOKKOS_ASSERT(global_reducer != NULL && "Global Reducer Check");
      KOKKOS_ASSERT(local_reducer != NULL && "Local Reducer Check");
      //KOKKOS_ASSERT(working_ptr != NULL && "Local Reducer Check");
      
      const size_t team_reduce_size  = 0 ; // Never shrinks
      const size_t team_shared_size  = 0 ; // Never shrinks
      const size_t thread_local_size = 0 ; // Never shrinks

      serial_resize_thread_team_data( m_reduce_size
                                    , team_reduce_size
                                    , team_shared_size
                                    , thread_local_size );

      HostThreadTeamData & data = *serial_get_thread_team_data();

      pointer_type ptr =
        m_result_ptr ? m_result_ptr : pointer_type(data.pool_reduce_local());

      reference_type update =
        ValueInit::init(  ReducerConditional::select(m_functor , m_reducer) , ptr );
        
      if ( m_policy_par_size > 0 ) {
		 
		 initialize_cilk_reducer(m_reduce_size);

         this-> template exec< WorkTag >( update );

         Kokkos::Impl::FunctorFinal< ReducerTypeFwd , WorkTagFwd >::
           final(  ReducerConditional::select(m_functor , m_reducer) , ptr );
	   }

    }

  template< class HostViewType >
  ParallelReduce( const FunctorType  & arg_functor ,
                  const Policy       & arg_policy ,
                  const HostViewType & arg_result_view ,
                  typename std::enable_if<
                               Kokkos::is_view< HostViewType >::value &&
                              !Kokkos::is_reducer_type<ReducerType>::value
                  ,void*>::type = NULL)
    : m_functor( arg_functor )
    , m_policy( arg_policy )
    , m_policy_len( get_policy_len() )
    , m_policy_par_loop( get_policy_par_loop() )
    , m_policy_int_loop( get_policy_int_loop() )
    , m_policy_par_size( get_policy_par_size() )
    , m_reducer( InvalidType() )
    , m_result_ptr( arg_result_view.data() )
    , m_reduce_size( get_reduce_size( m_functor, m_reducer ) )
    , global_reducer ( mw_malloc2d(Kokkos::Experimental::EmuReplicatedSpace::memory_zones(), // one for each memory zone...
                                   2 * sizeof(cilk_reducer_wrapper)
                                   ) 
                     )  
    , local_reducer ( mw_malloc2d(Kokkos::Experimental::EmuReplicatedSpace::memory_zones(), 
                                  sizeof(nodelet_reducer_type)
                                  )
                    ) 
 //   , working_ptr ( mw_malloc2d(Kokkos::Experimental::EmuReplicatedSpace::memory_zones(), 
 //                               m_policy_par_size * m_reduce_size
 //                               )
 //                 ) 
    {
//      printf("Working ptr: %lx \n", (unsigned long)working_ptr);
//      fflush(stdout);
      static_assert( Kokkos::is_view< HostViewType >::value
        , "Kokkos::Experimental::CilkPlus reduce result must be a View" );

      static_assert( std::is_same< typename HostViewType::memory_space , HostSpace >::value
        , "Kokkos::Experimental::CilkPlus reduce result must be a View in HostSpace" );
      for (int i = 0; i < Kokkos::Experimental::EmuReplicatedSpace::memory_zones(); i++) {
//	     pRef[i] = (pointer_type)mw_arrayindex((void*)working_ptr, i, Kokkos::Experimental::EmuReplicatedSpace::memory_zones(),  
//		                                m_policy_par_size * m_reduce_size);
         reducerRef[i] = (cilk_reducer_wrapper*)mw_arrayindex((void*)global_reducer, i, Kokkos::Experimental::EmuReplicatedSpace::memory_zones(),  
		                                       sizeof(cilk_reducer_wrapper));
	     localRef[i] = (local_reducer_type*)mw_arrayindex((void*)local_reducer, i, Kokkos::Experimental::EmuReplicatedSpace::memory_zones(),  
		                                       sizeof(nodelet_reducer_type));
      }
    }
  inline
  ParallelReduce( const FunctorType & arg_functor
                , Policy       arg_policy
                , const ReducerType& reducer )
    : m_functor( arg_functor )
    , m_policy(  arg_policy )
    , m_policy_len( get_policy_len() )
    , m_policy_par_loop( get_policy_par_loop() )
    , m_policy_int_loop( get_policy_int_loop() )
    , m_policy_par_size( get_policy_par_size() )    
    , m_reducer( reducer )
    , m_result_ptr(  reducer.view().data() )
    , m_reduce_size( get_reduce_size( m_functor, m_reducer ) )
    , global_reducer ( mw_malloc2d(Kokkos::Experimental::EmuReplicatedSpace::memory_zones(), sizeof(cilk_reducer_wrapper)) )  // one for each memory zone...
    , local_reducer ( mw_malloc2d(Kokkos::Experimental::EmuReplicatedSpace::memory_zones(), sizeof(nodelet_reducer_type)))     
//    , working_ptr ( mw_malloc2d(Kokkos::Experimental::EmuReplicatedSpace::memory_zones(), m_policy_par_size * m_reduce_size )) 
    {
//      printf("Working ptr: %lx \n", (unsigned long)working_ptr);
      fflush(stdout);
		for (int i = 0; i < Kokkos::Experimental::EmuReplicatedSpace::memory_zones(); i++) {
//		   pRef[i] = (pointer_type)mw_arrayindex((void*)working_ptr, i, Kokkos::Experimental::EmuReplicatedSpace::memory_zones(),  
//		                                m_policy_par_size * m_reduce_size);
		   reducerRef[i] = (cilk_reducer_wrapper*)mw_arrayindex((void*)global_reducer, i, Kokkos::Experimental::EmuReplicatedSpace::memory_zones(),  
		                                       sizeof(cilk_reducer_wrapper));
		   localRef[i] = (local_reducer_type*)mw_arrayindex((void*)local_reducer, i, Kokkos::Experimental::EmuReplicatedSpace::memory_zones(),  
		                                       sizeof(nodelet_reducer_type));
        }
    }
    
    inline
    ~ParallelReduce() {
//      printf("at dest ... Working ptr: %lx \n", (unsigned long)working_ptr);
//      fflush(stdout);
      mw_free((void*)global_reducer);
      mw_free((void*)local_reducer);
//      mw_free((void*)working_ptr);
	}
};
}
}

#endif
