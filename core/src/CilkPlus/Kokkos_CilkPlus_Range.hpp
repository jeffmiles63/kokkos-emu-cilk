#ifndef __CILKPLUS_RANGE_
#define __CILKPLUS_RANGE_

#ifdef KOKKOS_ENABLE_EMU
   #include<CilkPlus/Kokkos_CilkEmu_Reduce.hpp>
#else
   #include<CilkPlus/Kokkos_CilkPlus_Reduce.hpp>
#endif
#include <cilk/cilk.h>
#include <memoryweb/memory.h>

//#define KOKKOS_CILK_USE_PARALLEL_FOR
#define MAX_THREAD_COUNT 64
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
  inner_exec(int inLen_, int sc_, int iLoop) const {	  
	  if ( sc_ > 0 ) {
	     for (int s = 0; s < sc_; s++) {		  
		     _Cilk_spawn inner_exec<TagType>(inLen_, 0, iLoop * sc_ + s );
         }
      } else {
         int offset = iLoop * inLen_;
         for ( int j = 0; j < inLen_; j++) {
		     int ndx = offset + j;
		     if ( ndx < m_policy.end() )  
		        m_functor( ndx );
         }
      }
  }

  template< class TagType >
  typename std::enable_if< ! std::is_same< TagType , void >::value >::type
  inner_exec(const TagType t, int inLen_, int sc_, int iLoop) const {	  
	  if ( sc_ > 0 ) {
	     for (int s = 0; s < sc_; s++) {	
			 //printf("spawn inner: s = %d , inlen = %d, len = %d \n", (const int)s, (const int)inLen_, (const int)iLoop * sc_ + s);	  
		     _Cilk_spawn inner_exec<TagType>(t, inLen_, 0, iLoop * sc_ + s );
         }
      } else {
         int offset = iLoop * inLen_;
         for ( int j = 0; j < inLen_; j++) {
		     int ndx = offset + j;
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
      //printf(" cilk parallel for: b= %d, e = %d, l = %d, par = %d, int = %d \n", b, e, len, par_loop, int_loop);
      cilk_for (typename Policy::member_type i = 0 ; i < par_loop ; ++i ) {
         for ( typename Policy::member_type j = (int_loop * i); j < ( (int_loop * i) + int_loop); j++ ) {
           if (j < e)
              m_functor( (const typename Policy::member_type)j );
         }
       }
#else
      //long * refPtr = Kokkos::Experimental::EmuReplicatedSpace::getRefAddr();
      int mz = Kokkos::Experimental::EmuReplicatedSpace::memory_zones();
      int sc_count = par_loop / mz + ( ( (par_loop % mz) == 0) ? 0 : 1 );
      printf(" tree spawn parallel for: b= %d, e = %d, l = %d, par = %d, sc = %d, int = %d \n", b, e, len, par_loop, sc_count, int_loop);
      fflush(stdout);
      for (typename Policy::member_type i = 0 ; i < par_loop / sc_count ; ++i ) {  // This should be the number of nodes...
           //printf(" parallel for spawn: i = %d , %08x \n", (const int)i, &refPtr[i]);
           //fflush(stdout);
           //_Cilk_migrate_hint(&refPtr[i]);
           _Cilk_spawn inner_exec<TagType>(int_loop, sc_count, i);          
       }
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
      //printf("T parallel for: b= %d, e = %d, l = %d, par = %d, int = %d \n", b, e, len, par_loop, int_loop);
      cilk_for (typename Policy::member_type i = 0 ; i < par_loop ; ++i ) {
         for ( typename Policy::member_type j = (int_loop * i); j < ( (int_loop * i) + int_loop); j++ ) {
           if (j < e)
              m_functor( t, (const typename Policy::member_type)j );
         }
       }
#else
      long * refPtr = Kokkos::Experimental::EmuReplicatedSpace::getRefAddr();
      int sc_count = par_loop / Kokkos::Experimental::EmuReplicatedSpace::memory_zones();
      //printf("T tree spawn parallel for: b= %d, e = %d, l = %d, par = %d, sc = %d, int = %d \n", b, e, len, par_loop, sc_count, int_loop);
      for (typename Policy::member_type i = 0 ; i < par_loop / sc_count ; ++i ) {
           //printf("T parallel for spawn: i = %d \n", (const int)i);
           _Cilk_migrate_hint(&refPtr[i]);
           _Cilk_spawn inner_exec<TagType>(t, int_loop, sc_count, i);
       }
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
  typedef typename Analysis::reference_type  reference_type ;

  const FunctorType   m_functor ;
  const Policy        m_policy ;
  const ReducerType   m_reducer ;
  const pointer_type  m_result_ptr ;

  void internal_reduce(const typename Policy::member_type e, int par_size, int int_loop, int i, void* i_ptr, const size_t l_alloc_bytes) const {
	 int array_ndx = i / par_size; 
	 cilk_reducer_wrapper* pReducer = get_reducer<cilk_reducer_wrapper>(array_ndx);
	 //printf("[%d.%d] internal reduce: %08x, %d, %d, %d \n", NODE_ID(), THREAD_ID(), (unsigned long)pReducer, e, par_size, i);
	 //Kokkos::Experimental::print_pointer(i, pReducer, "reducer pointer" );
	 //fflush(stdout);         
     //printf("obtaining update pointer: %d, %d \n", i, array_ndx);          
     pointer_type pRef = (pointer_type)mw_arrayindex((void*)i_ptr, array_ndx, NODELETS(),  l_alloc_bytes * par_size);
     //Kokkos::Experimental::print_pointer(i, pRef, "internal reduce (outer)" );
     //Kokkos::Experimental::print_pointer(i, &pRef[i%par_size], "internal reduce (inner)" );
     //printf("[%d] obtaining update reference %d, %d: %08x, offset %d\n", NODE_ID(), i, array_ndx, pRef, i%par_size);
     //reference_type lupdate = ValueInit::init(  pReducer->f , &pRef[i%par_size] );
     reference_type lupdate = pRef[i%par_size] = 0;
     //printf("[%d.%d] pointer node: %d \n", NODE_ID(), THREAD_ID(), mw_ptrtonodelet(&lupdate) );
     //Kokkos::Experimental::print_pointer(i, &lupdate, "entering inner loop" );
     
     for ( typename Policy::member_type j = (int_loop * i); j < ( (int_loop * i) + int_loop); j++ ) {
		//printf("[%d] calling functor: %d - %d \n", NODE_ID(), e, (int)j);
		//fflush(stdout);
        if (j < e) {           
           pReducer->f( (const typename Policy::member_type)j , lupdate );
           pReducer->join( lupdate );
        }
     }
  }

  void initialize_cilk_reducer(const size_t l_alloc_bytes) const
  {      
	  long * refPtr = Kokkos::Experimental::EmuReplicatedSpace::getRefAddr();
      Kokkos::Experimental::EmuReplicatedSpace* pMem = ((Kokkos::Experimental::EmuReplicatedSpace*)mw_ptr1to0(Kokkos::Experimental::EmuReplicatedSpace::ers));
      global_reducer = pMem->allocate(sizeof(cilk_reducer_wrapper)); 
      global_reducer_local = pMem->allocate(sizeof(typename cilk_reducer_wrapper::local_reducer_type));     
      for (int i = 0; i < NODELETS(); i++) {
		 //printf("initializing reducer for node %d\n", i);		 
		 cilk_reducer_wrapper* pH = (cilk_reducer_wrapper*)mw_get_localto(global_reducer, &refPtr[i]);         
		 void* pLocalRed = (void*)mw_get_localto(global_reducer_local, &refPtr[i]);         
		 new (pLocalRed) NodeletReducer< typename cilk_reducer_wrapper::reduce_container >(cilk_reducer_wrapper::default_value());
         new (pH) cilk_reducer_wrapper(ReducerConditional::select(m_functor , m_reducer), l_alloc_bytes, pLocalRed);
         //Kokkos::Experimental::print_pointer( i, pH, "init reducer" );
      }
  }

  template< class TagType >
  inline
  typename std::enable_if< std::is_same< TagType , void >::value >::type
  exec( reference_type update, const size_t l_alloc_bytes ) const
    {
      Kokkos::HostSpace space;
      initialize_cilk_reducer(l_alloc_bytes);

      const typename Policy::member_type e = m_policy.end();
      const typename Policy::member_type b = m_policy.begin();
      const typename Policy::member_type len = e-b;
      const typename Policy::member_type par_loop = len > 16 ? 16 : len;
      typename Policy::member_type int_loop = 1;
      if ( par_loop > 0 )
          int_loop = (len / par_loop) + ( ( (len % par_loop) == 0) ? 0 : 1 );

      int par_size = par_loop/NODELETS();
      void * w_ptr = mw_malloc2d(par_loop, l_alloc_bytes * par_size);  // one for each thread...
      long * refPtr = Kokkos::Experimental::EmuReplicatedSpace::getRefAddr();
      
      //printf("parallel reduce: b= %d, e = %d, l = %d, par = %d, int = %d \n", b, e, len, par_loop, int_loop);
//      cilk_for (typename Policy::member_type i = 0 ; i < par_loop ; ++i ) {
//      }

      //printf("internal reduce size = %d ... spawning threads... \n", (int)l_alloc_bytes);
      //fflush(stdout);
      for (int i = 0; i < par_loop; i++) {
         cilk_spawn_at(&refPtr[i/par_size]) internal_reduce(e, par_size, int_loop, i, w_ptr, l_alloc_bytes);
      }
      get_reducer<cilk_reducer_wrapper>(NODE_ID())->update_value(update);
      get_reducer<cilk_reducer_wrapper>(NODE_ID())->release_resources();
      global_reducer = NULL;
      global_reducer_local = NULL;
      if (w_ptr != NULL) {
            mw_free(w_ptr);
//          space.deallocate(w_ptr, working_set);
      }
    }

  template< class TagType >
  inline
  typename std::enable_if< ! std::is_same< TagType , void >::value >::type
  exec( reference_type update, const size_t l_alloc_bytes ) const
    {
      const TagType t{} ;

      Kokkos::HostSpace space;
      cilk_reducer_wrapper cilk_reducer(ReducerConditional::select(m_functor , m_reducer), l_alloc_bytes);
      INITIALIZE_CILK_REDUCER( cilk_reducer_wrapper, cilk_reducer )
      size_t working_set = l_alloc_bytes * (m_policy.end() - m_policy.begin());
      void * w_ptr = NULL; 
      if (working_set > 0) {      
         w_ptr = space.allocate( working_set );
         memset( w_ptr, 0, working_set );
      }
      const typename Policy::member_type e = m_policy.end();
      const typename Policy::member_type b = m_policy.begin();
      const typename Policy::member_type len = e-b;
      const typename Policy::member_type par_loop = len > 16 ? 16 : len;
      typename Policy::member_type int_loop = 1;
      if ( par_loop > 0 )
          int_loop = (len / par_loop) + ( ( (len % par_loop) == 0) ? 0 : 1 );
//      printf("T: parallel reduce: b= %d, e = %d, l = %d, par = %d, int = %d \n", b, e, len, par_loop, int_loop);
      cilk_for (typename Policy::member_type i = 0 ; i < par_loop ; ++i ) {
        for ( typename Policy::member_type j = (int_loop * i); j < ( (int_loop * i) + int_loop); j++ ) {
           if (j < e && w_ptr != NULL) {
              void * const l_ptr = (void*)(((char*)w_ptr)+((i - m_policy.begin()) * l_alloc_bytes));
              reference_type lupdate = ValueInit::init(  ReducerConditional::select(m_functor , m_reducer) , l_ptr );
              m_functor( t, (const typename Policy::member_type)i , lupdate );
              cilk_reducer.join( lupdate );
           }
        }
      }
      cilk_reducer.update_value( update );
      cilk_reducer.release_resources();
      global_reducer = NULL;
      global_reducer_local = NULL;
      if (w_ptr != NULL) {
          space.deallocate(w_ptr, working_set);
      }
    }

public:

  inline
  void execute() const
    {

      const size_t pool_reduce_size =
        Analysis::value_size( ReducerConditional::select(m_functor , m_reducer) );
      const size_t team_reduce_size  = 0 ; // Never shrinks
      const size_t team_shared_size  = 0 ; // Never shrinks
      const size_t thread_local_size = 0 ; // Never shrinks

      serial_resize_thread_team_data( pool_reduce_size
                                    , team_reduce_size
                                    , team_shared_size
                                    , thread_local_size );

      HostThreadTeamData & data = *serial_get_thread_team_data();

      pointer_type ptr =
        m_result_ptr ? m_result_ptr : pointer_type(data.pool_reduce_local());

      reference_type update =
        ValueInit::init(  ReducerConditional::select(m_functor , m_reducer) , ptr );

      this-> template exec< WorkTag >( update, pool_reduce_size );

      Kokkos::Impl::FunctorFinal< ReducerTypeFwd , WorkTagFwd >::
        final(  ReducerConditional::select(m_functor , m_reducer) , ptr );


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
    , m_reducer( InvalidType() )
    , m_result_ptr( arg_result_view.data() )
    {
      static_assert( Kokkos::is_view< HostViewType >::value
        , "Kokkos::Experimental::CilkPlus reduce result must be a View" );

      static_assert( std::is_same< typename HostViewType::memory_space , HostSpace >::value
        , "Kokkos::Experimental::CilkPlus reduce result must be a View in HostSpace" );
    }
  inline
  ParallelReduce( const FunctorType & arg_functor
                , Policy       arg_policy
                , const ReducerType& reducer )
    : m_functor( arg_functor )
    , m_policy(  arg_policy )
    , m_reducer( reducer )
    , m_result_ptr(  reducer.view().data() )
    {
    }
};
}
}

#endif
