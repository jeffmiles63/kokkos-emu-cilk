#ifndef __CILKPLUS_RANGE_
#define __CILKPLUS_RANGE_

#ifdef KOKKOS_ENABLE_EMU
   #include<CilkPlus/Kokkos_CilkEmu_Reduce.hpp>
#else
   #include<CilkPlus/Kokkos_CilkPlus_Reduce.hpp>
#endif
#include <cilk/cilk.h>
#include <memory.h>
#include <pmanip.h>

//#define KOKKOS_CILK_USE_PARALLEL_FOR

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
			 //printf("spawn inner: s = %d , inlen = %d, len = %d \n", (const int)s, (const int)inLen_, (const int)iLoop * sc_ + s);	  
			 //fflush(stdout);
		     _Cilk_spawn inner_exec<TagType>(inLen_, 0, iLoop * sc_ + s, start );
         }
         cilk_sync;
      } else {
		 //printf("inner: inlen = %d, len = %d \n", (const int)inLen_, (const int)iLoop);	  
		 //fflush(stdout);
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
			 //printf("spawn inner: s = %d , inlen = %d, len = %d \n", (const int)s, (const int)inLen_, (const int)iLoop * sc_ + s);	  
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
      const typename Policy::member_type par_loop = len > Kokkos::Experimental::CilkPlus::thread_pool_size(0) ? 
                                                          Kokkos::Experimental::CilkPlus::thread_pool_size(0) : len;
      typename Policy::member_type int_loop = 1;
      if ( par_loop > 0 )
          int_loop = (len / par_loop) + ( ( (len % par_loop) == 0) ? 0 : 1 );
      
#ifdef KOKKOS_CILK_USE_PARALLEL_FOR
      //printf(" cilk parallel for: b= %d, e = %d, l = %d, par = %d, int = %d \n", b, e, len, par_loop, int_loop);
      cilk_for (typename Policy::member_type i = 0 ; i < par_loop ; ++i ) {
         for ( typename Policy::member_type j = (int_loop * i); j < ( (int_loop * i) + int_loop); j++ ) {
           if (j < e) {
              m_functor( (const typename Policy::member_type)(j+b) );
		   }
         }
       }
       cilk_sync;
#else
      //long * refPtr = Kokkos::Experimental::EmuReplicatedSpace::getRefAddr();
      int mz = Kokkos::Experimental::EmuReplicatedSpace::memory_zones();
      int sc_count = par_loop / mz + ( ( (par_loop % mz) == 0) ? 0 : 1 );
      
      if (sc_count == 0) sc_count = 1;
      //printf(" tree spawn parallel for: b= %d, e = %d, l = %d, par = %d, sc = %d, int = %d \n", b, e, len, par_loop, sc_count, int_loop);
      //fflush(stdout);
      for (typename Policy::member_type i = 0 ; i < mz; ++i ) {  // This should be the number of nodes...
           //printf(" parallel for spawn: i = %d , %08x \n", (const int)i, &refPtr[i]);
           //fflush(stdout);
           //_Cilk_migrate_hint(&refPtr[i]);
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
      const typename Policy::member_type par_loop = len > Kokkos::Experimental::CilkPlus::thread_pool_size(0) ? 
                                                          Kokkos::Experimental::CilkPlus::thread_pool_size(0) : len;
      typename Policy::member_type int_loop = 1;
      if ( par_loop > 0 )
          int_loop = (len / par_loop) + ( ( (len % par_loop) == 0) ? 0 : 1 );
      
#ifdef KOKKOS_CILK_USE_PARALLEL_FOR      
      //printf("T parallel for: b= %d, e = %d, l = %d, par = %d, int = %d \n", b, e, len, par_loop, int_loop);
      cilk_for (typename Policy::member_type i = 0 ; i < par_loop ; ++i ) {
         for ( typename Policy::member_type j = (int_loop * i); j < ( (int_loop * i) + int_loop); j++ ) {
           if (j < e) {			   
              m_functor( t, (const typename Policy::member_type)(j+b) );
		   }
         }
       }
       cilk_sync;
#else
      //long * refPtr = Kokkos::Experimental::EmuReplicatedSpace::getRefAddr();
      int mz = Kokkos::Experimental::EmuReplicatedSpace::memory_zones();
      int sc_count = ( par_loop / mz ) + ( ( (par_loop % mz ) == 0 ) ? 0 : 1);
      if (sc_count == 0) sc_count = 1;
      //printf("T tree spawn parallel for: b= %d, e = %d, l = %d, par = %d, sc = %d, int = %d \n", b, e, len, par_loop, sc_count, int_loop);
      for (typename Policy::member_type i = 0 ; i < mz ; ++i ) {
           //printf("T parallel for spawn: i = %d \n", (const int)i);           
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


template< class FunctorType , class ... Properties >
class ParallelFor< FunctorType
                 , Kokkos::TeamPolicy< Properties ... >
                 , Kokkos::Experimental::CilkPlus
                 >
{
private:

  enum { TEAM_REDUCE_SIZE = 512 };

  typedef Kokkos::Impl::TeamPolicyInternal< Kokkos::Experimental::CilkPlus, Properties ... > Policy ;
  typedef typename Policy::work_tag             WorkTag ;
  typedef typename Policy::schedule_type::type  SchedTag ;
  typedef typename Policy::member_type          Member ;

  const FunctorType    m_functor;
  const Policy         m_policy;
  const int            m_shmem_size;

  template< class TagType >
  inline static
  typename std::enable_if< ( std::is_same< TagType , void >::value ) >::type
  exec_team( const FunctorType & functor
           , HostThreadTeamData & data
           , const int league_rank_begin
           , const int league_rank_end
           , const int league_size )
    {
      for ( int r = league_rank_begin ; r < league_rank_end ; ) {

        functor( Member( data, r , league_size ) );

        if ( ++r < league_rank_end ) {
          // Don't allow team members to lap one another
          // so that they don't overwrite shared memory.
          if ( data.team_rendezvous() ) { data.team_rendezvous_release(); }
        }
      }
    }


  template< class TagType >
  inline static
  typename std::enable_if< ( ! std::is_same< TagType , void >::value ) >::type
  exec_team( const FunctorType & functor
           , HostThreadTeamData & data
           , const int rank_begin
           , const int rank_end
           , const int league_size )
    {
      const TagType t{};

      for ( int r = league_rank_begin ; r < league_rank_end ; ) {

        functor( t , Member( data, r , league_size ) );

        if ( ++r < league_rank_end ) {
          // Don't allow team members to lap one another
          // so that they don't overwrite shared memory.
          if ( data.team_rendezvous() ) { data.team_rendezvous_release(); }
        }
      }
    }
    
    void launch_exec_thread( int league_rank, int team_rank ) {
       std::pair<int64_t,int64_t> range(0,0);
       enum { is_dynamic = std::is_same< SchedTag , Kokkos::Dynamic >::value };
       
       do {

          range = is_dynamic ? data.get_work_stealing_chunk()
                             : data.get_work_partition();

          ParallelFor::template exec_team< WorkTag >
              ( m_functor , data
              , range.first , range.second , m_policy.league_size() );

        } while ( is_dynamic && 0 <= range.first );

	}
    
    void launch_exec_team(int league_rank) {
		
        for (int j = 0; j < m_policy.team_size(); j++) {
			cilk_spawn launch_exec_thread( league_rank, league_size );
		}      

	}

public:

  inline
  void execute() const
    {
      const size_t pool_reduce_size = 0 ; // Never shrinks
      const size_t team_reduce_size = TEAM_REDUCE_SIZE * m_policy.team_size();
      const size_t team_shared_size = m_shmem_size + m_policy.scratch_size(1);
      const size_t thread_local_size = 0 ; // Never shrinks
      long * pRef = Kokkos::Experimental::EmuReplicatedSpace::getRefAddr();

      for (int i = 0; i < m_policy.league_size(); i++) {
		  cilk_spawn_at(&pRef[i%Kokkos::Experimental::EmuReplicatedSpace::memory_zones()])
		       launch_exec_team(i);
      }
      cilk_sync;
    }


  inline
  ParallelFor( const FunctorType & arg_functor ,
               const Policy      & arg_policy )
    : m_instance( t_openmp_instance )
    , m_functor( arg_functor )
    , m_policy(  arg_policy )
    , m_shmem_size( arg_policy.scratch_size(0) +
                    arg_policy.scratch_size(1) +
                    FunctorTeamShmemSize< FunctorType >
                      ::value( arg_functor , arg_policy.team_size() ) )
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
  const void * working_ptr = NULL; 
  
  // length of the range
  const typename Policy::member_type get_policy_len() const {
      const typename Policy::member_type e = m_policy.end();
      const typename Policy::member_type b = m_policy.begin();
      return (e-b);
  }
  
  // number of threads
  const typename Policy::member_type get_policy_par_loop() const {
	  return (m_policy_len > Kokkos::Experimental::CilkPlus::thread_pool_size(0) ? 
	                         Kokkos::Experimental::CilkPlus::thread_pool_size(0) : m_policy_len);
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
	 int array_ndx = i % nl_;
	 int reduce_off = i / nl_;
	 //printf("[%d] reduce ndx = %d, off = %d \n", i, array_ndx, reduce_off);
	 //fflush(stdout);
	 cilk_reducer_wrapper* pReducer = get_reducer<cilk_reducer_wrapper>(global_reducer, array_ndx);
	 //printf("void [%d.%d] internal reduce: %lx, %d, %d, %d, %d \n", NODE_ID(), array_ndx, (unsigned long)pReducer, e, m_policy_par_size, i, m_policy_int_loop);
	 //Kokkos::Experimental::print_pointer(i, pReducer, "reducer pointer" );
	 
//     printf("obtaining update pointer: %d, %d %lx \n", i, array_ndx, i_ptr);          
     //fflush(stdout);

     pointer_type pRef = (pointer_type)mw_arrayindex((void*)working_ptr, array_ndx,  nl_,  l_alloc_size);
     //Kokkos::Experimental::print_pointer(i, pRef, "internal reduce (outer)" );
     //Kokkos::Experimental::print_pointer(i, &pRef[i%par_size], "internal reduce (inner)" );
//     printf("[%d] obtaining update reference %d: %lx, offset %d\n", i, array_ndx, pRef, i%par_size);
//     fflush(stdout);
     //reference_type lupdate = ValueInit::init(  pReducer->f , &pRef[i%par_size] );
     //pRef[i%par_size] = reduction_type();
     reduction_type & lupdate = (reduction_type &)*(&pRef[reduce_off]);     
     //printf("[%d.%d] pointer node: %d \n", NODE_ID(), THREAD_ID(), mw_ptrtonodelet(&lupdate) );
     //Kokkos::Experimental::print_pointer(i, &lupdate, "entering inner loop" );
     
     for ( typename Policy::member_type j = (b + (m_policy_int_loop * i)); j < (b + ( (m_policy_int_loop * i) + m_policy_int_loop)); j++ ) {
        if (j < e) {
		   pReducer->init(lupdate);
           pReducer->f( (const typename Policy::member_type)j , lupdate );
           //printf("[%d, %d] return from functor: %d - %d (%lx) \n", array_ndx, reduce_off, (int)j, lupdate, &lupdate);
 		   //fflush(stdout);
           pReducer->join( lupdate );
        }
     }
  }
  

  template< class TagType >
  inline
  typename std::enable_if< ! std::is_same< TagType , void >::value >::type
  internal_reduce(const typename Policy::member_type b, const typename Policy::member_type e, 
                  int i, const size_t l_alloc_size, int nl_) const {
	 const TagType t{} ;
	 
	 int array_ndx = i % nl_; 
	 int reduce_off = i / nl_;
	 //printf("[%d] reduce ndx = %d, off = %d \n", i, array_ndx, reduce_off);
	 //fflush(stdout);
	 
	 cilk_reducer_wrapper* pReducer = get_reducer<cilk_reducer_wrapper>(global_reducer, array_ndx);
	 //printf("Tag [%d] internal reduce: %lx, %d, %d, %d \n", array_ndx, (unsigned long)pReducer, b, m_policy_par_size, i);
	 //Kokkos::Experimental::print_pointer(i, pReducer, "reducer pointer" );
	 //fflush(stdout);         
//     printf("obtaining update pointer: %d, %d %lx \n", i, array_ndx, i_ptr);          
//     fflush(stdout);
     pointer_type pRef = (pointer_type)mw_arrayindex((void*)working_ptr, array_ndx, nl_,  l_alloc_size);
     //Kokkos::Experimental::print_pointer(i, pRef, "internal reduce (outer)" );
     //Kokkos::Experimental::print_pointer(i, &pRef[i%par_size], "internal reduce (inner)" );
     //printf("[%d] obtaining update reference %d, %d: %08x, offset %d\n", NODE_ID(), i, array_ndx, pRef, i%par_size);
     //reference_type lupdate = ValueInit::init(  pReducer->f , &pRef[i%par_size] );
     //pRef[i%par_size] = typename Analysis::value_type();
     reduction_type & lupdate = (reduction_type & )*(&pRef[reduce_off]);     
     //printf("[%d.%d] pointer node: %d \n", NODE_ID(), THREAD_ID(), mw_ptrtonodelet(&lupdate) );
     //Kokkos::Experimental::print_pointer(i, &lupdate, "entering inner loop" );
     
     for ( typename Policy::member_type j = (b+(m_policy_int_loop * i)); j < (b+( (m_policy_int_loop * i) + m_policy_int_loop)); j++ ) {
        if (j < e) {           
		   pReducer->init(lupdate);	   
           pReducer->f( t, (const typename Policy::member_type)j , lupdate );
           pReducer->join( lupdate );
        }
     }
  }
  
  void init_reducer(const size_t l_alloc_bytes, int i) const {
     cilk_reducer_wrapper* pH = get_reducer<cilk_reducer_wrapper>(global_reducer, i);         
	 typename cilk_reducer_wrapper::local_reducer_type* pLocalRed = get_reducer<typename cilk_reducer_wrapper::local_reducer_type>(local_reducer, i);
//	 printf("init reducer: %lx, %lx \n", (unsigned long)pH, (unsigned long)pLocalRed);
//	 fflush(stdout);
	 new (pLocalRed) NodeletReducer< typename cilk_reducer_wrapper::reduce_container >(global_reducer, i, cilk_reducer_wrapper::default_value());
     new (pH) cilk_reducer_wrapper(global_reducer, i, ReducerConditional::select(m_functor , m_reducer), l_alloc_bytes, pLocalRed);
  }

  void initialize_cilk_reducer(const size_t l_alloc_bytes) const
  {      
	  long * refPtr = Kokkos::Experimental::EmuReplicatedSpace::getRefAddr();	      
      //printf("reducer init: %lx, %lx, %lx \n", global_reducer, local_reducer, working_ptr);
      //fflush(stdout);
      for (int i = 0; i < Kokkos::Experimental::EmuReplicatedSpace::memory_zones(); i++) {
//		 printf("initializing reducer for node %d\n", i);		 
//		 fflush(stdout);
         cilk_spawn_at(&refPtr[i]) init_reducer(l_alloc_bytes, i);
      }
      cilk_sync;
      MIGRATE(&refPtr[0]);
  }

  template< class TagType >
  inline
  typename std::enable_if< std::is_same< TagType , void >::value >::type
  exec( reference_type update ) const
    {      
	  int nl_ = Kokkos::Experimental::EmuReplicatedSpace::memory_zones();
      //printf("void parallel reduce: %d \n", m_reduce_size);
      //fflush(stdout);
      
      const typename Policy::member_type e = m_policy.end();
      const typename Policy::member_type b = m_policy.begin();

      long * refPtr = Kokkos::Experimental::EmuReplicatedSpace::getRefAddr();
         
               
//      printf("parallel reduce: b= %d, e = %d, l = %d, par = %d, int = %d  %lx\n", b, e, len, par_loop, int_loop, w_ptr);
//      cilk_for (typename Policy::member_type i = 0 ; i < par_loop ; ++i ) {
//      }

      //printf("internal reduce size = %d ... spawning threads... \n", (int)l_alloc_bytes);
      //fflush(stdout);
      for (int i = 0; i < m_policy_par_loop; i++) {
 	     int node_ = i % nl_;
		 //printf("launch reduce at node %d \n", node_);
		 //fflush(stdout);
         cilk_spawn_at(&refPtr[node_]) this->template internal_reduce<TagType>(b, e, i, m_reduce_size * m_policy_par_size, nl_);
      }
      cilk_sync;
         
         
      cilk_reducer_wrapper* pReducerHost = get_reducer<cilk_reducer_wrapper>(global_reducer, 0);
      for (int i = 1; i < nl_; i++) {
		  cilk_reducer_wrapper* pReducerNode = get_reducer<cilk_reducer_wrapper>(global_reducer, i);
		  reduction_type lRef;
		  pReducerNode->update_value(lRef);
		  //printf("[%d] node value: %d \n", i, lRef);
		  pReducerHost->join(lRef);
	  }
         
      get_reducer<cilk_reducer_wrapper>(global_reducer, 0)->update_value(update);
      //printf("final node value: %d \n", update);
      for (int i = 1; i < nl_; i++) {
			 //printf("releasing resources: %d \n", i);
			 //fflush(stdout);
		 get_reducer<cilk_reducer_wrapper>(global_reducer, i)->release_resources();
	  }        
         
         /*printf("free working pointer reducer %lx \n", w_ptr); fflush(stdout);
         if (w_ptr != NULL) {
            mw_free((void*)w_ptr);
         }*/
      //mw_free((void*)working_ptr);
      //printf("free local reducer \n"); fflush(stdout);
      mw_free((void*)local_reducer);
      //printf("free global reducer \n"); fflush(stdout);
      mw_free((void*)global_reducer);
         
      //printf("reduction exec complete \n"); fflush(stdout);
  }

  template< class TagType >
  inline
  typename std::enable_if< ! std::is_same< TagType , void >::value >::type
  exec( reference_type update ) const
  {      
	  int nl_ = Kokkos::Experimental::EmuReplicatedSpace::memory_zones();
      //printf("worktag parallel reduce: %d \n", m_reduce_size);
      //fflush(stdout);      
      
      const typename Policy::member_type e = m_policy.end();
      const typename Policy::member_type b = m_policy.begin();

      long * refPtr = Kokkos::Experimental::EmuReplicatedSpace::getRefAddr();
      
	  //printf("internal reduce size = %d ... spawning threads... \n", (int)l_alloc_bytes);
	  //fflush(stdout);
	 
	  for (int i = 0; i < m_policy_par_loop; i++) {
		 int node_ = i % nl_;
		 //printf("launch reduce at node %d \n", node_);
		 //fflush(stdout);			 
		 cilk_spawn_at(&refPtr[node_]) this->template internal_reduce<TagType>(b, e, i, m_reduce_size*m_policy_par_size, nl_);
	  }
	  cilk_sync;
	  cilk_reducer_wrapper* pReducerHost = get_reducer<cilk_reducer_wrapper>(global_reducer, 0);
	  for (int i = 1; i < nl_; i++) {
	 	 cilk_reducer_wrapper* pReducerNode = get_reducer<cilk_reducer_wrapper>(global_reducer, i);
		 reduction_type lRef;
		 pReducerNode->update_value(lRef);
		 pReducerHost->join(lRef);
	  }
	 
	  get_reducer<cilk_reducer_wrapper>(global_reducer, 0)->update_value(update);
	 
	  for (int i = 1; i < Kokkos::Experimental::EmuReplicatedSpace::memory_zones(); i++) {
//			 printf("releasing resources: %d \n", i);
//			 fflush(stdout);
		 get_reducer<cilk_reducer_wrapper>(global_reducer, i)->release_resources();
 	  }        
	 
	  //mw_free((void*)working_ptr);
	  mw_free((void*)local_reducer);
	  mw_free((void*)global_reducer);
	   
   }

public:

  inline
  void execute() const
    {
      KOKKOS_ASSERT(global_reducer != NULL && "Global Reducer Check");
      KOKKOS_ASSERT(local_reducer != NULL && "Local Reducer Check");
      KOKKOS_ASSERT(working_ptr != NULL && "Local Reducer Check");
      
      const size_t team_reduce_size  = 0 ; // Never shrinks
      const size_t team_shared_size  = 0 ; // Never shrinks
      const size_t thread_local_size = 0 ; // Never shrinks

      serial_resize_thread_team_data( m_reduce_size
                                    , team_reduce_size
                                    , team_shared_size
                                    , thread_local_size );

      HostThreadTeamData & data = *serial_get_thread_team_data();

      //printf("Parallel reduce exec s = %d pl = %d, il = %d, ps = %d \n", m_policy_len, 
      //       m_policy_par_loop, m_policy_int_loop, m_policy_par_size);
      //fflush(stdout);
      pointer_type ptr =
        m_result_ptr ? m_result_ptr : pointer_type(data.pool_reduce_local());
//      printf("Parallel reduce ptr: %lx \n", ptr);
//      fflush(stdout);

      reference_type update =
        ValueInit::init(  ReducerConditional::select(m_functor , m_reducer) , ptr );
        
//      printf("parallel reduce calling internal exec \n");
//      fflush(stdout);
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
                                   sizeof(cilk_reducer_wrapper)
                                   ) 
                     )  
    , local_reducer ( mw_malloc2d(Kokkos::Experimental::EmuReplicatedSpace::memory_zones(), 
                                  sizeof(typename cilk_reducer_wrapper::local_reducer_type)
                                  )
                    ) 
    , working_ptr ( mw_malloc2d(Kokkos::Experimental::EmuReplicatedSpace::memory_zones(), 
                                m_policy_par_size * m_reduce_size
                                )
                  ) 
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
    , m_policy_len( get_policy_len() )
    , m_policy_par_loop( get_policy_par_loop() )
    , m_policy_int_loop( get_policy_int_loop() )
    , m_policy_par_size( get_policy_par_size() )    
    , m_reducer( reducer )
    , m_result_ptr(  reducer.view().data() )
    , m_reduce_size( get_reduce_size( m_functor, m_reducer ) )
    , global_reducer ( mw_malloc2d(Kokkos::Experimental::EmuReplicatedSpace::memory_zones(), sizeof(cilk_reducer_wrapper)) )  // one for each memory zone...
    , local_reducer ( mw_malloc2d(Kokkos::Experimental::EmuReplicatedSpace::memory_zones(), sizeof(typename cilk_reducer_wrapper::local_reducer_type)))     
    , working_ptr ( mw_malloc2d(Kokkos::Experimental::EmuReplicatedSpace::memory_zones(), m_policy_par_size * m_reduce_size )) 
    {
    }
};
}
}

#endif
