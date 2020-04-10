
#ifndef KOKKOS_CILK_REDUCER_H_
#define KOKKOS_CILK_REDUCER_H_

#include <cilk/cilk.h>
#include <cilk/reducer.h>
//Replace specific Emu headers with the tools header to allow x86 compilation
#include <emu_c_utils/emu_c_utils.h>
//#include <intrinsics.h>
//#include <pmanip.h>

namespace Kokkos {
namespace Impl {

template <class ReducerType>
ReducerType * get_reducer(const void * ptr, int refId) {
   ReducerType * pRet = (ReducerType *)mw_arrayindex((void*)ptr, refId, 
                      Kokkos::Experimental::EmuReplicatedSpace::memory_zones(),  sizeof(ReducerType) ); 
   return &pRet[0];
}

template <class ReduceWrapper, class ReducerType, class WorkTagFwd, class T = void>
struct CilkReduceContainer;

template <class ReduceWrapper, class ReducerType, class WorkTagFwd, class T = void>
struct CilkEmuReduceView;

template <class ReduceWrapper, class ReducerType, class WorkTagFwd>
struct CilkEmuReduceView<ReduceWrapper, ReducerType, WorkTagFwd, typename std::enable_if< Kokkos::is_reducer_type<ReducerType>::value ||
                                                                 Kokkos::is_view<ReducerType>::value>::type >
{

  typedef typename std::remove_reference<typename ReducerType::value_type>::type nr_value_type;
  typedef typename std::remove_pointer<nr_value_type>::type np_value_type;
  typedef typename std::remove_const<np_value_type>::type rd_value_type; 
  typedef Kokkos::Impl::FunctorValueJoin< ReducerType, WorkTagFwd >   ValueJoin;
  typedef Kokkos::Impl::FunctorValueInit< ReducerType, WorkTagFwd >   ValueInit;

  rd_value_type & val;


public:
  inline
  CilkEmuReduceView( const void * rp, int id, rd_value_type & val_ ) : val(val_) {
  }

  inline
  void identity( const void * rp, int id, rd_value_type * val ) {
     ReduceWrapper* ptr = get_reducer<ReduceWrapper>(rp, id);
     //printf("internal identity reduction value: %d, %lx, %d \n", id, ptr, *val);
     //fflush(stdout);
     if (ptr)
     {
        while (true) {
           if (Impl::lock_addr((unsigned long)val)) {
			   ValueInit::init( ptr->r, val );           
               Impl::unlock_addr((unsigned long)val);
               break;
            }
            Kokkos::Impl::emu_sleep((unsigned long)val);
        }		         
     }  
     //printf("after internal identity reduction value: %d, %lx, %d \n", id, ptr, *val);
     //fflush(stdout);
     
  }


  inline
  void join( const void * rp, int id, rd_value_type right ) {
     ReduceWrapper* ptr = get_reducer<ReduceWrapper>(rp, id);
     if (ptr)
     {
        while (true) {		 
//        printf("reducer view join (B): %ld, %ld \n", val, right);
           if (Impl::lock_addr((unsigned long)&val)) {
			   ValueJoin::join( ptr->r, &val, &right );
               Impl::unlock_addr((unsigned long)&val);
               break;
            }
            Kokkos::Impl::emu_sleep((unsigned long)&val);
        }
//        printf("reducer view join (A): %ld, %ld \n", val, right);
     }
  }

  inline
  static rd_value_type create( int id, rd_value_type val_ ) {
     return val_;
  }

  inline 
  static void destroy( int id, rd_value_type val_ ) {
  }
};


// Reducer/View access via value
template <class ReduceWrapper, class ReducerType, class WorkTagFwd>
struct CilkReduceContainer<ReduceWrapper, ReducerType, WorkTagFwd, typename std::enable_if< Kokkos::is_reducer_type<ReducerType>::value ||
                                                                 Kokkos::is_view<ReducerType>::value>::type >
{
  Kokkos::HostSpace space;
  enum { isPointer = 1 };
  typedef typename std::remove_reference<typename ReducerType::value_type>::type nr_value_type;
  typedef typename std::remove_pointer<nr_value_type>::type np_value_type;
  typedef typename std::remove_const<np_value_type>::type rd_value_type; 
  typedef Kokkos::Impl::FunctorValueJoin< ReducerType, WorkTagFwd >   ValueJoin;
  typedef Kokkos::Impl::FunctorValueInit< ReducerType, WorkTagFwd >   ValueInit;

  rd_value_type val;
  size_t alloc_bytes;

  typedef CilkEmuReduceView< ReduceWrapper, ReducerType, WorkTagFwd > ViewType;
  typedef rd_value_type ElementType;

  inline
  void identity( const void * rp, int id, rd_value_type * val )
  {
	 //printf("Reduce container identity: %d \n", id);
	 //fflush(stdout);
     ReduceWrapper* ptr = get_reducer<ReduceWrapper>(rp, id);
     if (ptr)
     {
        ValueInit::init( ptr->r, val );
        //printf("reducer init: %ld \n", *val);
     }
  }


  inline
  void reduce( const void * rp, int id, rd_value_type * left, rd_value_type const * right )
  {
	 //printf("Reduce container reduce: %d \n", id);
	 //fflush(stdout);
	  
     ReduceWrapper* ptr = get_reducer<ReduceWrapper>(rp, id);
     if (ptr)
     {
        //printf("[%d] reducer reduce: %ld, %ld \n", id, *left, *right);
        ValueJoin::join( ptr->r, left, right );
     }
  }

};

template <class ReduceWrapper, class ReducerType, class WorkTagFwd>
struct CilkEmuReduceView<ReduceWrapper, ReducerType, WorkTagFwd, typename std::enable_if< !Kokkos::is_reducer_type<ReducerType>::value &&
                                             !Kokkos::is_view<ReducerType>::value && 
                                           ( std::is_array< typename ReducerType::value_type >::value || 
                                             std::is_pointer< typename ReducerType::value_type >::value ) >::type >   {

  typedef typename std::remove_reference<typename ReducerType::value_type>::type nr_value_type;
  typedef typename std::remove_pointer<nr_value_type>::type np_value_type;
  typedef typename std::remove_extent<np_value_type>::type ne_value_type;
  typedef typename std::remove_const<ne_value_type>::type rd_value_type; 

  typedef Kokkos::Impl::FunctorValueJoin< ReducerType, WorkTagFwd >   ValueJoin;
  typedef Kokkos::Impl::FunctorValueInit< ReducerType, WorkTagFwd >   ValueInit;

  rd_value_type * val;

public:
  // need to allocate space for val as it is just a pointer, then 
  // copy *val_ 
  inline
  CilkEmuReduceView( const void * rp, int id, rd_value_type * val_ ) {
     //printf("array view constructor: %08x \n", val_);
     ReduceWrapper* ptr = get_reducer<ReduceWrapper>(rp, id);
     if (ptr)
     {        
        Kokkos::HostSpace space;
        val = (rd_value_type *)space.allocate( ptr->alloc_bytes );
        *val = *val_;
     } else {
        val = val_;
     }
  }
  
  inline
  void identity( const void * rp, int id, rd_value_type * val )
  {
     ReduceWrapper* ptr = get_reducer<ReduceWrapper>(rp, id);
     if (ptr)
     {
        ValueInit::init( ptr->r, val );
     }
  }  

  inline
  void join( const void * rp, int id, rd_value_type * right ) {
     ReduceWrapper* ptr = get_reducer<ReduceWrapper>(rp, id);
     if (ptr)
     {
        //printf("(B) array view join (P): %ld, %ld: %08x, %08x \n", val[0], right[0], val, right);
        ValueJoin::join( ptr->r, val, right );
        //printf("(A) array view join (P): %ld, %ld: %08x, %08x \n", val[0], right[0], val, right);
     }  
  }

  inline
  static rd_value_type * create( const void * rp, int id, rd_value_type * val_ ) {     
     rd_value_type * lVal = 0;
     ReduceWrapper* ptr = get_reducer<ReduceWrapper>(rp, id);
     if (ptr)
     {        
        Kokkos::HostSpace space;
        lVal = (rd_value_type *)space.allocate( ptr->alloc_bytes );
        *lVal = *val_;
     } else {
        lVal = val_;
     }
     //printf("create array view memory: %08x, %08x \n", val_, lVal);
     return lVal;
  }

  inline 
  static void destroy( const void * rp, int id, rd_value_type * val_ ) {
     //printf("array view destroy: %08x \n", val_);
     ReduceWrapper* ptr = get_reducer<ReduceWrapper>(rp, id);
     if (ptr)
     {        
        Kokkos::HostSpace space;
        space.deallocate( val_, ptr->alloc_bytes );
     }
  }

  inline ~CilkEmuReduceView() {
  }

};

// Functor with array/pointer
template <class ReduceWrapper, class ReducerType, class WorkTagFwd>
struct CilkReduceContainer<ReduceWrapper, ReducerType, WorkTagFwd, typename std::enable_if< !Kokkos::is_reducer_type<ReducerType>::value &&
                                             !Kokkos::is_view<ReducerType>::value && 
                                           ( std::is_array< typename ReducerType::value_type >::value || 
                                             std::is_pointer< typename ReducerType::value_type >::value ) >::type >   {

  typedef typename std::remove_reference<typename ReducerType::value_type>::type nr_value_type;
  typedef typename std::remove_pointer<nr_value_type>::type np_value_type;
  typedef typename std::remove_extent<np_value_type>::type ne_value_type;
  typedef typename std::remove_const<ne_value_type>::type rd_value_type; 
  typedef Kokkos::Impl::FunctorValueJoin< ReducerType, WorkTagFwd >   ValueJoin;
  typedef Kokkos::Impl::FunctorValueInit< ReducerType, WorkTagFwd >   ValueInit;

  typedef CilkEmuReduceView< ReduceWrapper, ReducerType, WorkTagFwd > ViewType;
  typedef rd_value_type * ElementType;

  rd_value_type * val;
  size_t alloc_bytes;

  inline
  void identity( const void * rp, int id, rd_value_type * val )
  {
 	 //printf("Reduce container identity: %d \n", id);
	 //fflush(stdout);

     ReduceWrapper* ptr = get_reducer<ReduceWrapper>(rp, id);
     if (ptr)
     {
        ValueInit::init( ptr->r, val );
     }
  }

  inline
  void reduce( const void * rp, int id, rd_value_type * left, rd_value_type const * right )
  {
	 printf("Reduce container reduce: %d \n", id);
	 fflush(stdout);
	  
     //printf("array reduce : %ld, %ld, %08x %08x \n", left[0], right[0], left, right);
     ReduceWrapper* ptr = get_reducer<ReduceWrapper>(rp, id);
     if (ptr)
     {
        ValueJoin::join( ptr->r, left, right );
     }
  }

};

// non-pointer, non-array view
template <class ReduceWrapper, class ReducerType, class WorkTagFwd>
struct CilkEmuReduceView<ReduceWrapper, ReducerType, WorkTagFwd, typename std::enable_if< !Kokkos::is_reducer_type<ReducerType>::value &&
                                             !Kokkos::is_view<ReducerType>::value && 
                                           ( !std::is_array< typename ReducerType::value_type >::value && 
                                             !std::is_pointer< typename ReducerType::value_type >::value ) >::type >   {

  typedef typename std::remove_reference<typename ReducerType::value_type>::type nr_value_type;
  typedef typename std::remove_pointer<nr_value_type>::type np_value_type;
  typedef typename std::remove_const<np_value_type>::type rd_value_type; 
  typedef Kokkos::Impl::FunctorValueJoin< ReducerType, WorkTagFwd >   ValueJoin;
  typedef Kokkos::Impl::FunctorValueInit< ReducerType, WorkTagFwd >  ValueInit;

  rd_value_type & val;

public:
  inline
  CilkEmuReduceView( const void * rp, int id, rd_value_type & val_ ) : val(val_) {
	  //printf("[%d] init reducer view: %lx \n", id, &val);
  }

  inline
  void identity( const void * rp, int id, rd_value_type * val )
  {
     ReduceWrapper* ptr = get_reducer<ReduceWrapper>(rp, id);
     if (ptr)
     {
        while (true) {
           if (Impl::lock_addr((unsigned long)val)) {
			   ValueInit::init( ptr->r, val );
               Impl::unlock_addr((unsigned long)val);
               break;
            }
            Kokkos::Impl::emu_sleep((unsigned long)val);
        }			         
     } else {
		 //printf("[%d] identity cannot get reducer ptr \n", id);
		 //fflush(stdout);
     }
  }
  
  inline
  void join( const void * rp, int id, rd_value_type right ) {
     ReduceWrapper* ptr = get_reducer<ReduceWrapper>(rp, id);
     if (ptr)
     {
        //printf("[%d] reducer view join (%lx): %d, %d \n", id, &val, val, right);
        //fflush(stdout);
        while (true) {
           if (Impl::lock_addr((unsigned long)&val)) {			   
			   ValueJoin::join( ptr->r, &val, &right );
               Impl::unlock_addr((unsigned long)&val);
               break;
            }
            Kokkos::Impl::emu_sleep((unsigned long)&val);
        }			                 
     }  
  }

  inline
  static rd_value_type create( const void * rp, int id, rd_value_type val_ ) {
     return val_;
  }

  inline 
  static void destroy( const void * rp, int id, rd_value_type val_ ) {
  }

};


// non-pointer, non-array moniod 
template <class ReduceWrapper, class ReducerType, class WorkTagFwd>
struct CilkReduceContainer<ReduceWrapper, ReducerType, WorkTagFwd, typename std::enable_if< !Kokkos::is_reducer_type<ReducerType>::value &&
                                             !Kokkos::is_view<ReducerType>::value && 
                                           ( !std::is_array< typename ReducerType::value_type >::value && 
                                             !std::is_pointer< typename ReducerType::value_type >::value ) >::type >   {

  typedef typename std::remove_reference<typename ReducerType::value_type>::type nr_value_type;
  typedef typename std::remove_pointer<nr_value_type>::type np_value_type;
  typedef typename std::remove_const<np_value_type>::type rd_value_type; 

  typedef CilkEmuReduceView< ReduceWrapper, ReducerType, WorkTagFwd > ViewType;
  typedef rd_value_type ElementType;

  typedef Kokkos::Impl::FunctorValueJoin< ReducerType, WorkTagFwd >   ValueJoin;
  typedef Kokkos::Impl::FunctorValueInit< ReducerType, WorkTagFwd >  ValueInit;

  inline
  void identity( const void * rp, int id, rd_value_type * val )
  {
	 //printf("Reduce container identity: %d \n", id);
	 //fflush(stdout);

     ReduceWrapper* ptr = get_reducer<ReduceWrapper>(rp, id);
     if (ptr)
     {
        ValueInit::init( ptr->r, val );
     }
  }

  inline
  void reduce( const void * rp, int id, rd_value_type * left, rd_value_type const * right )
  {
	 //printf("Reduce container reduce: %d \n", id);
	 //fflush(stdout);

     //printf("reducer - reduce (S): %ld, %ld [%d] \n", (*left).value[0], (*right).value[0], id);
     ReduceWrapper* ptr = get_reducer<ReduceWrapper>(rp, id);
     if (ptr)
     {
        ValueJoin::join( ptr->r, left, right );
     }
  }

};


template<class Mon> 
class NodeletReducer : public cilk::reducer<Mon> {
    // The underlying type. 
    typedef typename Mon::ElementType T;
    typedef typename Mon::ViewType View;
    int m_id;
    T myElt;
    View myView;

public:
    NodeletReducer(const void * rp_, int id, const T * initVal): m_id(id), myElt(*initVal),myView(rp_, id, *initVal) {
	}
    NodeletReducer(const void * rp_, int id, T initVal): m_id(id), myElt(initVal),myView(rp_, id, myElt) {
		//printf("NR: %d - %d \n", id, myElt);
		//fflush(stdout);
    }

    ~NodeletReducer() {}

    T get_value() {
        //Mon mono; 
        T reduced = myElt;
        //mono.reduce(m_id, &reduced, &(this->myElt));
        return reduced;
    }

    /** Replicated view on this replicated reducer's element. */
    virtual View* view() {
        return (&myView);
    }
}; // class 

template< typename F , typename = std::false_type >
struct lambda_only { using type = void ; };

template< typename F >
struct lambda_only
< F , typename std::is_same< typename F::value_type , void >::type >
{
  using type = typename F::value_type ; 
};

template <typename ReducerType, class Functor, class defaultType, class WorkTagFwd , class T = void>
struct kokkos_cilk_reducer;

template <typename ReducerType, class Functor, class defaultType, class WorkTagFwd >
struct kokkos_cilk_reducer< ReducerType, Functor, defaultType, WorkTagFwd , typename std::enable_if< 
                                                     Kokkos::is_reducer_type<ReducerType>::value ||
                                                     Kokkos::is_view<ReducerType>::value>::type > {

    typedef Kokkos::Impl::if_c< std::is_same< typename lambda_only< Functor >::type, void >::value, ReducerType, Functor> ReducerTypeCond;

    typedef typename ReducerTypeCond::type ReducerTypeFwd;

    typedef CilkReduceContainer< kokkos_cilk_reducer, typename Kokkos::Impl::if_c< 
                                           Kokkos::is_view<ReducerType>::value, Functor, ReducerType>::type, 
                                                           WorkTagFwd >    reduce_container;
                                                           
    typedef cilk::reducer < reduce_container > local_reducer_type;
    typedef Kokkos::Impl::FunctorValueJoin< typename Kokkos::Impl::if_c< 
                                           Kokkos::is_view<ReducerType>::value, Functor, ReducerType>::type, 
                                                           WorkTagFwd >    ValueJoin;

    typedef Kokkos::Impl::FunctorValueInit< typename Kokkos::Impl::if_c< 
                                           Kokkos::is_view<ReducerType>::value, Functor, ReducerType>::type, 
                                                           WorkTagFwd >    ValueInit;

    
    const int m_id;
    const ReducerType f;
    const ReducerType r;
    const size_t alloc_bytes;
    local_reducer_type * local_reducer = NULL;
    const void * global_reducer = NULL;

    kokkos_cilk_reducer (const void * gr_, const int id, const ReducerType & f_, const size_t l_alloc_bytes, void * ptr_reducer) :
                      m_id(id), f(f_), r(f_), alloc_bytes(l_alloc_bytes), 
                      local_reducer(reinterpret_cast<local_reducer_type *>(ptr_reducer)),
                      global_reducer(gr_) {        
    }

    void init( typename ReducerTypeFwd::value_type & ret ) {
        typename reduce_container::ViewType * cont = local_reducer->view();
        cont->identity( global_reducer, m_id, &ret );
    }  
   
    void join(typename ReducerTypeFwd::value_type & val_) {
        typename reduce_container::ViewType * cont = local_reducer->view();
        cont->join( global_reducer, m_id, val_ );
    }

    void update_value(typename ReducerTypeFwd::value_type & ret) {
       ret = local_reducer->get_value();
    }

    void release_resources() {
    }
    
    static typename ReducerTypeFwd::value_type default_value() {
		ReducerTypeFwd f_;
        typename ReducerTypeFwd::value_type lVal;
        ValueInit::init( f_, &lVal );
        return lVal;
	}

};

// case where the functor is actually a lambda -- implicit Add operation.
template <typename ReducerType, class Functor, class defaultType, class WorkTagFwd >
struct kokkos_cilk_reducer< ReducerType, Functor, defaultType, WorkTagFwd , typename std::enable_if< !Kokkos::is_reducer_type<ReducerType>::value &&
                                                                                        !Kokkos::is_view<ReducerType>::value &&
                                                                         std::is_same< typename lambda_only< Functor >::type, void >::value  >::type > {

    typedef CilkReduceContainer< kokkos_cilk_reducer, Kokkos::Experimental::Sum< defaultType >, void >   reduce_container;
    
    typedef Kokkos::Experimental::Sum< defaultType > ReducerTypeFwd;

    typedef Kokkos::Impl::FunctorValueJoin< ReducerTypeFwd, WorkTagFwd >  ValueJoin;
    typedef Kokkos::Impl::FunctorValueInit< ReducerTypeFwd, WorkTagFwd >  ValueInit;
    typedef cilk::reducer < reduce_container > local_reducer_type;

    const int m_id;
    defaultType local_value;
    const Functor f;
    const Kokkos::Experimental::Sum< defaultType > r;
    const size_t alloc_bytes;
    
    local_reducer_type * local_reducer = NULL;
    const void * global_reducer = NULL;

    kokkos_cilk_reducer (const void * gr_, const int id, const Functor & f_, const size_t l_alloc_bytes, void * ptr_reducer) : m_id(id), 
                                                                           local_value(0), 
                                                                           f(f_), 
                                                                           r(local_value), 
                                                                           alloc_bytes(l_alloc_bytes),
                                                                           local_reducer(reinterpret_cast<local_reducer_type *>(ptr_reducer)), 
                                                                           global_reducer(gr_) {
        //printf("constructing default scalar reducer (sum), size = %d , addr = %08x\n", (int)l_alloc_bytes, (unsigned long)this );        
        ValueInit::init( r, &local_value );
        //defaultType test_val;
        //update_value(test_val);
        //printf("[%d] local reducer init value: %d\n", id, test_val);
        //fflush(stdout);
    }
    
   void init( defaultType & ret ) {	    
        typename reduce_container::ViewType * cont = local_reducer->view();
        //printf("init reduction value: %lx, %d \n", cont, ret);
        //fflush(stdout);
        cont->identity( global_reducer, m_id, &ret );
   }    

    void join(defaultType & val_) {
        typename reduce_container::ViewType * cont = local_reducer->view();        
        cont->join( global_reducer, m_id, val_ );
        //printf("cilk reducer join: %d, %d \n", m_id, val_);
        //fflush(stdout);
    }

    void update_value(defaultType & ret) {
       ret = local_reducer->get_value();
    }

    void release_resources() {
    }
    
    static defaultType default_value() {
		defaultType val;
		Kokkos::Experimental::Sum< defaultType > r_(val);
        ValueInit::init( r_, &val );		
		return val;
    }


};


template <typename ReducerType, class Functor, class defaultType, class WorkTagFwd >
struct kokkos_cilk_reducer< ReducerType, Functor, defaultType, WorkTagFwd , typename std::enable_if< !Kokkos::is_reducer_type<ReducerType>::value &&
                                                                           !Kokkos::is_view<ReducerType>::value &&
                                           ( std::is_array< typename Functor::value_type >::value || 
                                             std::is_pointer< typename Functor::value_type >::value ) >::type > {

    typedef CilkReduceContainer< kokkos_cilk_reducer, Functor, WorkTagFwd > reduce_container;
    typedef cilk::reducer < reduce_container > local_reducer_type;
    typedef Functor ReducerTypeFwd;
    typedef Kokkos::Impl::FunctorValueJoin< Functor, WorkTagFwd >   ValueJoin;
    typedef Kokkos::Impl::FunctorValueInit< Functor, WorkTagFwd >  ValueInit;
  
    const int m_id;
    const Functor f;
    const Functor r;
    size_t alloc_bytes;

    typename reduce_container::rd_value_type * lVal = NULL;
    local_reducer_type * local_reducer = NULL;
    const void * global_reducer = NULL;

//    inline
//    kokkos_cilk_reducer & operator = ( const kokkos_cilk_reducer & rhs ) { 
//       f = rhs.f ; alloc_bytes = rhs.alloc_bytes ; lVal = rhs.lVal ; local_reducer = rhs.local_reducer ; return *this ; }

    kokkos_cilk_reducer (const void * gr_, const int id, const Functor & f_, const size_t l_alloc_bytes, void* ptr_reducer) : m_id(id), 
                                                                           f(f_), r(f_), alloc_bytes(l_alloc_bytes),
                                                                           local_reducer(reinterpret_cast<local_reducer_type *>(ptr_reducer)),
                                                                           global_reducer(gr_) {
        //printf("constructing reducer for array %d \n", id);
																			   
        uint64_t myRed = (long) this;
        this->lVal = (typename reduce_container::rd_value_type *)mw_localmalloc( l_alloc_bytes, (void*)myRed );

        // construct functor on nodelet
        new ((void *)&(this->f)) Functor(f_);
        ValueInit::init( this->f, this->lVal );
    }

   void init( typename reduce_container::rd_value_type * ret ) {
        typename reduce_container::ViewType * cont = local_reducer->view();
        cont->identity( global_reducer, m_id, ret );
   }
   
    void join(typename reduce_container::rd_value_type * val_) {
        
        typename reduce_container::ViewType * cont = local_reducer->view();
        cont->join( global_reducer, m_id, val_ );
    }

    void update_value(typename reduce_container::rd_value_type * ret) {
       typename reduce_container::rd_value_type * lRet = local_reducer->get_value();
       *ret = *lRet;
    }

    void release_resources() {
    }
    
    static typename reduce_container::rd_value_type default_value() {
        typename Functor::value_type lVal;
        Functor f_;
        ValueInit::init( f_, &lVal );
        return static_cast<typename reduce_container::rd_value_type>(lVal);
	}
    
};


// non-pointer non-array -- could be struct or scalar
template <typename ReducerType, class Functor, class defaultType, class WorkTagFwd >
struct kokkos_cilk_reducer< ReducerType, Functor, defaultType, WorkTagFwd , typename std::enable_if< !Kokkos::is_reducer_type<ReducerType>::value &&
                                                                           !Kokkos::is_view<ReducerType>::value &&
                                           ( !std::is_array< typename Functor::value_type >::value &&
                                             !std::is_pointer< typename Functor::value_type >::value ) >::type >   {

    typedef CilkReduceContainer< kokkos_cilk_reducer, ReducerType, WorkTagFwd > reduce_container;
    typedef cilk::reducer < reduce_container > local_reducer_type;
    typedef Functor ReducerTypeFwd;
    typedef Kokkos::Impl::FunctorValueJoin< Functor, WorkTagFwd >  ValueJoin;
    typedef Kokkos::Impl::FunctorValueInit< Functor, WorkTagFwd >  ValueInit;

    const int m_id;
    const Functor f;
    const Functor r;
    const size_t alloc_bytes;
    local_reducer_type * local_reducer = NULL;    
    const void * global_reducer= NULL;

    kokkos_cilk_reducer (const void * gr_, const int id, const Functor & f_, const size_t l_alloc_bytes, void * ptr_reducer) : m_id(id), 
                                                                           f(f_), r(f_), alloc_bytes(l_alloc_bytes),
                                                                           local_reducer(reinterpret_cast<local_reducer_type *>(ptr_reducer)),
                                                                           global_reducer(gr_) {
		typename Functor::value_type  test_val;
		update_value(test_val);
		//printf("[%d] cilk reducer (with functor) init val = %d \n", id, test_val);
		//fflush(stdout);
    }

   void init( typename Functor::value_type & ret ) {
        typename reduce_container::ViewType * cont = local_reducer->view();
        cont->identity( global_reducer, m_id, &ret );
   }

   void join(typename Functor::value_type & val_) {
        typename reduce_container::ViewType * cont = local_reducer->view();
        cont->join( global_reducer, m_id, val_ );
    }
    
    void update_value(typename Functor::value_type & ret) {
        ret = local_reducer->get_value();
    }

    void release_resources() {
    }
    
    static typename Functor::value_type default_value () {
        typename Functor::value_type lVal;
        Functor f_;
        ValueInit::init( f_, &lVal );
        return lVal;
    }


};

} 
}

#define INITIALIZE_CILK_REDUCER( wrapper, reducer ) {}


#endif
