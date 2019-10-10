/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/
#define KOKKOS_LOCK_FIRST
#include <Kokkos_Core.hpp>

#if defined(KOKKOS_ENABLE_EMU)
   #include <memoryweb.h>
   #include <repl.h>
   #include <pmanip.h>   
#endif

namespace Kokkos {
namespace Impl {
	
#define LIST_LOCK(A) (((long*)list_lock)[A])
#define CUR_LOCK_PTR(A) (((long*)cur_lock_ptr)[A])
#define LIST_HEAD(A) (((long*)list_head)[A])
#define LOCK_LIST(A) (((long**)lockList)[A][0])
#define LAST_LOCK(A) (((long*)lastLock)[A])
	
void emu_sleep( unsigned long ms ) {
	RESCHEDULE();
}

replicated long list_lock;         // mutex / lock for list
replicated long cur_lock_ptr;      // index of next available lock in lockList
replicated long list_head;         // pointer to first lock in linked list
replicated long lockList;          // strided storage location for locks pool
replicated long lastLock;          // strided storage location for locks pool

// initialize lock vars on particular node
void init_locks( int i ) {
	ATOMIC_SWAP(&LIST_LOCK(i), 0);
	CUR_LOCK_PTR(i) = KOKKOS_MEMORY_LOCK_LEN;
	LIST_HEAD(i) = NULL;
	LAST_LOCK(i) = NULL;
			
	for ( int r = 0; r < KOKKOS_MEMORY_LOCK_LEN; r++ ) {
       ((AddrLock*)&LOCK_LIST(i))[r].id = 0;
       ((AddrLock*)&LOCK_LIST(i))[r].pNext = NULL;
    }
}

/*
 * Single lock variable per nodelet 
 *   lock list is linked list where the head is striped memory, and the
 *   nodes are local to the striped head.
 * 
 */
void initialize_memory_locks() {
	int nCnt = NODELETS();
	long * ll_ = mw_malloc1dlong(nCnt);
	long * clp_ = mw_malloc1dlong(nCnt);
	long * lh_ = mw_malloc1dlong(nCnt);
	long * llock_ = mw_malloc1dlong(nCnt);
	long ** lol_ = (long**)mw_malloc2d(nCnt, KOKKOS_MEMORY_LOCK_LEN * sizeof(AddrLock));
	
	mw_replicated_init( &list_lock, (long)ll_ );
	mw_replicated_init( &cur_lock_ptr, (long)clp_ );
	mw_replicated_init( &list_head, (long)lh_ );
	mw_replicated_init( &lockList, (long)lol_ );
	mw_replicated_init( &lastLock, (long)llock_ );
			
	for ( int i = 0; i < nCnt; i++) {
		cilk_spawn_at(&ll_[i]) init_locks( i );
    }
    cilk_sync;
}

// Lock mutex for this node
bool get_lock(int i, unsigned long addr) {
   if (ATOMIC_CAS(&LIST_LOCK(i), 1L, 0L) == 0L) { 
	  LAST_LOCK(i) = addr;
      return true;
   }
   return false;
}

// free mutext for this node
bool free_lock(int i) {
   LAST_LOCK(i) = NULL;
   ATOMIC_SWAP(&LIST_LOCK(i), 0);    
   return true;
}

// if its in the list, it's locked...
bool lock_addr( unsigned long ad ) {
  unsigned long addr = ad;
  int nNode = mw_ptrtonodelet( addr );  
  bool bFound = true;
  if ( get_lock(nNode, addr) ) {	 
	 bFound = false;
	 //printf("lock obtained, searching for address in lock list: 0x%lx, %d \n", addr, nNode);
	 //fflush(stdout);
	 AddrLock * pWork = (AddrLock*)LIST_HEAD(nNode);
	 int curSearch = 0;
	 while (pWork != NULL && curSearch < KOKKOS_MEMORY_LOCK_LEN) {
		 if (pWork->id == addr) {
			 bFound = true;
			 break;
		 }
		 pWork = pWork->pNext;
		 curSearch++;
	 }
	 if (curSearch >= KOKKOS_MEMORY_LOCK_LEN) {
		 //printf("there is a circular situation in the lock list: %08x, %d \n", addr, nNode);
		 // force this to look like it couldn't get a lock
		 bFound = true;
	 }
	 
	 if (!bFound) {
		if (CUR_LOCK_PTR(nNode) > 0) {
			CUR_LOCK_PTR(nNode)--;
            AddrLock * ptr = &(((AddrLock*)&LOCK_LIST(nNode))[CUR_LOCK_PTR(nNode)]);
            ptr->id = addr;
            ptr->pNext = (AddrLock*)LIST_HEAD(nNode);
		    LIST_HEAD(nNode) = (long)ptr;
		 } else {
			bFound = true; // set this to true...if we can't find an open slot, we want to return false;
            for (int r = 0; r < KOKKOS_MEMORY_LOCK_LEN; r++) {
				if ( ((AddrLock*)&LOCK_LIST(nNode))[r].id == NULL ) {
                   AddrLock * ptr = &(((AddrLock*)&LOCK_LIST(nNode))[r]);
                   ptr->id = addr;
		           ptr->pNext = (AddrLock*)LIST_HEAD(nNode);
		           LIST_HEAD(nNode) = (long)ptr;		           
		           bFound = false;
		           break;		
				}
			}
			if (bFound) {
				printf("lock address [%08x] can't find empty slot...\n", addr );
			}
		 }		 
	 }
	 //printf("freeing lock list: [%08x] node locked successfull = %s \n", addr, (!bFound) ? "true" : "false");
     free_lock(nNode);
     //fflush(stdout);
  } else {
	  //printf("lock address [0x%lx] can't lock list 0x%lx...\n", addr, LAST_LOCK(nNode) );
	  //fflush(stdout);
  }
  return !bFound;
}

void free_node( int i, AddrLock * pWork) {
  
  if ( CUR_LOCK_PTR(i) >= 0 ) {
	  if ( ((AddrLock*)&LOCK_LIST(i))[CUR_LOCK_PTR(i)].id == pWork->id ) {
		  CUR_LOCK_PTR(i)++;
		  while ( CUR_LOCK_PTR(i) < KOKKOS_MEMORY_LOCK_LEN && 
		           ((AddrLock*)&LOCK_LIST(i))[CUR_LOCK_PTR(i)].id == 0 ) {
			  //printf("purging record from unused list...%d\n", CUR_LOCK_PTR(i));			  
			  if (LIST_HEAD(i) == (long)&(((AddrLock*)&LOCK_LIST(i))[CUR_LOCK_PTR(i)])) {
				  if ((CUR_LOCK_PTR(i)+1) < KOKKOS_MEMORY_LOCK_LEN) {
                      //printf("repositioning lock head: %d, %d 0x%lx, new head 0x%lx \n", i, CUR_LOCK_PTR(i), LIST_HEAD(i), 
                      //          (long)&(((AddrLock*)&LOCK_LIST(i))[CUR_LOCK_PTR(i)+1]));
				      //fflush(stdout);				  					  
					  LIST_HEAD(i) = (long)&(((AddrLock*)&LOCK_LIST(i))[CUR_LOCK_PTR(i)+1]);
				  } else {
                      //printf("repositioning lock head: %d, %d 0x%lx, new head NULL \n", i, CUR_LOCK_PTR(i), LIST_HEAD(i));
					  LIST_HEAD(i) = NULL;					  
			      }
			  }
	          CUR_LOCK_PTR(i)++;
		  }
	  }
  }
  pWork->id = 0;
  pWork->pNext = NULL;  
}

// take it out of the list to free the lock...
void unlock_addr( unsigned long ad ) {
  unsigned long addr = ad;
  int nNode = mw_ptrtonodelet( addr );  
  bool bLock = get_lock(nNode, addr);  // this has to get a lock...we will wait.
  while( !bLock  )  {
	  emu_sleep(10);
	  bLock = get_lock(nNode, addr);
  }
  if ( bLock ) {	 
	 bool bFound = false;
	 AddrLock * pWork = (AddrLock*)LIST_HEAD(nNode);
	 if (pWork != NULL && pWork->id == addr) {		 
		 LIST_HEAD(nNode) = (long)pWork->pNext;
		 //printf("[%d] unload addr 0x%lx, new head 0x%lx \n", nNode, addr, LIST_HEAD(nNode));
		 //fflush(stdout);
		 free_node(nNode, pWork);
		 bFound = true;
	 } else {
	   while (pWork != NULL && pWork->pNext != NULL) {
		 if (pWork->pNext->id == addr) {
			 AddrLock * pTemp = pWork->pNext;
			 pWork->pNext = pTemp->pNext;
			 free_node(nNode, pTemp);
			 bFound = true;
			 break;
		 }
		 pWork = pWork->pNext;
	   }
	 }
     free_lock(nNode);
  }
}


KOKKOS_THREAD_PREFIX int SharedAllocationRecord<void, void>::t_tracking_enabled = 1;

#ifdef KOKKOS_DEBUG
bool
SharedAllocationRecord< void , void >::
is_sane( SharedAllocationRecord< void , void > * arg_record )
{
  SharedAllocationRecord * const root = arg_record ? arg_record->m_root : 0 ;

  bool ok = root != 0 && root->use_count() == 0 ;

  if ( ok ) {
    SharedAllocationRecord * root_next = 0 ;
    static constexpr SharedAllocationRecord * zero = nullptr;
    // Lock the list:
    while ( ( root_next = Kokkos::atomic_exchange( & root->m_next , zero ) ) == nullptr );

    for ( SharedAllocationRecord * rec = root_next ; ok && rec != root ; rec = rec->m_next ) {
      const bool ok_non_null  = rec && rec->m_prev && ( rec == root || rec->m_next );
      const bool ok_root      = ok_non_null && rec->m_root == root ;
      const bool ok_prev_next = ok_non_null && ( rec->m_prev != root ? rec->m_prev->m_next == rec : root_next == rec );
      const bool ok_next_prev = ok_non_null && rec->m_next->m_prev == rec ;
      const bool ok_count     = ok_non_null && 0 <= rec->use_count() ;

      ok = ok_root && ok_prev_next && ok_next_prev && ok_count ;

      if ( ! ok ) {
        //Formatting dependent on sizeof(uintptr_t)
        const char * format_string;

        if (sizeof(uintptr_t) == sizeof(unsigned long)) {
          format_string = "Kokkos::Impl::SharedAllocationRecord failed is_sane: rec(0x%.12lx){ m_count(%d) m_root(0x%.12lx) m_next(0x%.12lx) m_prev(0x%.12lx) m_next->m_prev(0x%.12lx) m_prev->m_next(0x%.12lx) }\n";
        }
        else if (sizeof(uintptr_t) == sizeof(unsigned long long)) {
          format_string = "Kokkos::Impl::SharedAllocationRecord failed is_sane: rec(0x%.12llx){ m_count(%d) m_root(0x%.12llx) m_next(0x%.12llx) m_prev(0x%.12llx) m_next->m_prev(0x%.12llx) m_prev->m_next(0x%.12llx) }\n";
        }

        fprintf(stderr
            , format_string
            , reinterpret_cast< uintptr_t >( rec )
            , rec->use_count()
            , reinterpret_cast< uintptr_t >( rec->m_root )
            , reinterpret_cast< uintptr_t >( rec->m_next )
            , reinterpret_cast< uintptr_t >( rec->m_prev )
            , reinterpret_cast< uintptr_t >( rec->m_next != NULL ? rec->m_next->m_prev : NULL )
            , reinterpret_cast< uintptr_t >( rec->m_prev != rec->m_root ? rec->m_prev->m_next : root_next )
            );
      }

    }

    if ( nullptr != Kokkos::atomic_exchange( & root->m_next , root_next ) ) {
      Kokkos::Impl::throw_runtime_exception("Kokkos::Impl::SharedAllocationRecord failed is_sane unlocking");
    }
  }
  return ok ;
}

#else

bool
SharedAllocationRecord< void , void >::
is_sane( SharedAllocationRecord< void , void > * )
{
  Kokkos::Impl::throw_runtime_exception("Kokkos::Impl::SharedAllocationRecord::is_sane only works with KOKKOS_DEBUG enabled");
  return false ;
}
#endif //#ifdef KOKKOS_DEBUG

#ifdef KOKKOS_DEBUG
SharedAllocationRecord<void,void> *
SharedAllocationRecord<void,void>::find( SharedAllocationRecord<void,void> * const arg_root , void * const arg_data_ptr )
{
  SharedAllocationRecord * root_next = 0 ;
  static constexpr SharedAllocationRecord * zero = nullptr;

  // Lock the list:
  while ( ( root_next = Kokkos::atomic_exchange( & arg_root->m_next , zero ) ) == nullptr );

  // Iterate searching for the record with this data pointer

  SharedAllocationRecord * r = root_next ;

  while ( ( r != arg_root ) && ( r->data() != arg_data_ptr ) ) { r = r->m_next ; }

  if ( r == arg_root ) { r = 0 ; }

  if ( nullptr != Kokkos::atomic_exchange( & arg_root->m_next , root_next ) ) {
    Kokkos::Impl::throw_runtime_exception("Kokkos::Impl::SharedAllocationRecord failed locking/unlocking");
  }
  return r ;
}
#else
SharedAllocationRecord<void,void> *
SharedAllocationRecord<void,void>::find( SharedAllocationRecord<void,void> * const , void * const )
{
  Kokkos::Impl::throw_runtime_exception("Kokkos::Impl::SharedAllocationRecord::find only works with KOKKOS_DEBUG enabled");
  return nullptr;
}
#endif


/**\brief  Construct and insert into 'arg_root' tracking set.
 *         use_count is zero.
 */
SharedAllocationRecord< void , void >::
SharedAllocationRecord(
#ifdef KOKKOS_DEBUG
                        SharedAllocationRecord<void,void> * arg_root,
#endif
                        SharedAllocationHeader            * arg_alloc_ptr
                      , size_t                              arg_alloc_size
                      , SharedAllocationRecord< void , void >::function_type  arg_dealloc
                      )
  : m_alloc_ptr(  arg_alloc_ptr )
  , m_alloc_size( arg_alloc_size )
  , m_dealloc(    arg_dealloc )
#ifdef KOKKOS_DEBUG
  , m_root( arg_root )
  , m_prev( 0 )
  , m_next( 0 )
#endif
  , m_count( 0 )
  , m_custom_inc( nullptr )
  , m_custom_dec( nullptr )
{
  
  if ( 0 != m_alloc_ptr ) {

#ifdef KOKKOS_DEBUG
    // Insert into the root double-linked list for tracking
    //
    // before:  arg_root->m_next == next ; next->m_prev == arg_root
    // after:   arg_root->m_next == this ; this->m_prev == arg_root ;
    //              this->m_next == next ; next->m_prev == this

    m_prev = m_root ;
    static constexpr SharedAllocationRecord * zero = nullptr;

    // Read root->m_next and lock by setting to NULL
    while ( ( m_next = Kokkos::atomic_exchange( & m_root->m_next , zero ) ) == nullptr );

    m_next->m_prev = this ;

    // memory fence before completing insertion into linked list
    Kokkos::memory_fence();

    if ( nullptr != Kokkos::atomic_exchange( & m_root->m_next , this ) ) {
      Kokkos::Impl::throw_runtime_exception("Kokkos::Impl::SharedAllocationRecord failed locking/unlocking");
    }
#endif

  }
  else {
    Kokkos::Impl::throw_runtime_exception("Kokkos::Impl::SharedAllocationRecord given NULL allocation");
  }
}

void
SharedAllocationRecord< void , void >::
increment( Kokkos::Impl::SharedAllocationRecord< void , void > * arg_record )
{
  const int old_count = Kokkos::atomic_fetch_add( & arg_record->m_count , 1 );

  if ( old_count < 0 ) { // Error
    Kokkos::Impl::throw_runtime_exception("Kokkos::Impl::SharedAllocationRecord failed increment");
  }
}

void 
SharedAllocationRecord< void , void >::
custom_increment( Kokkos::Impl::SharedAllocationRecord< void , void > * rec) {
   if ( rec->m_custom_inc != nullptr ) {
//	   printf("calling custom increment...\n");
//	   fflush(stdout);
       rec->m_custom_inc( (void*)rec );
   } else {
//	  printf("calling default increment because no custom assigned\n");
//	  fflush(stdout);
      SharedAllocationRecord< void , void >::increment( rec );
   }  
//   printf("this has nothing to do with increment :) \n");
   fflush(stdout);
}


SharedAllocationRecord< void , void > *
SharedAllocationRecord< void , void >::
custom_decrement( Kokkos::Impl::SharedAllocationRecord< void , void > * rec) {
   if ( rec->m_custom_dec != nullptr ) {
        //printf("calling m_custom_dec\n");
        //fflush(stdout);
        return (SharedAllocationRecord< void , void > *)rec->m_custom_dec( (void*)rec );
   } else {
      //printf("calling decrement\n");
      //fflush(stdout);
      return SharedAllocationRecord< void , void >::decrement(rec);
   }
}



SharedAllocationRecord< void , void > *
SharedAllocationRecord< void , void >::
decrement( SharedAllocationRecord< void , void > * arg_record )
{
  const int old_count = Kokkos::atomic_fetch_sub( & arg_record->m_count , 1 );

  if ( old_count == 1 ) {

    if (!Kokkos::is_initialized()) {
      std::stringstream ss;
      ss << "Kokkos allocation \"";
      ss << arg_record->get_label();
      ss << "\" is being deallocated after Kokkos::finalize was called\n";
      auto s = ss.str();
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
      std::cerr << s;
      std::cerr << "This behavior is incorrect Kokkos usage, and will crash in future releases\n";
#else
      Kokkos::Impl::throw_runtime_exception(s);
#endif
    }

#ifdef KOKKOS_DEBUG
    // before:  arg_record->m_prev->m_next == arg_record  &&
    //          arg_record->m_next->m_prev == arg_record
    //
    // after:   arg_record->m_prev->m_next == arg_record->m_next  &&
    //          arg_record->m_next->m_prev == arg_record->m_prev

    SharedAllocationRecord * root_next = 0 ;
    static constexpr SharedAllocationRecord * zero = nullptr;

    // Lock the list:
    while ( ( root_next = Kokkos::atomic_exchange( & arg_record->m_root->m_next , zero ) ) == nullptr );

    arg_record->m_next->m_prev = arg_record->m_prev ;

    if ( root_next != arg_record ) {
      arg_record->m_prev->m_next = arg_record->m_next ;
    }
    else {
      // before:  arg_record->m_root == arg_record->m_prev
      // after:   arg_record->m_root == arg_record->m_next
      root_next = arg_record->m_next ;
    }

    Kokkos::memory_fence();

    // Unlock the list:
    if ( nullptr != Kokkos::atomic_exchange( & arg_record->m_root->m_next , root_next ) ) {
      Kokkos::Impl::throw_runtime_exception("Kokkos::Impl::SharedAllocationRecord failed decrement unlocking");
    }

    arg_record->m_next = 0 ;
    arg_record->m_prev = 0 ;
#endif

    function_type d = arg_record->m_dealloc ;
    if (d != nullptr) {
       //printf("calling dealloc functor: %s \n", arg_record->get_label().c_str());
       //fflush(stdout);
		
       //(*d)( arg_record );
       arg_record = 0 ;
    } else {
       //printf("record count is zero but there is nothing to deallocate: %s \n", arg_record->get_label().c_str());
       //fflush(stdout);		
	}
  }
  else if ( old_count < 1 ) { // Error
    fprintf(stderr,"Kokkos::Impl::SharedAllocationRecord '%s' failed decrement count = %d\n", arg_record->m_alloc_ptr->m_label , old_count );
    fflush(stderr);
    Kokkos::Impl::throw_runtime_exception("Kokkos::Impl::SharedAllocationRecord failed decrement count");
  }

  return arg_record ;
}

void
SharedAllocationRecord< void , void >::
print_host_accessible_records( std::ostream & s
                             , const char * const space_name
                             , const SharedAllocationRecord * const root
                             , const bool detail )
{
#ifdef KOKKOS_DEBUG
  const SharedAllocationRecord< void , void > * r = root ;

  char buffer[256] ;

  if ( detail ) {
    do {
      //Formatting dependent on sizeof(uintptr_t)
      const char * format_string;

      if (sizeof(uintptr_t) == sizeof(unsigned long)) {
        format_string = "%s addr( 0x%.12lx ) list( 0x%.12lx 0x%.12lx ) extent[ 0x%.12lx + %.8ld ] count(%d) dealloc(0x%.12lx) %s\n";
      }
      else if (sizeof(uintptr_t) == sizeof(unsigned long long)) {
        format_string = "%s addr( 0x%.12llx ) list( 0x%.12llx 0x%.12llx ) extent[ 0x%.12llx + %.8ld ] count(%d) dealloc(0x%.12llx) %s\n";
      }

      snprintf( buffer , 256
              , format_string
              , space_name
              , reinterpret_cast<uintptr_t>( r )
              , reinterpret_cast<uintptr_t>( r->m_prev )
              , reinterpret_cast<uintptr_t>( r->m_next )
              , reinterpret_cast<uintptr_t>( r->m_alloc_ptr )
              , r->m_alloc_size
              , r->use_count()
              , reinterpret_cast<uintptr_t>( r->m_dealloc )
              , r->m_alloc_ptr->m_label
              );
      s << buffer ;
      r = r->m_next ;
    } while ( r != root );
  }
  else {
    do {
      if ( r->m_alloc_ptr ) {
        //Formatting dependent on sizeof(uintptr_t)
        const char * format_string;

        if (sizeof(uintptr_t) == sizeof(unsigned long)) {
          format_string = "%s [ 0x%.12lx + %ld ] %s\n";
        }
        else if (sizeof(uintptr_t) == sizeof(unsigned long long)) {
          format_string = "%s [ 0x%.12llx + %ld ] %s\n";
        }

        snprintf( buffer , 256
                , format_string
                , space_name
                , reinterpret_cast< uintptr_t >( r->data() )
                , r->size()
                , r->m_alloc_ptr->m_label
                );
      }
      else {
        snprintf( buffer , 256 , "%s [ 0 + 0 ]\n" , space_name );
      }
      s << buffer ;
      r = r->m_next ;
    } while ( r != root );
  }
#else
  Kokkos::Impl::throw_runtime_exception(
      "Kokkos::Impl::SharedAllocationRecord::print_host_accessible_records"
      " only works with KOKKOS_DEBUG enabled");
#endif
}

} /* namespace Impl */
} /* namespace Kokkos */

