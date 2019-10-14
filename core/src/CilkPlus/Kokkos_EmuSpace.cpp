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
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EMU

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <atomic>

#define COMPILE_EMU_SPACE

#include <Kokkos_Core.hpp>
#include <Kokkos_CilkPlus.hpp>
#include <Kokkos_EmuSpace.hpp>
#include <impl/Kokkos_Error.hpp>

#include <intrinsics.h>

struct emu_pointer {
    uint64_t view;
    uint64_t node_id;
    uint64_t nodelet_id;
    uint64_t nodelet_addr;
    uint64_t byte_offset;
};

extern "C"{
   struct emu_pointer
      examine_emu_pointer(void * ptr);
}

#if defined(KOKKOS_ENABLE_PROFILING)
#include <impl/Kokkos_Profiling_Interface.hpp>
#endif


/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

DeepCopy<Kokkos::Experimental::EmuLocalSpace,Kokkos::Experimental::EmuLocalSpace,Kokkos::Experimental::CilkPlus>::DeepCopy( void * dst , const void * src , size_t n )
{ }

DeepCopy<Kokkos::Experimental::EmuLocalSpace,Kokkos::Experimental::EmuLocalSpace,Kokkos::Experimental::CilkPlus>::DeepCopy( const Kokkos::Experimental::CilkPlus & instance , void * dst , const void * src , size_t n )
{  }

DeepCopy<Kokkos::Experimental::EmuLocalSpace,HostSpace,Kokkos::Experimental::CilkPlus>::DeepCopy( void * dst , const void * src , size_t n )
{  }

DeepCopy<Kokkos::Experimental::EmuLocalSpace,HostSpace,Kokkos::Experimental::CilkPlus>::DeepCopy( const Kokkos::Experimental::CilkPlus & instance , void * dst , const void * src , size_t n )
{  }

DeepCopy<HostSpace,Kokkos::Experimental::EmuLocalSpace,Kokkos::Experimental::CilkPlus>::DeepCopy( void * dst , const void * src , size_t n )
{  }

DeepCopy<HostSpace,Kokkos::Experimental::EmuLocalSpace,Kokkos::Experimental::CilkPlus>::DeepCopy( const Kokkos::Experimental::CilkPlus & instance , void * dst , const void * src , size_t n )
{  }

void DeepCopyAsyncEmu( void * dst , const void * src , size_t n) {
}

} // namespace Impl
} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/


namespace Kokkos {
namespace Experimental {
	
	
void print_pointer( int i, void* ptr, const char * name ) {
    //emu_pointer pchk = examine_emu_pointer(ptr);
    //printf("'%s-%d' st: %ld, view: %ld \n", name, i, pchk.nodelet_id, pchk.view);
}

void EmuLocalSpace::access_error()
{
  const std::string msg("Kokkos::Experimental::EmuLocalSpace::access_error attempt to execute Emu function from non-Emu space" );
  Kokkos::Impl::throw_runtime_exception( msg );
}

void EmuLocalSpace::access_error( const void * const )
{
  const std::string msg("Kokkos::Experimental::EmuLocalSpace::access_error attempt to execute Emu function from non-Emu space" );
  Kokkos::Impl::throw_runtime_exception( msg );
}


} // namespace Experimental
} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Experimental {

static long * ref_ptr = 0;

static long * getRefPtr() {
   if ( ref_ptr == 0) {
      ref_ptr = mw_malloc1dlong( NODELETS() );
   }
   return ref_ptr;
}

EmuLocalSpace::EmuLocalSpace()
{
//printf("construct emu local \n");
//fflush(stdout);
}

EmuReplicatedSpace::EmuReplicatedSpace()
{
}

long * EmuReplicatedSpace::getRefAddr()
{ return getRefPtr(); }

int EmuReplicatedSpace::memory_zones() 
{ return NODELETS(); }

int EmuStridedSpace::memory_zones() 
{ return NODELETS(); }

int EmuLocalSpace::memory_zones() 
{ return NODELETS(); }

EmuStridedSpace::EmuStridedSpace()
{
}


void EmuReplicatedSpace::custom_increment( void * pRec ) {   
   Kokkos::Impl::SharedAllocationRecord< Kokkos::Experimental::EmuReplicatedSpace , void >::
               custom_increment((Kokkos::Impl::SharedAllocationRecord<void,void> *)pRec);
}


void *
EmuReplicatedSpace::custom_decrement( void * pRec )
{
//   printf("calling AllSh<EmuRepl> custom decrement\n");
//   fflush(stdout);
   return (void*)Kokkos::Impl::SharedAllocationRecord< Kokkos::Experimental::EmuReplicatedSpace , void >::
              custom_decrement( (Kokkos::Impl::SharedAllocationRecord<void,void> *)pRec );
}

void EmuStridedSpace::custom_increment( void * pRec ) {   
   Kokkos::Impl::SharedAllocationRecord< Kokkos::Experimental::EmuStridedSpace , void >::
               custom_increment((Kokkos::Impl::SharedAllocationRecord<void,void> *)pRec);
}


void *
EmuStridedSpace::custom_decrement( void * pRec )
{
//   printf("calling AllSh<EmuStride> custom decrement\n");
//   fflush(stdout);
   return (void*)Kokkos::Impl::SharedAllocationRecord< Kokkos::Experimental::EmuStridedSpace , void >::
              custom_decrement( (Kokkos::Impl::SharedAllocationRecord<void,void> *)pRec );
}


void * EmuLocalSpace::allocate( const size_t arg_alloc_size ) const
{
  //MIGRATE((void*)this);  
  //ENTER_CRITICAL_SECTION();
  void * ptr = NULL;
  ptr = (void*)mw_localmalloc( arg_alloc_size, (void*)this );
  long * ref = (long*)ptr;
  ref[sizeof(Kokkos::Impl::SharedAllocationHeader)] = NODE_ID();
  //EXIT_CRITICAL_SECTION();
  return ptr ;
}

void * EmuReplicatedSpace::allocate( const size_t arg_alloc_size ) const
{
  void * ptr = NULL;

  // created replicated 
  ptr = (void*)mw_mallocrepl(arg_alloc_size);
  
  //printf("Replicated space allocated %lx, %d \n", ptr, arg_alloc_size);
  //fflush(stdout);

  return ptr ;
}

void * EmuStridedSpace::allocate( const size_t arg_alloc_size ) const
{
  // Important note here...arg_alloc_size needs to be evenly divisible by NODELETS()
  // because the depth needs to hold n number of elements of size element_type
  KOKKOS_EXPECTS( (arg_alloc_size % NODELETS()) == 0 )
  void * ptr = NULL;

  size_t tcnt = NODELETS();
  size_t depth = ( arg_alloc_size / tcnt );
  //printf("strided space memory allocated: %d, %d \n", tcnt, depth);
  //fflush(stdout);
  
  ptr = (void*)mw_malloc2d( tcnt, depth );
  //printf("                              :  %lx \n", ptr);
  //fflush(stdout);

  return ptr ;
}

void EmuLocalSpace::deallocate( void * const arg_alloc_ptr , const size_t /* arg_alloc_size */ ) const
{
    if ( arg_alloc_ptr != nullptr ) {
        mw_localfree( arg_alloc_ptr );
    }
}

void EmuReplicatedSpace::deallocate( void * const arg_alloc_ptr , const size_t /* arg_alloc_size */ ) const
{
    if ( arg_alloc_ptr != nullptr ) {
      mw_free(arg_alloc_ptr);
    }
}

void EmuStridedSpace::deallocate( void * const arg_alloc_ptr , const size_t /* arg_alloc_size */ ) const
{
   mw_free(arg_alloc_ptr);
}

void * EmuStridedSpace::ess = nullptr;
void * EmuReplicatedSpace::ers = nullptr;
void * EmuReplicatedSpace::repl_root_record = nullptr;
void * EmuLocalSpace::local_root_record = nullptr;
void * EmuStridedSpace::strided_root_record = nullptr;
long * EmuReplicatedSpace::node_count = nullptr;
long * EmuReplicatedSpace::ref_addr = nullptr;

typedef Kokkos::Impl::SharedAllocationRecord< Kokkos::Experimental::EmuReplicatedSpace , void > repl_shared_rec;
typedef Kokkos::Impl::SharedAllocationRecord< Kokkos::Experimental::EmuLocalSpace , void > local_shared_rec;
typedef Kokkos::Impl::SharedAllocationRecord< Kokkos::Experimental::EmuStridedSpace , void > strided_shared_rec;

void initialize_repl_space(int i) {
   repl_shared_rec::RecordBase* rb = (repl_shared_rec::RecordBase*)mw_get_nth(EmuReplicatedSpace::repl_root_record, i);
   new (rb) repl_shared_rec::RecordBase();
   local_shared_rec::RecordBase* lb = (local_shared_rec::RecordBase*)mw_get_nth(EmuLocalSpace::local_root_record, i);
   new (lb) local_shared_rec::RecordBase();
   strided_shared_rec::RecordBase* sb = (strided_shared_rec::RecordBase*)mw_get_nth(EmuStridedSpace::strided_root_record, i);
   new (sb) strided_shared_rec::RecordBase();
   EmuReplicatedSpace * er = (EmuReplicatedSpace*)mw_get_nth(EmuReplicatedSpace::ers, i);
   new (er) EmuReplicatedSpace();
   EmuStridedSpace * es = (EmuStridedSpace*)mw_get_nth(EmuStridedSpace::ess, i);
   new (es) EmuStridedSpace();
   long* nc = (long*)mw_get_nth(EmuReplicatedSpace::node_count, i);
   *nc = NODELETS();
   long* ra = (long*)mw_get_nth(EmuReplicatedSpace::ref_addr, i);
   *ra = (long)&ref_ptr;
}

void initialize_memory_space() {

   long * ptr = getRefPtr();
   EmuReplicatedSpace::repl_root_record = mw_mallocrepl(sizeof(repl_shared_rec::RecordBase));
   EmuLocalSpace::local_root_record = mw_mallocrepl(sizeof(local_shared_rec::RecordBase));
   EmuStridedSpace::strided_root_record = mw_mallocrepl(sizeof(local_shared_rec::RecordBase));
   EmuReplicatedSpace::ers = mw_mallocrepl(sizeof(EmuReplicatedSpace));
   EmuStridedSpace::ess = mw_mallocrepl(sizeof(EmuStridedSpace));
   EmuReplicatedSpace::node_count = (long*)mw_mallocrepl(sizeof(long));
   EmuReplicatedSpace::ref_addr = (long*)mw_mallocrepl(sizeof(long));

   for ( int i = 0; i < NODELETS(); i++) {
       cilk_spawn_at(&ptr[i]) initialize_repl_space(i);
   }
   cilk_sync;
}


} // namespace Experimental
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

std::string
SharedAllocationRecord< Kokkos::Experimental::EmuLocalSpace , void >::get_label() const
{
  SharedAllocationHeader * header = (SharedAllocationHeader *)mw_ptr1to0(RecordBase::head());

  return std::string( header->m_label );
}

std::string
SharedAllocationRecord< Kokkos::Experimental::EmuReplicatedSpace , void >::get_label() const
{
  return std::string( RecordBase::head()->m_label );
}

std::string
SharedAllocationRecord< Kokkos::Experimental::EmuStridedSpace , void >::get_label() const
{
  return std::string( RecordBase::head()->m_label );
}

SharedAllocationRecord< Kokkos::Experimental::EmuLocalSpace , void > *
SharedAllocationRecord< Kokkos::Experimental::EmuLocalSpace , void >::
allocate( const Kokkos::Experimental::EmuLocalSpace &  arg_space
        , const std::string       &  arg_label
        , const size_t               arg_alloc_size
        )
{
//   printf("in allocate: %s \n", arg_label.c_str() );
//   fflush(stdout);
   typedef SharedAllocationRecord< Kokkos::Experimental::EmuLocalSpace , void > local_shared_rec;
/*   int cur_node = NODE_ID();
   Kokkos::Experimental::EmuLocalSpace* spPtr = (Kokkos::Experimental::EmuLocalSpace*)mw_ptr1to0(&arg_space);
   const char * szLabel = (const char *)mw_ptr1to0(arg_label.c_str());
   local_shared_rec::RecordBase* rb = (local_shared_rec::RecordBase*)mw_ptr1to0(Kokkos::Experimental::EmuLocalSpace::local_root_record);
   void * sr = arg_space.allocate( sizeof(SharedAllocationRecord< Kokkos::Experimental::EmuLocalSpace , void >) );
   new (sr) SharedAllocationRecord( rb, spPtr , szLabel , arg_alloc_size, cur_node );
   return (SharedAllocationRecord< Kokkos::Experimental::EmuLocalSpace , void > *)sr;*/
   return new SharedAllocationRecord( (local_shared_rec::RecordBase*)Kokkos::Experimental::EmuLocalSpace::local_root_record, arg_label.c_str(), arg_alloc_size, &arg_space,  0);
}

void local_repl_alloc (int i, void * vr, void * vh, const char * arg_label, const size_t arg_alloc_size) {
  Kokkos::Experimental::repl_shared_rec * pRec = (Kokkos::Experimental::repl_shared_rec*)mw_get_nth(vr,i);
  SharedAllocationHeader* pH = (SharedAllocationHeader*)mw_get_nth(vh, i);
  Kokkos::Experimental::repl_shared_rec::RecordBase* rb = (Kokkos::Experimental::repl_shared_rec::RecordBase*)mw_get_nth(Kokkos::Experimental::EmuReplicatedSpace::repl_root_record, i);
  new (pRec) Kokkos::Experimental::repl_shared_rec( rb, arg_label, arg_alloc_size, pH, (int)i );
}

SharedAllocationRecord< Kokkos::Experimental::EmuReplicatedSpace , void > *
SharedAllocationRecord< Kokkos::Experimental::EmuReplicatedSpace , void >::
allocate( const char *                  arg_label
        , const size_t                  arg_alloc_size
        )
{
   
   long * lRef = (long*)Kokkos::Experimental::getRefPtr();
   Kokkos::Experimental::EmuReplicatedSpace* pMem = ((Kokkos::Experimental::EmuReplicatedSpace*)mw_ptr1to0(Kokkos::Experimental::EmuReplicatedSpace::ers));
   void *vr = pMem->allocate(sizeof(Kokkos::Experimental::repl_shared_rec));
   void *vh = pMem->allocate(sizeof(SharedAllocationHeader) + arg_alloc_size);
                          
   for ( int i = 0; i < NODELETS(); i++) {
	  cilk_spawn_at( &lRef[i] ) local_repl_alloc( i, vr, vh, arg_label, arg_alloc_size );
   }
   cilk_sync;
   return (SharedAllocationRecord< Kokkos::Experimental::EmuReplicatedSpace , void >*)vr;
}

void local_stride_alloc (int i, void * vr, void * vh, void * vd, const char * arg_label, const size_t arg_alloc_size) {
   Kokkos::Experimental::strided_shared_rec * pRec = (Kokkos::Experimental::strided_shared_rec*)mw_get_nth(vr,i);
   SharedAllocationHeader* pH = (SharedAllocationHeader*)mw_get_nth(vh, i);
   Kokkos::Experimental::strided_shared_rec::RecordBase* rb = (Kokkos::Experimental::strided_shared_rec::RecordBase*)
                                                               mw_get_nth(Kokkos::Experimental::EmuStridedSpace::strided_root_record, i);

   new (pRec) Kokkos::Experimental::strided_shared_rec( rb, arg_label, arg_alloc_size, pH, vd, i );  // each replica of the record/header points to the same memory head.
}	
SharedAllocationRecord< Kokkos::Experimental::EmuStridedSpace , void > *
SharedAllocationRecord< Kokkos::Experimental::EmuStridedSpace , void >::
allocate( const char *                         arg_label
        , const size_t                         arg_alloc_size
        )
{
   //printf("strided SAR 2 par allocate: %d, %s \n", arg_alloc_size, arg_label);
   //fflush(stdout);
   long * lRef = (long*)Kokkos::Experimental::EmuReplicatedSpace::getRefAddr();
   Kokkos::Experimental::EmuStridedSpace* pMem = ((Kokkos::Experimental::EmuStridedSpace*)mw_ptr1to0(Kokkos::Experimental::EmuStridedSpace::ess));
   Kokkos::Experimental::EmuReplicatedSpace* pRepl = ((Kokkos::Experimental::EmuReplicatedSpace*)mw_ptr1to0(Kokkos::Experimental::EmuReplicatedSpace::ers));

   void *vr = pRepl->allocate(sizeof(Kokkos::Experimental::strided_shared_rec));      // allocate the record replicated...
   void *vh = pRepl->allocate(sizeof(EmuStridedAllocationHeader)); // allocate the header replicated...
   void *vd = pMem->allocate(arg_alloc_size);  // This is the strided memory
                          
   for ( int i = 0; i < NODELETS(); i++) {
	   cilk_spawn_at(&lRef[i]) local_stride_alloc( i, vr, vh, vd, arg_label, arg_alloc_size );
   }
   cilk_sync;
   return (SharedAllocationRecord< Kokkos::Experimental::EmuStridedSpace , void >*)vr;
}

void
SharedAllocationRecord< Kokkos::Experimental::EmuLocalSpace , void >::
deallocate( SharedAllocationRecord< void , void > * arg_rec )
{
  delete static_cast<SharedAllocationRecord*>(arg_rec);
}

void
SharedAllocationRecord< Kokkos::Experimental::EmuReplicatedSpace , void >::
deallocate( SharedAllocationRecord< void , void > * arg_rec )
{
  delete static_cast<SharedAllocationRecord*>(arg_rec);
}

void
SharedAllocationRecord< Kokkos::Experimental::EmuStridedSpace , void >::
deallocate( SharedAllocationRecord< void , void > * arg_rec )
{
  delete static_cast<SharedAllocationRecord*>(arg_rec);
}

SharedAllocationRecord< Kokkos::Experimental::EmuLocalSpace , void >::
~SharedAllocationRecord()
{
  #if defined(KOKKOS_ENABLE_PROFILING)
  if(Kokkos::Profiling::profileLibraryLoaded()) {

    SharedAllocationHeader header;
    Kokkos::Impl::DeepCopy<Kokkos::Experimental::EmuLocalSpace,HostSpace>( &header , RecordBase::m_alloc_ptr , sizeof(SharedAllocationHeader) );

    Kokkos::Profiling::deallocateData(
      Kokkos::Profiling::SpaceHandle(Kokkos::Experimental::EmuLocalSpace::name()),header.m_label,
      data(),size());
  }
  #endif

  m_space->deallocate( SharedAllocationRecord< void , void >::m_alloc_ptr
                    , SharedAllocationRecord< void , void >::m_alloc_size
                    );
}

SharedAllocationRecord< Kokkos::Experimental::EmuReplicatedSpace , void >::
~SharedAllocationRecord()
{
  #if defined(KOKKOS_ENABLE_PROFILING)
  if(Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::deallocateData(
      Kokkos::Profiling::SpaceHandle(Kokkos::Experimental::EmuReplicatedSpace::name()),RecordBase::m_alloc_ptr->m_label,
      data(),size());
  }
  #endif

  Kokkos::Experimental::EmuReplicatedSpace* pMem = (Kokkos::Experimental::EmuReplicatedSpace*)mw_get_localto((void*)Kokkos::Experimental::EmuReplicatedSpace::ers, this);
  pMem->deallocate( SharedAllocationRecord< void , void >::m_alloc_ptr
                    , SharedAllocationRecord< void , void >::m_alloc_size
                    );
}

SharedAllocationRecord< Kokkos::Experimental::EmuStridedSpace , void >::
~SharedAllocationRecord()
{
  #if defined(KOKKOS_ENABLE_PROFILING)
  if(Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::deallocateData(
      Kokkos::Profiling::SpaceHandle(Kokkos::Experimental::EmuStridedSpace::name()),RecordBase::m_alloc_ptr->m_label,
      data(),size());
  }
  #endif

  Kokkos::Experimental::EmuStridedSpace* pMem = (Kokkos::Experimental::EmuStridedSpace*)mw_get_localto((void*)Kokkos::Experimental::EmuStridedSpace::ess, this);
  pMem->deallocate( SharedAllocationRecord< void , void >::m_alloc_ptr
                    , SharedAllocationRecord< void , void >::m_alloc_size
                    );
}

SharedAllocationRecord< Kokkos::Experimental::EmuLocalSpace , void >::
SharedAllocationRecord( RecordBase*                      basePtr
                        , const char *                     arg_label
                        , const size_t                     arg_alloc_size
                        , const Kokkos::Experimental::EmuLocalSpace        * arg_space
                        , int node
                        , const RecordBase::function_type  arg_dealloc
                        )
  : SharedAllocationRecord< void , void > ( 
#ifdef KOKKOS_DEBUG
        basePtr, 
#endif
        reinterpret_cast<SharedAllocationHeader*>( arg_space->allocate( sizeof(SharedAllocationHeader) + arg_alloc_size ) )
      , sizeof(SharedAllocationHeader) + arg_alloc_size
      , arg_dealloc
      )
   , m_space(arg_space)
{
//  #if defined(KOKKOS_ENABLE_PROFILING)
//  if(Kokkos::Profiling::profileLibraryLoaded()) {
//    Kokkos::Profiling::allocateData(Kokkos::Profiling::SpaceHandle(arg_space.name()),arg_label,data(),arg_alloc_size);
//  }
//  #endif
  SharedAllocationHeader * header = reinterpret_cast<SharedAllocationHeader*>(RecordBase::m_alloc_ptr);

  // Fill in the Header information
  header->m_record = static_cast< SharedAllocationRecord< void , void > * >( this );  
  int iX = 0;
  for (int i = 0; i < SharedAllocationHeader::maximum_label_length-2; i++) {
     header->m_label[i] = *(arg_label+i);
     if (*(arg_label+i) == 0) {
        iX = i;
        break;
     }
  }
  if (iX > 0) {
     header->m_label[iX] = 0x30 + (char)node;
     header->m_label[iX+1] = 0;
  }
}

SharedAllocationRecord< Kokkos::Experimental::EmuLocalSpace , void >::
  SharedAllocationRecord( RecordBase*                      basePtr
                        , const char *                     arg_label
                        , const size_t                     arg_alloc_size
                        , const Kokkos::Experimental::EmuLocalSpace        * arg_space
                        , int node
                        )
  // Pass through allocated [ SharedAllocationHeader , user_memory ]
  // Pass through deallocation function
  : SharedAllocationRecord< void , void > ( 
#ifdef KOKKOS_DEBUG
        basePtr, 
#endif
        reinterpret_cast<SharedAllocationHeader*>( arg_space->allocate( sizeof(SharedAllocationHeader) + arg_alloc_size ) )
      , sizeof(SharedAllocationHeader) + arg_alloc_size
      , nullptr
      )
  , m_space( arg_space )
{

//  #if defined(KOKKOS_ENABLE_PROFILING)
//  if(Kokkos::Profiling::profileLibraryLoaded()) {
//    Kokkos::Profiling::allocateData(Kokkos::Profiling::SpaceHandle(arg_space.name()),arg_label,data(),arg_alloc_size);
//  }
//  #endif

  SharedAllocationHeader * header = reinterpret_cast<SharedAllocationHeader*>(RecordBase::m_alloc_ptr);

  // Fill in the Header information
  header->m_record = static_cast< SharedAllocationRecord< void , void > * >( this );  
  int iX = 0;
  for (int i = 0; i < SharedAllocationHeader::maximum_label_length-2; i++) {
     header->m_label[i] = *(arg_label+i);
     if (*(arg_label+i) == 0) {
        iX = i;
        break;
     }
  }
  if (iX > 0) {
     header->m_label[iX] = 0x30 + (char)node;
     header->m_label[iX+1] = 0;
  }
}

void SharedAllocationRecord< Kokkos::Experimental::EmuReplicatedSpace , void >::increment_repl_count ( int i, Kokkos::Impl::SharedAllocationRecord<void,void>* pRec ) {
  // printf("increment count on node: %d\n", i);
   Kokkos::Impl::SharedAllocationRecord<void,void>* pL = (Kokkos::Impl::SharedAllocationRecord<void,void>*)mw_get_nth(pRec,i);
   Kokkos::atomic_fetch_add( & pL->m_count , 1 );
}


void SharedAllocationRecord< Kokkos::Experimental::EmuReplicatedSpace , void >::
custom_increment( Kokkos::Impl::SharedAllocationRecord<void,void>* pRec ) {   
   long * pRef = Kokkos::Experimental::getRefPtr();
   for ( int i = 0; i < NODELETS(); i++) {
      cilk_spawn_at(&pRef[i]) increment_repl_count(i, pRec);
   }
   cilk_sync;
}

void SharedAllocationRecord< Kokkos::Experimental::EmuStridedSpace , void >::
custom_increment( Kokkos::Impl::SharedAllocationRecord<void,void>* pRec ) {
   for ( int i = 0; i < NODELETS(); i++) {
      //printf("increment count on node: %d\n", i);
      Kokkos::Impl::SharedAllocationRecord<void,void>* pL = (Kokkos::Impl::SharedAllocationRecord<void,void>*)mw_get_nth(pRec,i);
      Kokkos::atomic_fetch_add( & pL->m_count , 1 );
   }
}


Kokkos::Impl::SharedAllocationRecord<void,void>*
SharedAllocationRecord< Kokkos::Experimental::EmuReplicatedSpace , void >::custom_decrement( Kokkos::Impl::SharedAllocationRecord<void,void>* pRec ) {
#ifdef KOKKOS_DEBUG
   constexpr static SharedAllocationRecord<void,void> * zero = nullptr ;
#endif
   bool bFreeMemory = false;   
   for ( int i = 0; i < NODELETS(); i++) {
    //  printf("decrement count on node: %d\n", i);
    //  fflush(stdout);
      Kokkos::Impl::SharedAllocationRecord<void,void>* pL = (Kokkos::Impl::SharedAllocationRecord<void,void>*)mw_get_nth(pRec,i);
      const int old_count = Kokkos::atomic_fetch_add( & pL->m_count , -1 );
      if ( old_count == 1 ) {
        bFreeMemory = true;

#ifdef KOKKOS_DEBUG
        // before:  arg_record->m_prev->m_next == arg_record  &&
        //          arg_record->m_next->m_prev == arg_record
        //
        // after:   arg_record->m_prev->m_next == arg_record->m_next  &&
        //          arg_record->m_next->m_prev == arg_record->m_prev
    
        SharedAllocationRecord<void,void> * root_next = 0 ;

        // Lock the list:
        while (root_next == zero) {			
			root_next = (SharedAllocationRecord<void,void> *)Kokkos::atomic_exchange( & pL->m_root->m_next , zero );
			if (root_next == zero)
			   RESCHEDULE();
        }
        pL->m_next->m_prev = pL->m_prev ;

        if ( root_next != pL ) {
          pL->m_prev->m_next = pL->m_next ;
        }
        else {
          // before:  arg_record->m_root == arg_record->m_prev
          // after:   arg_record->m_root == arg_record->m_next
          root_next = pL->m_next ;
        }

        Kokkos::memory_fence();

        // Unlock the list:
        if ( zero != (SharedAllocationRecord<void,void> *)Kokkos::atomic_exchange( & pL->m_root->m_next , root_next ) ) {
           Kokkos::Impl::throw_runtime_exception("Kokkos::Impl::SharedAllocationRecord failed decrement unlocking");
        }

        pL->m_next = 0 ;
        pL->m_prev = 0 ;
#endif
        //printf("cleared linked list on %d, now calling dealloc\n", i);
        //fflush(stdout);

        function_type d = pL->m_dealloc ;
        if (d != nullptr) {
           (*d)( pL );
        }
     }
     else if ( old_count < 1 ) { // Error
        Kokkos::Impl::throw_runtime_exception("Kokkos::Impl::SharedAllocationRecord failed decrement count");
     }
  }
  if (bFreeMemory) {
     Kokkos::Experimental::EmuReplicatedSpace* pMem = ((Kokkos::Experimental::EmuReplicatedSpace*)mw_ptr1to0(Kokkos::Experimental::EmuReplicatedSpace::ers));
     pMem->deallocate(mw_ptr1to0(pRec->m_alloc_ptr), pRec->m_alloc_size);
     pMem->deallocate(mw_ptr1to0(pRec), sizeof(SharedAllocationRecord< Kokkos::Experimental::EmuReplicatedSpace , void >));
  }

  return pRec;
}

Kokkos::Impl::SharedAllocationRecord<void,void>*
SharedAllocationRecord< Kokkos::Experimental::EmuStridedSpace , void >::custom_decrement( Kokkos::Impl::SharedAllocationRecord<void,void>* pRec ) {
#ifdef KOKKOS_DEBUG
   constexpr static SharedAllocationRecord<void,void> * zero = nullptr ;
#endif
   typedef Kokkos::Impl::SharedAllocationRecord<void,void> record_base;
   bool bFreeMemory = false;   
   record_base* rb = (record_base*)mw_get_nth(pRec, NODE_ID());
   EmuStridedAllocationHeader * pEmuHead = (EmuStridedAllocationHeader*)rb->m_alloc_ptr;
   void* sd = pEmuHead->m_stridedData;

   for ( int i = 0; i < NODELETS(); i++) {
      //printf("decrement count on node: %d\n", i);
      //fflush(stdout);
      Kokkos::Impl::SharedAllocationRecord<void,void>* pL = (Kokkos::Impl::SharedAllocationRecord<void,void>*)mw_get_nth(pRec,i);
      const int old_count = Kokkos::atomic_fetch_add( & pL->m_count , -1 );
   //   printf("count decremented on node: %d -> %d \n", i, old_count);
   //   fflush(stdout);      
      if ( old_count == 1 ) {
        bFreeMemory = true;

#ifdef KOKKOS_DEBUG
        // before:  arg_record->m_prev->m_next == arg_record  &&
        //          arg_record->m_next->m_prev == arg_record
        //
        // after:   arg_record->m_prev->m_next == arg_record->m_next  &&
        //          arg_record->m_next->m_prev == arg_record->m_prev
    
        SharedAllocationRecord<void,void> * root_next = 0 ;

        // Lock the list:
        while ( root_next == zero ) {
			root_next = (SharedAllocationRecord<void,void> *)Kokkos::atomic_exchange( & pL->m_root->m_next , zero );
			if (root_next == zero)
			   RESCHEDULE();
		}

        pL->m_next->m_prev = pL->m_prev ;

        if ( root_next != pL ) {
          pL->m_prev->m_next = pL->m_next ;
        }
        else {
          // before:  arg_record->m_root == arg_record->m_prev
          // after:   arg_record->m_root == arg_record->m_next
          root_next = pL->m_next ;
        }

        Kokkos::memory_fence();

        // Unlock the list:
        if ( zero != (SharedAllocationRecord<void,void> *)Kokkos::atomic_exchange( & pL->m_root->m_next , root_next ) ) {
           Kokkos::Impl::throw_runtime_exception("Kokkos::Impl::SharedAllocationRecord failed decrement unlocking");
        }

        pL->m_next = 0 ;
        pL->m_prev = 0 ;
#endif

/*        if ( i == 0 ) { // only do this on the first one...  
	       printf("calling deallocator: %d\n", i);     
	       fflush(stdout);
           function_type d = pL->m_dealloc ;
           if (d != nullptr) {
              (*d)( pL );
           }
        }*/
     }
     else if ( old_count < 1 ) { // Error
        Kokkos::Impl::throw_runtime_exception("Kokkos::Impl::SharedAllocationRecord failed decrement count");
     }
  }
  if (bFreeMemory) {
	 //printf("strided calling free memory ...\n");
	 //fflush(stdout);
	 
     Kokkos::Experimental::EmuStridedSpace* pMem = ((Kokkos::Experimental::EmuStridedSpace*)mw_ptr1to0(Kokkos::Experimental::EmuStridedSpace::ess));
     Kokkos::Experimental::EmuReplicatedSpace* pRepl = ((Kokkos::Experimental::EmuReplicatedSpace*)mw_ptr1to0(Kokkos::Experimental::EmuReplicatedSpace::ers));
     if (sd != nullptr) {
        pMem->deallocate(sd, 0);
     }
     pRepl->deallocate(mw_ptr1to0(pRec), sizeof(SharedAllocationRecord< Kokkos::Experimental::EmuStridedSpace , void >));
     
  }

  return pRec;
}



SharedAllocationRecord< Kokkos::Experimental::EmuReplicatedSpace , void >::
SharedAllocationRecord( RecordBase*                  basePtr
                      , const char *                 arg_label
                      , const size_t                 arg_alloc_size
                      , void *                       arg_ptr
                      , int node
                      )
  // Pass through allocated [ SharedAllocationHeader , user_memory ]
  // Pass through deallocation function
  : SharedAllocationRecord< Kokkos::Experimental::EmuReplicatedSpace , void > ( 
        basePtr, 
        arg_label,
        arg_alloc_size,
        arg_ptr,
        node,
        nullptr
      )
{
}

SharedAllocationRecord< Kokkos::Experimental::EmuReplicatedSpace , void >::
SharedAllocationRecord( RecordBase*                        basePtr
                        , const char *                     arg_label
                        , const size_t                     arg_alloc_size
                        , void *                           arg_ptr
                        , int node
                        , const RecordBase::function_type  arg_dealloc
                      )
  : SharedAllocationRecord< void , void > ( 
#ifdef KOKKOS_DEBUG
        basePtr,
#endif
        reinterpret_cast<SharedAllocationHeader*>( arg_ptr )
      , sizeof(SharedAllocationHeader) + arg_alloc_size
      , arg_dealloc
      )
{
//  #if defined(KOKKOS_ENABLE_PROFILING)
//  if(Kokkos::Profiling::profileLibraryLoaded()) {
//    Kokkos::Profiling::allocateData(Kokkos::Profiling::SpaceHandle(arg_space.name()),arg_label,data(),arg_alloc_size);
//  }
//  #endif
  this->m_custom_inc = Kokkos::Experimental::EmuReplicatedSpace::custom_increment;
  this->m_custom_dec = Kokkos::Experimental::EmuReplicatedSpace::custom_decrement;
  SharedAllocationHeader * pHead = (SharedAllocationHeader*)RecordBase::m_alloc_ptr;  
  pHead->m_record = this;     
  int iX = -1;
  const char* pL = arg_label;
  for (int i = 0; i < SharedAllocationHeader::maximum_label_length - 2; i++) {
     pHead->m_label[i] = *(pL+i);
     if (*(pL+i) == 0) {
        iX = i;
        break;
     }
  }
  if (iX > 0) {
     pHead->m_label[iX] = 0x30 + (char)node;
     pHead->m_label[iX+1] = 0;
  }
}

SharedAllocationRecord< Kokkos::Experimental::EmuStridedSpace , void >::
SharedAllocationRecord( RecordBase*                      basePtr
                        , const char *                     arg_label
                        , const size_t                     arg_alloc_size
                        , void *                           arg_ptr
                        , void *                           data_ptr
                        , int node
                        )
  // Pass through allocated [ SharedAllocationHeader , user_memory ]
  // Pass through deallocation function
  : SharedAllocationRecord< Kokkos::Experimental::EmuStridedSpace , void > ( 
        basePtr, 
        arg_label,
        arg_alloc_size,
        arg_ptr,
        data_ptr,
        node,
        nullptr
      )
{
}


SharedAllocationRecord< Kokkos::Experimental::EmuStridedSpace , void >::
SharedAllocationRecord( RecordBase*                      basePtr
                        , const char *                     arg_label
                        , const size_t                     arg_alloc_size
                        , void *                           arg_ptr
                        , void *                           data_ptr
                        , int node
                        , const RecordBase::function_type  arg_dealloc
                        )
  // Pass through allocated [ SharedAllocationHeader , user_memory ]
  // Pass through deallocation function
  : SharedAllocationRecord< void , void > ( 
#ifdef KOKKOS_DEBUG
      basePtr, 
#endif
        reinterpret_cast<SharedAllocationHeader*>( arg_ptr )
      , sizeof(SharedAllocationHeader) + arg_alloc_size
      , arg_dealloc
      )
{
//  #if defined(KOKKOS_ENABLE_PROFILING)
//  if(Kokkos::Profiling::profileLibraryLoaded()) {
//    Kokkos::Profiling::allocateData(Kokkos::Profiling::SpaceHandle(arg_space.name()),arg_label,data(),arg_alloc_size);
//  }
//  #endif
  //printf("strided SAR constructor %lx \n", data_ptr);
  //fflush(stdout);
  // Fill in the Header information, directly accessible via UVM
  this->m_custom_inc = Kokkos::Experimental::EmuStridedSpace::custom_increment;
  this->m_custom_dec = Kokkos::Experimental::EmuStridedSpace::custom_decrement;
  EmuStridedAllocationHeader * pEmuHead = (EmuStridedAllocationHeader*)RecordBase::m_alloc_ptr;
  pEmuHead->m_stridedData = data_ptr;
  SharedAllocationHeader * pHead = (SharedAllocationHeader*)RecordBase::m_alloc_ptr;  
  pHead->m_record = this;
  int iX = -1;
  const char* pL = arg_label;
  for (int i = 0; i < SharedAllocationHeader::maximum_label_length - 2; i++) {
     pHead->m_label[i] = *(pL+i);
     if (*(pL+i) == 0) {
        iX = i;
        break;
     }
  }
  if (iX > 0) {
     pHead->m_label[iX] = 0x30 + (char)node;
     pHead->m_label[iX+1] = 0;
  }
}

//----------------------------------------------------------------------------

void * SharedAllocationRecord< Kokkos::Experimental::EmuLocalSpace , void >::
allocate_tracked( const Kokkos::Experimental::EmuLocalSpace & arg_space
                , const std::string & arg_alloc_label
                , const size_t arg_alloc_size )
{
  if ( ! arg_alloc_size ) return (void *) 0 ;

  SharedAllocationRecord * const r =
    allocate( arg_space , arg_alloc_label , arg_alloc_size );

  RecordBase::increment( r );

  return r->data();
}

void SharedAllocationRecord< Kokkos::Experimental::EmuLocalSpace , void >::
deallocate_tracked( void * const arg_alloc_ptr )
{
  if ( arg_alloc_ptr != 0 ) {
    SharedAllocationRecord * const r = get_record( arg_alloc_ptr );

    RecordBase::decrement( r );
  }
}

void * SharedAllocationRecord< Kokkos::Experimental::EmuLocalSpace , void >::
reallocate_tracked( void * const arg_alloc_ptr
                  , const size_t arg_alloc_size )
{
  SharedAllocationRecord * const r_old = get_record( arg_alloc_ptr );
  SharedAllocationRecord * const r_new = allocate( *(r_old->m_space) , r_old->get_label() , arg_alloc_size );

  Kokkos::Impl::DeepCopy<Kokkos::Experimental::EmuLocalSpace,Kokkos::Experimental::EmuLocalSpace>( r_new->data() , r_old->data()
                                             , std::min( r_old->size() , r_new->size() ) );

  RecordBase::increment( r_new );
  RecordBase::decrement( r_old );

  return r_new->data();
}

void * SharedAllocationRecord< Kokkos::Experimental::EmuReplicatedSpace , void >::
allocate_tracked( const Kokkos::Experimental::EmuReplicatedSpace & arg_space
                , const std::string & arg_alloc_label
                , const size_t arg_alloc_size )
{
  if ( ! arg_alloc_size ) return (void *) 0 ;

  SharedAllocationRecord * const r =
    allocate( arg_alloc_label.c_str() , arg_alloc_size );

  RecordBase::increment( r );

  return r->data();
}

void SharedAllocationRecord< Kokkos::Experimental::EmuReplicatedSpace , void >::
deallocate_tracked( void * const arg_alloc_ptr )
{
  if ( arg_alloc_ptr != 0 ) {

    SharedAllocationRecord * const r = get_record( arg_alloc_ptr );

    RecordBase::decrement( r );
  }
}

void * SharedAllocationRecord< Kokkos::Experimental::EmuReplicatedSpace , void >::
reallocate_tracked( void * const arg_alloc_ptr
                  , const size_t arg_alloc_size )
{
  SharedAllocationRecord * const r_old = get_record( arg_alloc_ptr );
  SharedAllocationRecord * const r_new = allocate( r_old->get_label().c_str() , arg_alloc_size );

  Kokkos::Impl::DeepCopy<Kokkos::Experimental::EmuReplicatedSpace,Kokkos::Experimental::EmuReplicatedSpace>( r_new->data() , r_old->data()
                                             , std::min( r_old->size() , r_new->size() ) );

  RecordBase::increment( r_new );
  RecordBase::decrement( r_old );

  return r_new->data();
}

void * SharedAllocationRecord< Kokkos::Experimental::EmuStridedSpace , void >::
allocate_tracked( const Kokkos::Experimental::EmuStridedSpace & arg_space
                , const std::string & arg_alloc_label
                , const size_t arg_alloc_size )
{
  if ( ! arg_alloc_size ) return (void *) 0 ;

  SharedAllocationRecord * const r =
    allocate( arg_alloc_label.c_str() , arg_alloc_size );

  RecordBase::increment( r );

  return r->data();
}

void SharedAllocationRecord< Kokkos::Experimental::EmuStridedSpace , void >::
deallocate_tracked( void * const arg_alloc_ptr )
{
  if ( arg_alloc_ptr != 0 ) {
    SharedAllocationRecord * const r = get_record( arg_alloc_ptr );

    RecordBase::decrement( r );
  }
}

void * SharedAllocationRecord< Kokkos::Experimental::EmuStridedSpace , void >::
reallocate_tracked( void * const arg_alloc_ptr
                  , const size_t arg_alloc_size )
{
  SharedAllocationRecord * const r_old = get_record( arg_alloc_ptr );
  SharedAllocationRecord * const r_new = allocate( r_old->get_label().c_str() , arg_alloc_size );

  Kokkos::Impl::DeepCopy<Kokkos::Experimental::EmuStridedSpace,Kokkos::Experimental::EmuStridedSpace>( r_new->data() , r_old->data()
                                             , std::min( r_old->size() , r_new->size() ) );

  RecordBase::increment( r_new );
  RecordBase::decrement( r_old );

  return r_new->data();
}

//----------------------------------------------------------------------------

SharedAllocationRecord< Kokkos::Experimental::EmuLocalSpace , void > *
SharedAllocationRecord< Kokkos::Experimental::EmuLocalSpace , void >::get_record( void * alloc_ptr )
{
  //using RecordBase = SharedAllocationRecord< void , void > ;
  using Header     = SharedAllocationHeader ;
  using RecordEmu = SharedAllocationRecord< Kokkos::Experimental::EmuLocalSpace , void > ;


  // Iterate the list to search for the record among all allocations
  // requires obtaining the root of the list and then locking the list.

  Header * const h = alloc_ptr ? reinterpret_cast< Header * >( alloc_ptr ) - 1 : (Header *) 0 ;

  if ( ! alloc_ptr || h->m_record->m_alloc_ptr != h ) {
    Kokkos::Impl::throw_runtime_exception( std::string("Kokkos::Impl::SharedAllocationRecord< Kokkos::Experimental::EmuLocalSpace , void >::get_record ERROR" ) );
  }

  return static_cast< RecordEmu * >( h->m_record );

}

SharedAllocationRecord< Kokkos::Experimental::EmuReplicatedSpace , void > *
SharedAllocationRecord< Kokkos::Experimental::EmuReplicatedSpace , void >::get_record( void * alloc_ptr )
{
  using Header     = SharedAllocationHeader ;
  using RecordEmu = SharedAllocationRecord< Kokkos::Experimental::EmuReplicatedSpace , void > ;

  Header * const h = alloc_ptr ? reinterpret_cast< Header * >( alloc_ptr ) - 1 : (Header *) 0 ;

  if ( ! alloc_ptr || h->m_record->m_alloc_ptr != h ) {
    Kokkos::Impl::throw_runtime_exception( std::string("Kokkos::Impl::SharedAllocationRecord< Kokkos::Experimental::EmuReplicatedSpace , void >::get_record ERROR" ) );
  }

  return static_cast< RecordEmu * >( h->m_record );
}

SharedAllocationRecord< Kokkos::Experimental::EmuStridedSpace , void > *
SharedAllocationRecord< Kokkos::Experimental::EmuStridedSpace , void >::get_record( void * alloc_ptr )
{
  using Header     = SharedAllocationHeader ;
  using RecordEmu = SharedAllocationRecord< Kokkos::Experimental::EmuStridedSpace , void > ;

  Header * const h = alloc_ptr ? reinterpret_cast< Header * >( alloc_ptr ) - 1 : (Header *) 0 ;

  if ( ! alloc_ptr || h->m_record->m_alloc_ptr != h ) {
    Kokkos::Impl::throw_runtime_exception( std::string("Kokkos::Impl::SharedAllocationRecord< Kokkos::Experimental::EmuStridedSpace , void >::get_record ERROR" ) );
  }

  return static_cast< RecordEmu * >( h->m_record );
}

// Iterate records to print orphaned memory ...
void
SharedAllocationRecord< Kokkos::Experimental::EmuLocalSpace , void >::
print_records( std::ostream & s , const Kokkos::Experimental::EmuLocalSpace & , bool detail )
{
#ifdef KOKKOS_DEBUG
  SharedAllocationRecord< void , void > * rs = (SharedAllocationRecord< void , void > *)mw_ptr1to0(Kokkos::Experimental::EmuLocalSpace::local_root_record);
  SharedAllocationRecord< void , void > * r = rs;

  char buffer[256] ;

  SharedAllocationHeader head ;

  if ( detail ) {
    do {
      if ( r->m_alloc_ptr ) {
        Kokkos::Impl::DeepCopy<HostSpace,Kokkos::Experimental::EmuLocalSpace>( & head , r->m_alloc_ptr , sizeof(SharedAllocationHeader) );
      }
      else {
        head.m_label[0] = 0 ;
      }

      //Formatting dependent on sizeof(uintptr_t)
      const char * format_string;

      if (sizeof(uintptr_t) == sizeof(unsigned long)) {
        format_string = "Emu addr( 0x%.12lx ) list( 0x%.12lx 0x%.12lx ) extent[ 0x%.12lx + %.8ld ] count(%d) dealloc(0x%.12lx) %s\n";
      }
      else if (sizeof(uintptr_t) == sizeof(unsigned long long)) {
        format_string = "Emu addr( 0x%.12llx ) list( 0x%.12llx 0x%.12llx ) extent[ 0x%.12llx + %.8ld ] count(%d) dealloc(0x%.12llx) %s\n";
      }

      snprintf( buffer , 256
              , format_string
              , reinterpret_cast<uintptr_t>( r )
              , reinterpret_cast<uintptr_t>( r->m_prev )
              , reinterpret_cast<uintptr_t>( r->m_next )
              , reinterpret_cast<uintptr_t>( r->m_alloc_ptr )
              , r->m_alloc_size
              , r->m_count
              , reinterpret_cast<uintptr_t>( r->m_dealloc )
              , head.m_label
              );
      s << buffer ;
      r = r->m_next ;
    } while ( r != rs );
  }
  else {
    do {
      if ( r->m_alloc_ptr ) {

        Kokkos::Impl::DeepCopy<HostSpace,Kokkos::Experimental::EmuLocalSpace>( & head , r->m_alloc_ptr , sizeof(SharedAllocationHeader) );

        //Formatting dependent on sizeof(uintptr_t)
        const char * format_string;

        if (sizeof(uintptr_t) == sizeof(unsigned long)) {
          format_string = "Emu [ 0x%.12lx + %ld ] %s\n";
        }
        else if (sizeof(uintptr_t) == sizeof(unsigned long long)) {
          format_string = "Emu [ 0x%.12llx + %ld ] %s\n";
        }

        snprintf( buffer , 256
                , format_string
                , reinterpret_cast< uintptr_t >( r->data() )
                , r->size()
                , head.m_label
                );
      }
      else {
        snprintf( buffer , 256 , "Emu [ 0 + 0 ]\n" );
      }
      s << buffer ;
      r = r->m_next ;
    } while ( r != rs );
  }
#else
  Kokkos::Impl::throw_runtime_exception(
      "Kokkos::Impl::SharedAllocationRecord<EmuLocalSpace,void>::print_records"
      " only works with KOKKOS_DEBUG enabled");
#endif
}

void
SharedAllocationRecord< Kokkos::Experimental::EmuReplicatedSpace , void >::
print_records( std::ostream & s , const Kokkos::Experimental::EmuReplicatedSpace & , bool detail )
{
//  SharedAllocationRecord< void , void >::print_host_accessible_records( s , "EmuReplicated" , (RecordBase*)s_root_record , detail );
}

void
SharedAllocationRecord< Kokkos::Experimental::EmuStridedSpace , void >::
print_records( std::ostream & s , const Kokkos::Experimental::EmuStridedSpace & , bool detail )
{
//  SharedAllocationRecord< void , void >::print_host_accessible_records( s , "EmuStrided" , & s_root_record , detail );
}


} // namespace Impl
} // namespace Kokkos
#else
void KOKKOS_CORE_SRC_EMU_EmuLocalSpace_PREVENT_LINK_ERROR() {}
#endif // KOKKOS_ENABLE_EMU

