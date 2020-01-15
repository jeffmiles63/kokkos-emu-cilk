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

#include <gtest/gtest.h>

#include <stdexcept>
#include <sstream>
#include <iostream>

#include <Kokkos_Core.hpp>

/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Experimental {
extern void initialize_memory_space();
}
}  // namespace Kokkos

namespace Test {

struct SharedAllocDestroy {
  volatile int* count;

  SharedAllocDestroy() = default;
  SharedAllocDestroy(int* arg) : count(arg) {}

  void destroy_shared_allocation() { Kokkos::atomic_increment(count); }
};

template <class MemorySpace, class ExecutionSpace>
void test_shared_alloc() {
#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)
  typedef const Kokkos::Impl::SharedAllocationHeader Header;
  typedef Kokkos::Impl::SharedAllocationTracker Tracker;
  typedef Kokkos::Impl::SharedAllocationRecord<void, void> RecordBase;
  typedef Kokkos::Impl::SharedAllocationRecord<MemorySpace, void> RecordMemS;
  typedef Kokkos::Impl::SharedAllocationRecord<MemorySpace, SharedAllocDestroy>
      RecordFull;

  static_assert(sizeof(Tracker) == sizeof(int*),
                "SharedAllocationTracker has wrong size!");

  Kokkos::Experimental::initialize_memory_space();
  MemorySpace s;

  const size_t N = 128;

  RecordMemS* rarray[N];
  Header* harray[N];

  RecordMemS** const r = rarray;
  Header** const h     = harray;

  Kokkos::RangePolicy<ExecutionSpace> range(0, N);

  {
    // Since always executed on host space, leave [=]
    Kokkos::parallel_for(range, [=](size_t i) {
      long test_count       = 0;
      const int old_count_1 = Kokkos::atomic_fetch_add(&test_count, 1);
      const int old_count_2 = Kokkos::atomic_fetch_sub(&test_count, 1);
      // printf("[%d] c1 = %d, c2 = %d \n", i, old_count_1, old_count_2);

      const size_t size = 8;
      char name[64];
      sprintf(name, "test_%.2d", int(i));
      //      printf("A:[%d] allocating record: %d \n", i, (int)size * ( i + 1
      //      )); r[i] = RecordMemS::allocate( name, size * ( i + 1 ) );
      r[i] = RecordMemS::allocate(s, name, size * (i + 1));

      //      printf("A:[%d] get header: %08x \n", i, (unsigned long)r[i]);
      //      fflush(stdout);
      h[i] = Header::get_header(r[i]->data());

      ASSERT_EQ(r[i]->use_count(), 0);

      //      printf("A:[%d] increment record count: %08x \n", i, (unsigned
      //      long)r[i]);
      for (size_t j = 0; j < (i / 10) + 1; ++j) RecordBase::increment(r[i]);

      long* pd = (long*)r[i]->data();
      *pd      = i * 2;

      // printf("increment record counter: %d, %d \n", i, r[i]->use_count());
      // fflush(stdout);
      ASSERT_EQ(r[i]->use_count(), (i / 10) + 1);
      ASSERT_EQ(r[i], RecordMemS::get_record(r[i]->data()));
    });

    ExecutionSpace::fence();

    for (int i = 0; i < N; i++) {
      std::string value = r[i]->get_label();
      // printf("record [%d]: %s\n", i, value.c_str());
      // fflush(stdout);
    }

    ExecutionSpace::fence();
#ifdef KOKKOS_DEBUG
    // Sanity check for the whole set of allocation records to which this record
    // belongs.
    RecordBase::is_sane(r[0]);
    // RecordMemS::print_records( std::cout, s, true );
#endif

    Kokkos::parallel_for(range, [=](size_t i) {
      while (0 !=
             (r[i] = static_cast<RecordMemS*>(RecordBase::decrement(r[i])))) {
        //  printf("still waiting: %d \n", r[i]->use_count());
        // fflush(stdout);
#ifdef KOKKOS_DEBUG
        if (r[i]->use_count() == 1) RecordBase::is_sane(r[i]);
#endif
      }
    });
  }

  {
    int destroy_count = 0;
    SharedAllocDestroy counter(&destroy_count);

    Kokkos::parallel_for(range, [=](size_t i) {
      const size_t size = 8;
      char name[64];
      sprintf(name, "test_%.2d", int(i));

      //      printf("B:[%d] allocating record: %d \n", i, (int)size * ( i + 1
      //      ));
      RecordFull* rec =
          RecordFull::allocate(MemorySpace(), name, size * (i + 1));

      rec->m_destroy = counter;

      r[i] = rec;
      h[i] = Header::get_header(r[i]->data());

      ASSERT_EQ(r[i]->use_count(), 0);

      for (size_t j = 0; j < (i / 10) + 1; ++j) RecordBase::increment(r[i]);

      ASSERT_EQ(r[i]->use_count(), (i / 10) + 1);
      ASSERT_EQ(r[i], RecordMemS::get_record(r[i]->data()));
    });

#ifdef KOKKOS_DEBUG
    RecordBase::is_sane(r[0]);
#endif

    Kokkos::parallel_for(range, [=](size_t i) {
      while (0 !=
             (r[i] = static_cast<RecordMemS*>(RecordBase::decrement(r[i])))) {
        // printf("decrement record counter: %d, %d \n", i, r[i]->use_count());
#ifdef KOKKOS_DEBUG
        if (r[i]->use_count() == 1) RecordBase::is_sane(r[i]);
#endif
      }
    });

    ASSERT_EQ(destroy_count, int(N));
  }

  {
    int destroy_count = 0;

    {
      const size_t size = 8;
      //      printf("C:allocating record: %d \n", (int)size);
      RecordFull* rec = RecordFull::allocate(MemorySpace(), "test", size);

      // ... Construction of the allocated { rec->data(), rec->size() }

      // Copy destruction function object into the allocation record.
      rec->m_destroy = SharedAllocDestroy(&destroy_count);

      ASSERT_EQ(rec->use_count(), 0);

      // Start tracking, increments the use count from 0 to 1.
      Tracker track;
      track.assign_allocated_record_to_uninitialized(rec);

      ASSERT_EQ(rec->use_count(), 1);
      ASSERT_EQ(track.use_count(), 1);

      // Verify construction / destruction increment.
      for (size_t i = 0; i < N; ++i) {
        ASSERT_EQ(rec->use_count(), 1);
        {
          Tracker local_tracker;
          //          printf("[%d] tracker assigning record \n", i );
          local_tracker.assign_allocated_record_to_uninitialized(rec);
          ASSERT_EQ(rec->use_count(), 2);
          ASSERT_EQ(local_tracker.use_count(), 2);
        }

        ASSERT_EQ(rec->use_count(), 1);
        ASSERT_EQ(track.use_count(), 1);
      }

      Kokkos::parallel_for(range, [=](size_t i) {
        Tracker local_tracker;
        local_tracker.assign_allocated_record_to_uninitialized(rec);
        ASSERT_GT(rec->use_count(), 1);
      });

      ASSERT_EQ(rec->use_count(), 1);
      ASSERT_EQ(track.use_count(), 1);

      // Destruction of 'track' object deallocates the 'rec' and invokes the
      // destroy function object.
    }

    ASSERT_EQ(destroy_count, 1);
  }

#endif /* #if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST ) */
}

template <class MemorySpace, class ExecutionSpace>
void test_repl_shared_alloc() {
#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)
  typedef const Kokkos::Impl::SharedAllocationHeader Header;
  typedef Kokkos::Impl::SharedAllocationTracker Tracker;
  typedef Kokkos::Impl::SharedAllocationRecord<void, void> RecordBase;
  typedef Kokkos::Impl::SharedAllocationRecord<MemorySpace, void> RecordMemS;
  typedef Kokkos::Impl::SharedAllocationRecord<MemorySpace, SharedAllocDestroy>
      RecordFull;

  static_assert(sizeof(Tracker) == sizeof(int*),
                "SharedAllocationTracker has wrong size!");

  ExecutionSpace::fence();
  //  MemorySpace s;

  const size_t N = 512;
  RecordMemS* rarray[N];
  Header* harray[N];

  RecordMemS** const r = rarray;
  Header** const h     = harray;

  Kokkos::RangePolicy<ExecutionSpace> range(0, N);
  Kokkos::Experimental::initialize_memory_space();
  {
    // Since always executed on host space, leave [=]
    //    Kokkos::parallel_for( range, [=] ( size_t i ) {
    for (int i = 0; i < N; i++) {
      const size_t size = 8;
      char name[64];
      sprintf(name, "test_%.2d", int(i));
      //      printf("A:[%d] allocating record: %d \n", i, (int)size * ( i + 1
      //      ));
      r[i] = RecordMemS::allocate(name, size * (i + 1));

      FENCE();
      //   printf("alloc record created %d \n", i);
      //   fflush(stdout);

      Tracker track;
      track.assign_allocated_record_to_uninitialized(r[i]);

      Kokkos::Experimental::EmuReplicatedSpace* pMem =
          ((MemorySpace*)mw_get_nth(
              Kokkos::Experimental::EmuReplicatedSpace::ers, 0));
      long* pRef = (long*)pMem->getRefAddr();

      for (int j = 0; j < NODELETS(); j++) {
        RecordMemS* pr = (RecordMemS*)mw_get_nth(r[i], j);
        MIGRATE((void*)&pRef[j]);
        std::string value = pr->get_label();
        long cnt          = pr->use_count();
        //  printf("record [%d]: %s, count = %d\n", j, value.c_str(), cnt);
        // fflush(stdout);
      }

      //      r[i] = RecordMemS::allocate( s, name, size * ( i + 1 ) );
      //      printf("A:[%d] get header: %08x \n", i, (unsigned long)r[i]);
      //      fflush(stdout);
      /*      h[i] = Header::get_header( r[i]->data() );

            ASSERT_EQ( r[i]->use_count(), 0 );

      //      printf("A:[%d] increment record count: %08x \n", i, (unsigned
      long)r[i]);
      //      for ( size_t j = 0; j < ( i / 10 ) + 1; ++j )
      RecordBase::increment( r[i] );

            long* pd = (long*)r[i]->data();
            *pd = i * 2;

            ASSERT_EQ( r[i]->use_count(), ( i / 10 ) + 1 );
            ASSERT_EQ( r[i], RecordMemS::get_record( r[i]->data() ) );*/
      //    });
      // printf("about to free tracker\n");
      // fflush(stdout);
    }
    /*
        ExecutionSpace::fence();
        printf("construction loop complete\n");
        fflush(stdout);

        for (int i = 0; i < N; i++) {
           for (int j = 0; j < NODELETS(); j++) {
              RecordMemS * pr = (RecordMemS *)mw_get_nth(r[i], j);
              MemorySpace* pMem = ((MemorySpace*)mw_get_nth(&s, j));
              MIGRATE((void*)&(((long*)pMem->getRefAddr())[i]));
              std::string value = pr->get_label();
              printf("record [%d-%d]: %s\n", i, j, value.c_str());
              fflush(stdout);
           }
        }
    */
    /*
        // Sanity check for the whole set of allocation records to which this
    record belongs. #ifdef KOKKOS_DEBUG RecordBase::is_sane( r[0] );
        // RecordMemS::print_records( std::cout, s, true );
    #endif

        Kokkos::parallel_for( range, [=] ( size_t i ) {
          while ( 0 != ( r[i] = static_cast< RecordMemS * >(
    RecordBase::decrement( r[i] ) ) ) ) {
    //        printf("still waiting: %d \n", r[i]->use_count());
    //        fflush(stdout);
            if ( r[i]->use_count() == 1 ) RecordBase::is_sane( r[i] );
          }
        });
    */
    printf("before the end \n");
  }
/*
  {
    int destroy_count = 0;
    SharedAllocDestroy counter( &destroy_count );

    Kokkos::parallel_for( range, [=] ( size_t i ) {
      const size_t size = 8;
      char name[64];
      sprintf( name, "test_%.2d", int( i ) );

//      printf("B:[%d] allocating record: %d \n", i, (int)size * ( i + 1 ));
      RecordFull * rec = RecordFull::allocate( s, name, size * ( i + 1 ) );

      rec->m_destroy = counter;

      r[i] = rec;
      h[i] = Header::get_header( r[i]->data() );

      ASSERT_EQ( r[i]->use_count(), 0 );

      for ( size_t j = 0; j < ( i / 10 ) + 1; ++j ) RecordBase::increment( r[i]
);

      ASSERT_EQ( r[i]->use_count(), ( i / 10 ) + 1 );
      ASSERT_EQ( r[i], RecordMemS::get_record( r[i]->data() ) );
    });

#ifdef KOKKOS_DEBUG
    RecordBase::is_sane( r[0] );
#endif

    Kokkos::parallel_for( range, [=] ( size_t i ) {
      while ( 0 != ( r[i] = static_cast< RecordMemS * >( RecordBase::decrement(
r[i] ) ) ) ) { #ifdef KOKKOS_DEBUG if ( r[i]->use_count() == 1 )
RecordBase::is_sane( r[i] ); #endif
      }
    });

    ASSERT_EQ( destroy_count, int( N ) );
  }

  {
    int destroy_count = 0;

    {
      const size_t size = 8;
//      printf("C:allocating record: %d \n", (int)size);
      RecordFull * rec = RecordFull::allocate( s, "test", size );

      // ... Construction of the allocated { rec->data(), rec->size() }

      // Copy destruction function object into the allocation record.
      rec->m_destroy = SharedAllocDestroy( & destroy_count );

      ASSERT_EQ( rec->use_count(), 0 );

      // Start tracking, increments the use count from 0 to 1.
      Tracker track;

      track.assign_allocated_record_to_uninitialized( rec );

      ASSERT_EQ( rec->use_count(), 1 );
      ASSERT_EQ( track.use_count(), 1 );

      // Verify construction / destruction increment.
      for ( size_t i = 0; i < N; ++i ) {
        ASSERT_EQ( rec->use_count(), 1 );

        {
          Tracker local_tracker;
//          printf("[%d] tracker assigning record \n", i );
          local_tracker.assign_allocated_record_to_uninitialized( rec );
          ASSERT_EQ( rec->use_count(), 2 );
          ASSERT_EQ( local_tracker.use_count(), 2 );
        }

        ASSERT_EQ( rec->use_count(), 1 );
        ASSERT_EQ( track.use_count(), 1 );
      }

      Kokkos::parallel_for( range, [=] ( size_t ) {
        Tracker local_tracker;
        local_tracker.assign_allocated_record_to_uninitialized( rec );
        ASSERT_GT( rec->use_count(), 1 );
      });

      ASSERT_EQ( rec->use_count(), 1 );
      ASSERT_EQ( track.use_count(), 1 );

      // Destruction of 'track' object deallocates the 'rec' and invokes the
destroy function object.
    }

    ASSERT_EQ( destroy_count, 1 );
  }
*/
#endif /* #if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST ) */
}

}  // namespace Test
