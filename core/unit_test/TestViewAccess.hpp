
#include <stdexcept>
#include <limits>
#include <math.h>
#include <memoryweb.h>

#include <Kokkos_Core.hpp>

namespace Kokkos {
namespace Experimental {
   extern void * ers;
   extern replicated EmuReplicatedSpace els;   
}}

namespace test {
	
template<class Scalar>
struct TestContainer {
	long a;
	long b;
	long c;
};

template<class Scalar>
struct ConFunctor {
	typedef Kokkos::View< TestContainer<Scalar> *,  Kokkos::Experimental::EmuStridedSpace > view_type;
	
	
	struct TagInit{};
	struct TagUpdate{};
	
	view_type vt;
	//void * data_ptr;
	//Kokkos::accessor_strided<TestContainer> as;
	
	ConFunctor( const view_type & vt_ ) : vt(vt_){
	}
//	ConFunctor( void * dt_, int bs_ ) : data_ptr(dt_), as(bs_) {
//	}

	
	void operator() (const TagInit&, int i) const { 
		  TestContainer<Scalar>  & tc = vt(i);
		  //TestContainer & tc = as.access((TestContainer*)data_ptr, i);
		  //printf("view pointer [%d] %lx \n", i, &tc);
		  //fflush(stdout);
		  
		  //long* ptr = (long*)(&tc);
		  //ptr[0] = 5;
		  //tc.a = (i % 2);
		  //tc.b = tc.a + 1;
		  //tc.b = tc.a + 2;
          new (&tc) TestContainer<Scalar> {};
          //if (tc.a > 0) printf("[%d] tc.a = %d \n", i, tc.a); fflush(stdout);		
	}

	void operator() (const TagUpdate&, int i) const { 
		  TestContainer<Scalar>  & tc = vt(i);    
		  //TestContainer & tc = as.access((TestContainer*)data_ptr, i);
//		  long* ptr = (long*)(&tc);
//		  ptr[0] = 5;
		  
          //tc.a = i;
          //tc.b = tc.a + 2;
          //tc.c = tc.b + 3;
		  tc.a = (2);
		  tc.b = tc.a + 1;
		  tc.c = tc.a + 2;
          //if (tc.a > 0) printf("[%d] tc.a = %d \n", i, tc.a); fflush(stdout);
          KOKKOS_EXPECTS( tc.c == 4 );
	}

	
};

template< class Scalar, class ExecSpace >
void TestViewAccess( int N ) {
   //starttiming();
   
   Kokkos::View< Scalar*, Kokkos::HostSpace > hs_view( "host", N );
//   printf ("host view size is %ld \n", (unsigned long)sizeof(hs_view));
//   fflush(stdout);

   for (int i = 0; i < N; i++) {
//	  printf("testing host access: %d\n", i);
//	  fflush(stdout);
      hs_view(i) = i*2;
   }
   for (int i = 0; i < N; i++) {
      ASSERT_EQ( hs_view(i), i*2 );
   }

   Kokkos::View< const Scalar*, Kokkos::HostSpace > cp_view( hs_view );
   //printf ("host view size is %ld \n", (unsigned long)sizeof(cp_view));
   Kokkos::parallel_for (N, KOKKOS_LAMBDA (int i) {
      hs_view(i) = i*3;
   } );
   Kokkos::fence();   
   for (int i = 0; i < N; i++) {
      ASSERT_EQ( cp_view(i), i*3 );
   }
    

//   printf("Testing 2D view...\n");
//   fflush(stdout);   
   Kokkos::View< Scalar**, Kokkos::HostSpace > dd_view( "2d view", 8, N );
   //printf ("host view size is %ld \n", (unsigned long)sizeof(cp_view));
   Kokkos::parallel_for (N, KOKKOS_LAMBDA (int i) {
   //for (int i = 0; i < N; i++) {
	   for (int r = 0; r < 8; r++) {
         dd_view(r,i) = i*3 + r;
	   }
   }
   );
   Kokkos::fence();   
   printf("host access test complete %d\n", N);
   fflush(stdout);   
      
   
   
   Scalar total = 0;
   Kokkos::parallel_reduce (N, KOKKOS_LAMBDA (int i, Scalar& update) {
	  //printf("[%d] inside reduce functor %d\n", NODE_ID(), i);
	  //fflush(stdout);
	  //Scalar upd = hs_view(i);
	  //printf("[%d] updating reduce var %d, %ld\n", NODE_ID(), i, upd);
	  //fflush(stdout);	  
      update += 1;
      //printf("[%d] finished updating %d\n", NODE_ID(), i);
	  //fflush(stdout);	        
   }, total );
   Kokkos::fence();   
   printf("parallel reduce test complete %d\n", total);
   fflush(stdout);      
   
/*
//   long * refPtr = Kokkos::Experimental::EmuReplicatedSpace::getRefAddr();
   for (int i = 0; i < Kokkos::Experimental::EmuReplicatedSpace::memory_zones(); i++) {
      MIGRATE(&refPtr[i]);
//      Kokkos::View< Scalar*, Kokkos::Experimental::EmuLocalSpace > local_space_view( Kokkos::ViewAllocateWithoutInitializing("local"), N );
      Kokkos::View< Scalar*, Kokkos::Experimental::EmuLocalSpace > local_space_view( "local", N );      
  
      if (mw_islocal(local_space_view.data())) {
         for (int i = 0; i < N; i++) {
            local_space_view(i) = (Scalar)i;
         }
      } else {
         printf("local mem check skipped, pointer is not local \n");
      }      
      int node_id = NODE_ID();
      printf("local mem current node: %d \n", node_id);
      fflush(stdout);
   }
   printf("local memory view test complete\n");

*/
/*
   {
      Kokkos::View< Scalar*, Kokkos::Experimental::EmuReplicatedSpace > replicated_space_view( "replicated", N );   
      printf ("replicated view size is %ld \n", (unsigned long)sizeof(replicated_space_view));
      printf("Testing access to replicated space view\n");
      fflush(stdout);

      Kokkos::parallel_for (Kokkos::Experimental::EmuReplicatedSpace::memory_zones(), KOKKOS_LAMBDA (int i) {
         long * refPtr = Kokkos::Experimental::EmuReplicatedSpace::getRefAddr();
//         printf("inside loop: %d \n", i); fflush(stdout);
         MIGRATE(&refPtr[i]);
         for (int n = 0; n < N; n++) {
            replicated_space_view(n) = (Scalar)n;
		 }
      });
   } 
  */ 
   
   Kokkos::fence();
   {
      Kokkos::View< Scalar*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::ForceRemote> > global_space_view( "global", N );
//      printf ("global view size is %ld \n", (unsigned long)sizeof(global_space_view));
//      Kokkos::View< Scalar*, Kokkos::HostSpace, Kokkos::MemoryTraits<0> > global_space_view( "global", N );      


//      printf("Testing access to global space view\n");
//      fflush(stdout);

      Kokkos::parallel_for(N, KOKKOS_LAMBDA (const int i) {
//      for (int i = 0; i < N; i++) {
         global_space_view(i) = (Scalar)i;
      });
//      }
   }
   Kokkos::fence();
  
   printf("Access to global space view test complete: %d\n", N);
   fflush(stdout);
   
   for (int r = 0; r < 1; r++) 
   {
      Kokkos::View< Scalar*, Kokkos::Experimental::EmuStridedSpace > strided_space_view( "strided", N );
//      printf ("strided view size is %ld \n", (unsigned long)sizeof(strided_space_view));
//      fflush(stdout);

//      printf("updating strided space view\n");
//      fflush(stdout);

      Kokkos::parallel_for(N, KOKKOS_LAMBDA (const int i) {      
//      for (int i = 0; i < N; i++ ) {
         strided_space_view(i) = (Scalar)i;
      });
//      }
   }
   
   
   {
      Kokkos::View< Scalar**, Kokkos::Experimental::EmuStridedSpace > strided_space_view( "2D strided", 8, N );
//      printf ("strided view size is %ld \n", (unsigned long)sizeof(strided_space_view));
//      fflush(stdout);

//      printf("updating strided space view\n");
//      fflush(stdout);

      Kokkos::parallel_for(8, KOKKOS_LAMBDA (const int i) {
         for (int r = 0; r < N; r++ ) {
            strided_space_view(i,r) = (Scalar)i * 5 + r;
		 }
      });
   }
   
   
   for (int r = 0; r < 1; r++) 
   {
      Kokkos::View< TestContainer<long>*, Kokkos::Experimental::EmuStridedSpace > strided_space_view( "strided", N );

      // have to initialize it first (view doesn't do that automatically right now.)
      printf("initializing container view: %d \n", N); fflush(stdout);
      
      //void* test_data = (void*)mw_malloc2d(NODELETS(), block_size * size_of_type);
      ConFunctor<long> cf(strided_space_view);
      //ConFunctor cf(test_data, block_size);
      Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Experimental::CilkPlus,ConFunctor<long>::TagInit>(0,N), cf);
     
      printf("using container view: %d \n", N); fflush(stdout);
      // now we can use it..
      Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Experimental::CilkPlus,ConFunctor<long>::TagUpdate>(0,N), cf);
      

   }
   
   
   Kokkos::fence();
   printf("Done Testing access to strided space view: %d\n", N);
   fflush(stdout);
   
}



TEST_F( TEST_CATEGORY, view_access )
{
  TestViewAccess< long, TEST_EXECSPACE >( 128 );
  TestViewAccess< long, TEST_EXECSPACE >( 8192 );
  TestViewAccess< long, TEST_EXECSPACE >( 16 );
}

}
