
#include <stdexcept>
#include <limits>
#include <math.h>

#include <Kokkos_Core.hpp>

namespace Kokkos {
namespace Experimental {
   extern void * ers;
   extern replicated EmuReplicatedSpace els;   
}}

namespace test {

template< class Scalar, class ExecSpace >
void TestViewAccess( int N ) {
   //starttiming();
   
   Kokkos::View< Scalar*, Kokkos::HostSpace > hs_view( "host", N );
   printf ("host view size is %ld \n", (unsigned long)sizeof(hs_view));
   fflush(stdout);

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
    

   printf("Testing 2D view...\n");
   fflush(stdout);   
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
	  Scalar upd = hs_view(i);
	  //printf("[%d] updating reduce var %d, %ld\n", NODE_ID(), i, upd);
	  //fflush(stdout);	  
      update += upd;
      //printf("[%d] finished updating %d\n", NODE_ID(), i);
	  //fflush(stdout);	        
   }, total );
   Kokkos::fence();   
   printf("parallel reduce test complete %d\n", N);
   fflush(stdout);      
   

//   long * refPtr = Kokkos::Experimental::EmuReplicatedSpace::getRefAddr();
/*   for (int i = 0; i < Kokkos::Experimental::EmuReplicatedSpace::memory_zones(); i++) {
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

      Kokkos::parallel_for (N, KOKKOS_LAMBDA (int i) {
         replicated_space_view(i) = (Scalar)i;
      });
   }
 */
   
   
   Kokkos::fence();
   {
      Kokkos::View< Scalar*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::ForceRemote> > global_space_view( "global", N );
      printf ("global view size is %ld \n", (unsigned long)sizeof(global_space_view));
//      Kokkos::View< Scalar*, Kokkos::HostSpace, Kokkos::MemoryTraits<0> > global_space_view( "global", N );      


      printf("Testing access to global space view\n");
      fflush(stdout);

      Kokkos::parallel_for(N, KOKKOS_LAMBDA (const int i) {
//      for (int i = 0; i < N; i++) {
         global_space_view(i) = (Scalar)i;
      });
//      }
   }
   Kokkos::fence();
   printf("Testing access to strided space view\n");
   fflush(stdout);
      
   {
      Kokkos::View< Scalar*, Kokkos::Experimental::EmuStridedSpace > strided_space_view( "strided", N );
      printf ("strided view size is %ld \n", (unsigned long)sizeof(strided_space_view));
      fflush(stdout);

      printf("updating strided space view\n");
      fflush(stdout);

      Kokkos::parallel_for(N, KOKKOS_LAMBDA (const int i) {      
//      for (int i = 0; i < N; i++ ) {
         strided_space_view(i) = (Scalar)i;
      });
//      }
   }
   {
      Kokkos::View< Scalar**, Kokkos::Experimental::EmuStridedSpace > strided_space_view( "2D strided", 8, N );
      printf ("strided view size is %ld \n", (unsigned long)sizeof(strided_space_view));
      fflush(stdout);

      printf("updating strided space view\n");
      fflush(stdout);

      Kokkos::parallel_for(8, KOKKOS_LAMBDA (const int i) {
         for (int r = 0; r < N; r++ ) {
            strided_space_view(i,r) = (Scalar)i * 5 + r;
		 }
      });
   }
   
   Kokkos::fence();
   printf("Done Testing access to strided space view\n");
   fflush(stdout);
   
}



TEST_F( TEST_CATEGORY, view_access )
{
  TestViewAccess< long, TEST_EXECSPACE >( 128 );
  TestViewAccess< long, TEST_EXECSPACE >( 8192 );
  TestViewAccess< long, TEST_EXECSPACE >( 16 );
}

}
