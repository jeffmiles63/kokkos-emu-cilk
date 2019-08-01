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

#ifndef KOKKOS_IMPL_CILKPLUS_TASK_HPP
#define KOKKOS_IMPL_CILKPLUS_TASK_HPP

#include <Kokkos_Macros.hpp>
#include <Kokkos_TaskScheduler_fwd.hpp>
#include <impl/Kokkos_TaskQueue.hpp>
#include <impl/Kokkos_HostThreadTeam.hpp>


namespace Kokkos {
	
namespace Impl {

/**\brief  Impl::TaskExec<CilkPlus> is the TaskScheduler<CilkPlus>::member_type
 *         passed to tasks running in a CilkPlus space.
 *
 *    team member
 *    team size
 *    league rank
 *    league size
 *
 */
template <class Scheduler>
class TaskExec<Kokkos::Experimental::CilkPlus, Scheduler> {
 private:
  enum { emu_league_size = 8 };

  TaskExec(TaskExec&&)      = delete;
  TaskExec(TaskExec const&) = delete;
  TaskExec& operator=(TaskExec&&) = delete;
  TaskExec& operator=(TaskExec const&) = delete;

  const int m_team_rank;
  const int m_team_size;  
  const int m_league_rank;
  Scheduler m_scheduler;
  
 public:
  
  KOKKOS_INLINE_FUNCTION
  TaskExec(Scheduler const& parent_scheduler, int arg_team_rank = 0, 
           int arg_team_size = 1, int arg_league_rank = 0)
      : m_team_rank(arg_team_rank),
        m_team_size(arg_team_size),
        m_league_rank(arg_league_rank),
        m_scheduler(parent_scheduler.get_team_scheduler(league_rank())) {}

 public:
  using thread_team_member = TaskExec;

  int team_rank() const { return m_team_rank; }
  int team_size() const { return m_team_size; }
  int league_rank() const { return m_league_rank; }
  int league_size() const { return emu_league_size; }

  void team_barrier() const {
    if (1 < m_team_size) {
      cilk_sync;
    }
  }

  template <class ValueType>
  void team_broadcast(ValueType& val, const int thread_id) const {
    
  }

  KOKKOS_INLINE_FUNCTION Scheduler const& scheduler() const noexcept {
    return m_scheduler;
  }
  KOKKOS_INLINE_FUNCTION Scheduler& scheduler() noexcept { return m_scheduler; }
};


/*template <>
struct TaskQueueSpecialization<
  Kokkos::SimpleTaskScheduler<Kokkos::Experimental::CilkPlus, Impl::SingleTaskQueue<
    Kokkos::Experimental::CilkPlus,
    Kokkos::Experimental::EmuStridedSpace,
    Impl::TaskQueueTraitsLockBased
    > >
>*/
template<>
class TaskQueueSpecialization<
  SimpleEmuTaskScheduler<Kokkos::Experimental::CilkPlus>
>
{
public:

  using execution_space = Kokkos::Experimental::CilkPlus;
  using scheduler_type = SimpleEmuTaskScheduler<Kokkos::Experimental::CilkPlus>;
  using member_type     = TaskExec<Kokkos::Experimental::CilkPlus, scheduler_type>;
  using memory_space = Kokkos::Experimental::EmuStridedSpace;
  
  using queue_type = typename scheduler_type::task_queue_type; 
  using task_base_type = typename queue_type::task_base_type;  
  static const int layer_width = 8;
  
  static
  void iff_single_thread_recursive_execute( scheduler_type const& scheduler ) {    
    
  }

  static void launch_task( void * ptr, int offset, int i, int n, scheduler_type const& scheduler, long* data_ref) {
	  int ndx = offset + i * layer_width + n;  // i is nodelet, n is layer index
	  	  
	  task_base_type * task_ptr = (task_base_type *)ptr;
	  printf("inside task thread %d, %d : %08x \n", i, n, task_ptr);
	  fflush(stdout);
	  

         	  
	  auto& queue = scheduler.queue(i);    
      auto team_scheduler = scheduler.get_team_scheduler(i);
      member_type member(scheduler, n, layer_width, i);
	  
	  if ( task_ptr ) {
		  		  
		 printf("task run [%d,%d] -->\n", i, n);
		 fflush(stdout);
		 
         auto current_task = OptionalRef<task_base_type>(*task_ptr);	  
         current_task->as_runnable_task().run(member);
         
		 printf("task complete [%d,%d] -->\n", i, n);
		 fflush(stdout);
     
          // Respawns are handled in the complete function
          queue.complete(
             (*std::move(current_task)).as_runnable_task(),
             team_scheduler.team_scheduler_info()
          );	  
 	  } else {
		  printf("no task for this thread: %d \n", ndx);
		  fflush(stdout);
	  }
	  
  }

  
  static void team_task_head(int offset, int i, scheduler_type const& scheduler, long* data_ref) {
	  printf("team task head: %d, %d, %d.  \n", offset, i );
	  fflush(stdout);
	  	  	  	  
	  auto& queue = scheduler.queue(i);
      auto team_scheduler = scheduler.get_team_scheduler(i);
      
      printf("head [%d] entering queue processing loop \n", i );
	  fflush(stdout);      
           
      int n = 0;
      while ( (!queue.is_done()) && n < layer_width ) {
		  
		 //printf("head [%d] looking for task %d \n", i, n );
		 //fflush(stdout);
		 auto current_task = OptionalRef<task_base_type>(nullptr);
         current_task = queue.pop_ready_task(team_scheduler.team_scheduler_info());
         //printf("[%d] task head returned from pop_ready_task \n", i);
         //fflush(stdout);
	  
	     if ( current_task.get() != nullptr ) {
		  	 printf("head [%d] launching task thread %d : %08x \n", i, n, current_task.get());
		  	 fflush(stdout);
		  	 
		  	 void* ptr = (void*)current_task.get();
             
		     cilk_spawn_at(ptr) launch_task( ptr, offset, i, n, scheduler, data_ref ); 
		     n++;

	      }
	      
	      RESCHEDULE();
	    
 	  }
  }
  
  static bool all_queues_are_done(scheduler_type const& scheduler) {
	  bool bAllDone = true;
	  for ( int i = 0; i < NODELETS(); i++ ) {
		  if (!scheduler.queue(i).is_done()) {
			  bAllDone = false;
			  break;
		  }
	  }
	  return bAllDone;
  }

  // Must provide task queue execution function
  static void execute(scheduler_type const& scheduler)
  {

    // Set default buffers
    serial_resize_thread_team_data(
      0,   /* global reduce buffer */
      512, /* team reduce buffer */
      0,   /* team shared buffer */
      0    /* thread local buffer */
    );

    long * data_ref = mw_malloc1dlong(NODELETS());

    int offset = 0;
    while(not all_queues_are_done(scheduler)) {

       printf("cilk task exec loop: %d \n", offset);
       fflush(stdout);
       // blocks of 64 ... for now.  if the queue doesn't have 64, then they should all just return...
       for ( int i = 0; i < NODELETS(); i++ ) {
          cilk_spawn_at(&data_ref[i]) team_task_head( offset, i, scheduler, data_ref );
       }
       cilk_sync;
       offset += 64;
    }
    
  }
  
  template< typename TaskType >
  static void
  get_function_pointer(
    typename TaskType::function_type& ptr,
    typename TaskType::destroy_type& dtor
  ) { 
    ptr = TaskType::apply;
    dtor = TaskType::destroy;
  }  
  
};

template<class Scheduler>
class TaskQueueSpecializationConstrained<
  Scheduler,
  typename std::enable_if<
    std::is_same<typename Scheduler::execution_space, Kokkos::Experimental::CilkPlus>::value
  >::type
>
{
public:

  // Note: Scheduler may be an incomplete type at class scope (but not inside
  // of the methods, obviously)

  using execution_space = Kokkos::Experimental::CilkPlus;
  using memory_space = Kokkos::Experimental::EmuStridedSpace;
  using scheduler_type = Scheduler;
  using member_type = TaskTeamMemberAdapter<
    HostThreadTeamMember<Kokkos::Experimental::CilkPlus>, scheduler_type
  >;
  
  using queue_type = typename scheduler_type::task_queue_type; 
  using task_base_type = typename queue_type::task_base_type;  
  static const int layer_width = 8;
    

  static
  void iff_single_thread_recursive_execute(scheduler_type const& scheduler) {


  }

  static void launch_task(int offset, int layer, int i, scheduler_type const& scheduler, long* data_ref) {
	  if (layer >= 0) {
          // blocks of 64 ... for now.  if the queue doesn't have 64, then they should all just return...
          for ( int n = 0; n < layer_width; n++ ) {
             cilk_spawn launch_task( offset + i*layer_width, 0, n, scheduler, data_ref );      
          }		  
	  }
	  
	  int ndx = offset + i; 	  
	  printf("task thread: %d\n", ndx);
	  auto& queue = scheduler.queue();
      Impl::HostThreadTeamData& self = *Impl::serial_get_thread_team_data();
      auto team_scheduler = scheduler.get_team_scheduler(ndx);
      member_type member(scheduler, self);

      auto current_task = OptionalRef<task_base_type>(nullptr);
      current_task = queue.pop_ready_task(team_scheduler.team_scheduler_info());    
	  
	  if ( current_task ) {
         current_task->as_runnable_task().run(member);
     
         // Respawns are handled in the complete function
         queue.complete(
            (*std::move(current_task)).as_runnable_task(),
            team_scheduler.team_scheduler_info()
         );	  
	  }
  }

  // Must provide task queue execution function
  static void execute(scheduler_type const& scheduler)
  {

    // Set default buffers
    serial_resize_thread_team_data(
      0,   /* global reduce buffer */
      512, /* team reduce buffer */
      0,   /* team shared buffer */
      0    /* thread local buffer */
    );

    auto& queue = scheduler.queue();
    long * data_ref = mw_malloc1dlong(NODELETS());

    int offset = 0;
    while(not queue.is_done()) {

       // blocks of 64 ... for now.  if the queue doesn't have 64, then they should all just return...
       for ( int i = 0; i < NODELETS(); i++ ) {
          cilk_spawn_at(&data_ref[i]) launch_task( offset, 1, i, scheduler, data_ref );
       }
       cilk_sync;
       offset += 64;
    }
  }

  template <typename TaskType>
  static void
  get_function_pointer(
    typename TaskType::function_type& ptr,
    typename TaskType::destroy_type& dtor
  )
  {
    ptr = TaskType::apply;
    dtor = TaskType::destroy;
  }
};

extern template class TaskQueue< Kokkos::Experimental::CilkPlus, Kokkos::Experimental::EmuStridedSpace > ;


} // Impl
} //Kokkos

#endif
