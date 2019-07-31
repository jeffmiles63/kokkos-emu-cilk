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

#ifndef KOKKOS_EMU_SIMPLETASKSCHEDULER_HPP
#define KOKKOS_EMU_SIMPLETASKSCHEDULER_HPP

//----------------------------------------------------------------------------

#include <Kokkos_Macros.hpp>
#if defined( KOKKOS_ENABLE_TASKDAG )

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_TaskScheduler_fwd.hpp>
//----------------------------------------------------------------------------
#include <impl/Kokkos_Tags.hpp>

#include <Kokkos_Future.hpp>
#include <impl/Kokkos_TaskQueue.hpp>
#include <impl/Kokkos_SingleTaskQueue.hpp>
#include <impl/Kokkos_MultipleTaskQueue.hpp>
#include <impl/Kokkos_TaskQueueMultiple.hpp>
#include <impl/Kokkos_TaskPolicyData.hpp>
#include <impl/Kokkos_TaskTeamMember.hpp>
#include <impl/Kokkos_EBO.hpp>

#include <intrinsics.h>

namespace Kokkos {
	
template <class ExecSpace>
class SimpleEmuTaskScheduler;
	
// For now, hack this in as a partial specialization
// TODO @tasking @cleanup Make this the "normal" class template and make the old code the specialization
template <typename ValueType, typename ExecutionSpace>
class BasicFuture<ValueType, SimpleEmuTaskScheduler<ExecutionSpace> >
{
public:

  using value_type = ValueType;
  using execution_space = ExecutionSpace;
  using scheduler_type = SimpleEmuTaskScheduler<ExecutionSpace>;
  using queue_type = typename scheduler_type::task_queue_type;


private:

  template <class>
  friend class SimpleEmuTaskScheduler; 
  template <class, class>
  friend class BasicFuture;

  using task_base_type = typename scheduler_type::task_base_type;
  using task_queue_type = typename scheduler_type::task_queue_type;

  using task_queue_traits = typename scheduler_type::task_queue_traits;
  using task_scheduling_info_type = typename scheduler_type::task_scheduling_info_type;

  using result_storage_type =
    Impl::TaskResultStorage<
      ValueType,
      Impl::SchedulingInfoStorage<
        Impl::RunnableTaskBase<task_queue_traits>,
        task_scheduling_info_type
      >
    >;


protected:
  OwningRawPtr<task_base_type> m_task = nullptr;
  
private:
  KOKKOS_INLINE_FUNCTION
  explicit
  BasicFuture(task_base_type* task)
    : m_task(task)
  {
    // Note: reference count starts at 2 to account for initial increment
    // TODO @tasking @minor DSH verify reference count here and/or encapsulate starting reference count closer to here
  }

public:

  KOKKOS_INLINE_FUNCTION
  BasicFuture() noexcept : m_task(nullptr) { }

  KOKKOS_INLINE_FUNCTION
  BasicFuture(BasicFuture&& rhs) noexcept
    : m_task(std::move(rhs.m_task))
  {
    rhs.m_task = nullptr;
  }

  KOKKOS_INLINE_FUNCTION
  BasicFuture(BasicFuture const& rhs)
    : m_task(rhs.m_task)
  {
    if(m_task) m_task->increment_reference_count();
  }

  KOKKOS_INLINE_FUNCTION
  BasicFuture& operator=(BasicFuture&& rhs) noexcept
  {
    if(m_task != rhs.m_task) {
      clear();
      m_task = std::move(rhs.m_task);
      // rhs.m_task reference count is unchanged, since this is a move
    }
    else {
      // They're the same, but this is a move, so 1 fewer references now
      rhs.clear();
    }
    rhs.m_task = nullptr;
    return *this ;
  }

  KOKKOS_INLINE_FUNCTION
  BasicFuture& operator=(BasicFuture const& rhs)
  {
    if(m_task != rhs.m_task) {
      clear();
      m_task = rhs.m_task;
      if(m_task != nullptr) { m_task->increment_reference_count(); }
    }
    return *this;
  }

  //----------------------------------------

  template <class T, class S>
  KOKKOS_INLINE_FUNCTION
  BasicFuture(BasicFuture<T, S>&& rhs) noexcept // NOLINT(google-explicit-constructor)
    : m_task(std::move(rhs.m_task))
  {
    static_assert(
      std::is_same<scheduler_type, void>::value ||
        std::is_same<scheduler_type, S>::value,
      "Moved Futures must have the same scheduler"
    );

    static_assert(
      std::is_same<value_type, void>::value ||
        std::is_same<value_type, T>::value,
      "Moved Futures must have the same value_type"
    );

    // reference counts are unchanged, since this is a move
    rhs.m_task = nullptr;
  }

  template <class T, class S>
  KOKKOS_INLINE_FUNCTION
  BasicFuture(BasicFuture<T, S> const& rhs) // NOLINT(google-explicit-constructor)
    : m_task(rhs.m_task)
  {
    static_assert(
      std::is_same<scheduler_type, void>::value ||
        std::is_same<scheduler_type, S>::value,
      "Copied Futures must have the same scheduler"
    );

    static_assert(
      std::is_same<value_type, void>::value ||
        std::is_same<value_type, T>::value,
      "Copied Futures must have the same value_type"
    );

    if(m_task) m_task->increment_reference_count();
  }

  template <class T, class S>
  KOKKOS_INLINE_FUNCTION
  BasicFuture&
  operator=(BasicFuture<T, S> const& rhs)
  {
    static_assert(
      std::is_same<scheduler_type, void>::value ||
        std::is_same<scheduler_type, S>::value,
      "Assigned Futures must have the same scheduler"
    );

    static_assert(
      std::is_same<value_type, void>::value ||
        std::is_same<value_type, T>::value,
      "Assigned Futures must have the same value_type"
    );

    if(m_task != rhs.m_task) {
      clear();
      m_task = rhs.m_task;
      if(m_task != nullptr) { m_task->increment_reference_count(); }
    }
    return *this;
  }

  template<class T, class S>
  KOKKOS_INLINE_FUNCTION
  BasicFuture& operator=(BasicFuture<T, S>&& rhs)
  {
    static_assert(
      std::is_same<scheduler_type, void>::value ||
        std::is_same<scheduler_type, S>::value,
      "Assigned Futures must have the same scheduler"
    );

    static_assert(
      std::is_same<value_type, void>::value ||
        std::is_same<value_type, T>::value,
      "Assigned Futures must have the same value_type"
    );

    if(m_task != rhs.m_task) {
      clear();
      m_task = std::move(rhs.m_task);
      // rhs.m_task reference count is unchanged, since this is a move
    }
    else {
      // They're the same, but this is a move, so 1 fewer references now
      rhs.clear();
    }
    rhs.m_task = nullptr;
    return *this ;
  }

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION
  ~BasicFuture() noexcept { clear(); }

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION
  void clear() noexcept {
    if(m_task) {
      bool should_delete = m_task->decrement_and_check_reference_count();
      if(should_delete) {
        static_cast<task_queue_type*>(m_task->ready_queue_base_ptr())
          ->deallocate(std::move(*m_task));
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  bool is_null() const noexcept {
    return m_task == nullptr;
  }


  KOKKOS_INLINE_FUNCTION
  bool is_ready() const noexcept {
    return (m_task == nullptr) || m_task->wait_queue_is_consumed();
  }

  KOKKOS_INLINE_FUNCTION
  const typename Impl::TaskResult< ValueType >::reference_type
  get() const
  {
    KOKKOS_EXPECTS(is_ready());
    return static_cast<result_storage_type*>(m_task)->value_reference();
    //return Impl::TaskResult<ValueType>::get(m_task);
  }

};

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

using cilk_queue_type = Impl::SingleTaskQueue<
    Kokkos::Experimental::CilkPlus,
    Kokkos::Experimental::EmuStridedSpace,
    Impl::TaskQueueTraitsLockBased
    >;

template <class ExecSpace>
class SimpleEmuTaskScheduler
  : public Impl::TaskSchedulerBase,
    protected Impl::ExecutionSpaceInstanceStorage<ExecSpace>,
    protected Impl::MemorySpaceInstanceStorage<Kokkos::Experimental::EmuStridedSpace>,
    private Impl::NoUniqueAddressMemberEmulation<typename cilk_queue_type::team_scheduler_info_type>
{
public:
  // TODO @tasking @generalization (maybe?) don't force QueueType to be complete here  
  using QueueType = cilk_queue_type;
  using queue_type = QueueType;
  using scheduler_type = SimpleEmuTaskScheduler; // tag as scheduler concept
  using execution_space = ExecSpace;
  using task_queue_type = QueueType;
  using memory_space = typename task_queue_type::memory_space;
  using memory_pool = typename task_queue_type::memory_pool;

  using team_scheduler_info_type = typename task_queue_type::team_scheduler_info_type;
  using task_scheduling_info_type = typename task_queue_type::task_scheduling_info_type;
  using specialization = Impl::TaskQueueSpecialization<SimpleEmuTaskScheduler>;
  using member_type = typename specialization::member_type;

  template <class Functor>
  using runnable_task_type = typename QueueType::template runnable_task_type<Functor, SimpleEmuTaskScheduler>;

  using task_base_type = typename task_queue_type::task_base_type;
  using runnable_task_base_type = typename task_queue_type::runnable_task_base_type;

  using task_queue_traits = typename QueueType::task_queue_traits;

  template <class ValueType>
  using future_type = Kokkos::BasicFuture<ValueType, SimpleEmuTaskScheduler>;
  template <class FunctorType>
  using future_type_for_functor = future_type<typename FunctorType::value_type>;

private:

  template <typename, typename>
  friend class BasicFuture;

  using track_type = Kokkos::Impl::SharedAllocationTracker;
  using execution_space_storage = Impl::ExecutionSpaceInstanceStorage<execution_space>;
  using memory_space_storage = Impl::MemorySpaceInstanceStorage<memory_space>;
  using team_scheduler_info_storage = Impl::NoUniqueAddressMemberEmulation<team_scheduler_info_type>;

  void * m_queue_rep = nullptr;
  long next_node = 0;
  
  KOKKOS_INLINE_FUNCTION
  task_queue_type* _get_queue( int i = 0 ) const {
	  if ( m_queue_rep != nullptr )
	  {
		  //Kokkos::Experimental::print_pointer( i, m_queue_rep, "task queue - head");
		  //fflush(stdout);
	      task_queue_type * rec = (task_queue_type*)&(((long**)m_queue_rep)[i][0]);
	      //Kokkos::Experimental::print_pointer( i, rec, "task queue - rec");
	      //fflush(stdout);

	      return rec; 
	  } else {
		  return nullptr;
	  }
  }
  
  KOKKOS_INLINE_FUNCTION
  long get_scheduler_node( void * ptr ) {
	  int nNode = mw_ptrtonodelet( ptr );
	  if ( nNode == 0 ) {
		  nNode = ATOMIC_ADDMS(&next_node, 1);		  
	  }
	  return nNode;
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr task_base_type* _get_task_ptr(std::nullptr_t) { return nullptr; }

  template <class ValueType>
  KOKKOS_INLINE_FUNCTION
  static constexpr task_base_type* _get_task_ptr(future_type<ValueType>&& f)
  {
    return f.m_task;
  }

  template <
    int TaskEnum,
    class DepTaskType,
    class FunctorType
  >
  KOKKOS_FUNCTION
  future_type_for_functor<typename std::decay<FunctorType>::type>
  _spawn_impl(
    DepTaskType arg_predecessor_task,
    TaskPriority arg_priority,
    typename runnable_task_base_type::function_type apply_function_ptr,
    typename runnable_task_base_type::destroy_type destroy_function_ptr,
    FunctorType&& functor
  )
  {
    KOKKOS_EXPECTS(m_queue_rep != nullptr);

    using functor_future_type = future_type_for_functor<typename std::decay<FunctorType>::type>;
    using task_type = typename task_queue_type::template runnable_task_type<
      FunctorType, scheduler_type
    >;
    
    printf("scheduler constructing runnable task \n");
    fflush(stdout);

    int nNode = get_scheduler_node( &arg_predecessor_task );
    
    task_queue_type* queue = _get_queue(nNode);
    
    // Reference count starts at two:
    //   +1 for the matching decrement when task is complete
    //   +1 for the future
    auto& runnable_task = *queue->template allocate_and_construct<task_type>(
      /* functor = */ std::forward<FunctorType>(functor),
      /* apply_function_ptr = */ apply_function_ptr,
      /* task_type = */ static_cast<Impl::TaskType>(TaskEnum),
      /* priority = */ arg_priority,
      /* queue_base = */ queue,
      /* initial_reference_count = */ 2
    );
    printf("ready to initialize the runnable task \n");
    fflush(stdout);

    if(arg_predecessor_task != nullptr) {
      queue->initialize_scheduling_info_from_predecessor(
        runnable_task, *arg_predecessor_task
      );
      runnable_task.set_predecessor(*arg_predecessor_task);
    }
    else {
      queue->initialize_scheduling_info_from_team_scheduler_info(
        runnable_task, team_scheduler_info()
      );
    }

    auto rv = functor_future_type(&runnable_task);

    Kokkos::memory_fence(); // fence to ensure dependent stores are visible

    queue->schedule_runnable(
      std::move(runnable_task),
      team_scheduler_info()
    );
    // note that task may be already completed even here, so don't touch it again

    return rv;
  }


public:

  //----------------------------------------------------------------------------
  // <editor-fold desc="Constructors, destructor, and assignment"> {{{2

  SimpleEmuTaskScheduler() = delete;
  
  KOKKOS_INLINE_FUNCTION
  SimpleEmuTaskScheduler( const SimpleEmuTaskScheduler& rhs ) : execution_space_storage(rhs),
                                                          memory_space_storage(rhs),
                                                          m_queue_rep( rhs.m_queue_rep ) {
															  //printf("emu task scheduler copy Constructor called: %08x ", m_queue_rep);
															  //fflush(stdout);															  
														  }

  explicit
  SimpleEmuTaskScheduler(
    execution_space const& arg_execution_space,
    memory_space const& arg_memory_space,
    memory_pool const& arg_memory_pool
  ) : execution_space_storage(arg_execution_space),
      memory_space_storage(arg_memory_space)
  {
    // Ask the task queue how much space it needs (usually will just be
    // sizeof(task_queue_type), but some queues may need additional storage
    // dependent on runtime conditions or properties of the execution space)
    auto const allocation_size = task_queue_type::task_queue_allocation_size(
      arg_execution_space,
      arg_memory_space,
      arg_memory_pool
    );
        
    //printf("allocating queue memory %d \n", allocation_size);
    //fflush(stdout);
    
    m_queue_rep = arg_memory_space.allocate( allocation_size * NODELETS() );  // create strided space as big as allocation size for each node 
      
    for ( int i = 0; i < NODELETS(); i++) { 
		long* refPtr = Kokkos::Experimental::EmuReplicatedSpace::getRefAddr();
		
		cilk_spawn_at(&refPtr[i]) initialize_queue( i, arg_execution_space,
		                                                    arg_memory_space,
		                                                    arg_memory_pool );
	}
	cilk_sync;
  }
  
  KOKKOS_INLINE_FUNCTION
  void initialize_queue( int i, execution_space const& arg_execution_space,
                                     memory_space const& arg_memory_space,
                                     memory_pool const& arg_memory_pool ) {
										 
      auto * loc_rec = (task_queue_type*)&(((long**)m_queue_rep)[i][0]);
      new (loc_rec) task_queue_type(
                            arg_execution_space,
                            arg_memory_space,
                            arg_memory_pool
                            );
  }

  explicit
  SimpleEmuTaskScheduler(
    execution_space const& arg_execution_space,
    memory_pool const& pool
  ) : SimpleEmuTaskScheduler(arg_execution_space, memory_space{}, pool)
  { /* forwarding ctor, must be empty */ }

  explicit
  SimpleEmuTaskScheduler(memory_pool const& pool)
    : SimpleEmuTaskScheduler(execution_space{}, memory_space{}, pool)
  { /* forwarding ctor, must be empty */ }

  SimpleEmuTaskScheduler(
    memory_space const & arg_memory_space,
    size_t const mempool_capacity,
    unsigned const mempool_min_block_size, // = 1u << 6
    unsigned const mempool_max_block_size, // = 1u << 10
    unsigned const mempool_superblock_size // = 1u << 12
  ) : SimpleEmuTaskScheduler(
        execution_space{},
        arg_memory_space,
        memory_pool(
          arg_memory_space, mempool_capacity, mempool_min_block_size,
          mempool_max_block_size, mempool_superblock_size
        )
      )
  { /* forwarding ctor, must be empty */ }

  // </editor-fold> end Constructors, destructor, and assignment }}}2
  //----------------------------------------------------------------------------

  // Note that this is an expression of shallow constness
  KOKKOS_INLINE_FUNCTION
  task_queue_type& queue() const
  {
    KOKKOS_EXPECTS(m_queue_rep != nullptr);
    return *(_get_queue(0));
  }
  
  // Note that this is an expression of shallow constness
  KOKKOS_INLINE_FUNCTION
  task_queue_type& queue(int rank_in_league) const
  {
    KOKKOS_EXPECTS(m_queue_rep != nullptr);
    return *(_get_queue(rank_in_league));
  }
  

  KOKKOS_INLINE_FUNCTION
  SimpleEmuTaskScheduler
  get_team_scheduler(int rank_in_league) const noexcept
  {
	//printf("get team scheduler: %d \n", rank_in_league);
	//fflush(stdout);
    KOKKOS_EXPECTS(m_queue_rep != nullptr);
    auto rv = SimpleEmuTaskScheduler{ *this };
	//printf("team scheduler get queue: %d \n", rank_in_league);
	//fflush(stdout);
    
    task_queue_type* queue = _get_queue(rank_in_league);    
	//printf("get team scheduler info: %d \n", rank_in_league);
	//fflush(stdout);
    
    rv.team_scheduler_info() = queue->initial_team_scheduler_info(rank_in_league);
    return rv;
  }

  KOKKOS_INLINE_FUNCTION
  execution_space const& get_execution_space() const { return this->execution_space_instance(); }

  KOKKOS_INLINE_FUNCTION
  team_scheduler_info_type& team_scheduler_info() &
  {
    return this->team_scheduler_info_storage::no_unique_address_data_member();
  }

  KOKKOS_INLINE_FUNCTION
  team_scheduler_info_type const& team_scheduler_info() const &
  {
    return this->team_scheduler_info_storage::no_unique_address_data_member();
  }

  template <int TaskEnum, typename DepFutureType, typename FunctorType>
  KOKKOS_FUNCTION
  static
  Kokkos::BasicFuture<typename FunctorType::value_type, scheduler_type>
  spawn(
    Impl::TaskPolicyWithScheduler<TaskEnum, scheduler_type, DepFutureType>&& arg_policy,
    typename runnable_task_base_type::function_type arg_function,
    typename runnable_task_base_type::destroy_type arg_destroy,
    FunctorType&& arg_functor
  )
  {
    return std::move(arg_policy.scheduler()).template _spawn_impl<TaskEnum>(
      _get_task_ptr(std::move(arg_policy.predecessor())),
      arg_policy.priority(),
      arg_function,
      arg_destroy,
      std::forward<FunctorType>(arg_functor)
    );
  }

  template <int TaskEnum, typename DepFutureType, typename FunctorType>
  KOKKOS_FUNCTION
  Kokkos::BasicFuture<typename FunctorType::value_type, scheduler_type>
  spawn(
    Impl::TaskPolicyWithPredecessor<TaskEnum, DepFutureType>&& arg_policy,
    FunctorType&& arg_functor
  )
  {
    static_assert(
      std::is_same<typename DepFutureType::scheduler_type, scheduler_type>::value,
      "Can't create a task policy from a scheduler and a future from a different scheduler"
    );

    using task_type = runnable_task_type<FunctorType>;
    typename task_type::function_type const ptr = task_type::apply;
    typename task_type::destroy_type const dtor = task_type::destroy;

    return _spawn_impl<TaskEnum>(
      std::move(arg_policy).predecessor().m_task,
      arg_policy.priority(),
      ptr, dtor,
      std::forward<FunctorType>(arg_functor)
    );
  }

  template <class FunctorType, class ValueType, class Scheduler>
  KOKKOS_FUNCTION
  static void
  respawn(
    FunctorType* functor,
    BasicFuture<ValueType, Scheduler> const& predecessor,
    TaskPriority priority = TaskPriority::Regular
  ) {
    using task_type = typename task_queue_type::template runnable_task_type<
      FunctorType, scheduler_type
    >;

    auto& task = *static_cast<task_type*>(functor);
    task.set_priority(priority);
    task.set_predecessor(*predecessor.m_task);
    task.set_respawn_flag(true);
  }

  template <class FunctorType, class ValueType, class Scheduler>
  KOKKOS_FUNCTION
  static void
  respawn(
    FunctorType* functor,
    scheduler_type const&,
    TaskPriority priority = TaskPriority::Regular
  ) {
    using task_type = typename task_queue_type::template runnable_task_type<
      FunctorType, scheduler_type
    >;

    auto& task = *static_cast<task_type*>(functor);
    task.set_priority(priority);
    KOKKOS_ASSERT(not task.has_predecessor());
    task.set_respawn_flag(true);
  }


  template <class ValueType>
  KOKKOS_FUNCTION
  future_type<void>
  when_all(BasicFuture<ValueType, scheduler_type> const predecessors[], int n_predecessors) {

    // TODO @tasking @generalization DSH propagate scheduling info

    using task_type = typename task_queue_type::aggregate_task_type;

    future_type<void> rv;

    if(n_predecessors > 0) {
      task_queue_type* queue_ptr = nullptr;

      // Loop over the predecessors to find the queue and increment the reference
      // counts
      for(int i_pred = 0; i_pred < n_predecessors; ++i_pred) {

        auto* predecessor_task_ptr = predecessors[i_pred].m_task;

        if(predecessor_task_ptr != nullptr) {
          // TODO @tasking @cleanup DSH figure out when this is allowed to be nullptr (if at all anymore)

          // Increment reference count to track subsequent assignment.
          // TODO @tasking @optimization DSH figure out if this reference count increment is necessary
          predecessor_task_ptr->increment_reference_count();

          // TODO @tasking @cleanup DSH we should just set a boolean here instead to make this more readable
          int nNode = mw_ptrtonodelet( predecessor_task_ptr );
     
          queue_ptr = _get_queue(nNode);
        }

      } // end loop over predecessors
      

      // This only represents a non-ready future if at least one of the predecessors
      // has a task (and thus, a queue)
      if(queue_ptr != nullptr) {
        auto& q = *queue_ptr;

        printf("constructing aggregate task \n");
        fflush(stdout);


        auto* aggregate_task_ptr = q.template allocate_and_construct_with_vla_emulation<
          task_type, task_base_type*
        >(
          /* n_vla_entries = */ n_predecessors,
          /* aggregate_predecessor_count = */ n_predecessors,
          /* queue_base = */ &q,
          /* initial_reference_count = */ 2
        );
        
        printf("future reference aggregate task \n");
        fflush(stdout);        

        rv = future_type<void>(aggregate_task_ptr);

        for(int i_pred = 0; i_pred < n_predecessors; ++i_pred) {
          aggregate_task_ptr->vla_value_at(i_pred) = predecessors[i_pred].m_task;
        }

        Kokkos::memory_fence(); // we're touching very questionable memory, so be sure to fence

        q.schedule_aggregate(std::move(*aggregate_task_ptr), team_scheduler_info());
        // the aggregate may be processed at any time, so don't touch it after this
      }
    }

    return rv;
  }

  template <class F>
  KOKKOS_FUNCTION
  future_type<void>
  when_all(int n_calls, F&& func)
  {
    // TODO @tasking @generalization DSH propagate scheduling info?

    // later this should be std::invoke_result_t
    using generated_type = decltype(func(0));
    using task_type = typename task_queue_type::aggregate_task_type;

    static_assert(
      is_future<generated_type>::value,
      "when_all function must return a Kokkos future (an instance of Kokkos::BasicFuture)"
    );
    static_assert(
      std::is_base_of<scheduler_type, typename generated_type::scheduler_type>::value,
      "when_all function must return a Kokkos::BasicFuture of a compatible scheduler type"
    );

    task_queue_type* queue = _get_queue(0);
    auto* aggregate_task = queue->template allocate_and_construct_with_vla_emulation<
      task_type, task_base_type*
    >(
      /* n_vla_entries = */ n_calls,
      /* aggregate_predecessor_count = */ n_calls,
      /* queue_base = */ queue,
      /* initial_reference_count = */ 2
    );

    auto rv = future_type<void>(aggregate_task);

    for(int i_call = 0; i_call < n_calls; ++i_call) {

      auto generated_future = func(i_call);

      if(generated_future.m_task != nullptr) {
        generated_future.m_task->increment_reference_count();
        aggregate_task->vla_value_at(i_call) = generated_future.m_task;

        KOKKOS_ASSERT(queue == generated_future.m_task->ready_queue_base_ptr()
          && "Queue mismatch in when_all"
        );
      }

    }

    Kokkos::memory_fence();

    queue->schedule_aggregate(std::move(*aggregate_task), team_scheduler_info());
    // This could complete at any moment, so don't touch anything after this

    return rv;
  }

};

template<class Space>
struct is_scheduler<SimpleEmuTaskScheduler<Space>> : public std::true_type {};


template<class ExecSpace>
inline
void wait(SimpleEmuTaskScheduler<ExecSpace> const& scheduler)
{
  using scheduler_type = SimpleEmuTaskScheduler<ExecSpace>;
  scheduler_type::specialization::execute(scheduler);
}

} // namespace Kokkos

//----------------------------------------------------------------------------
//---------------------------------------------------------------------------#endif /* #if defined( KOKKOS_ENABLE_TASKDAG ) */

#endif /* #if defined( KOKKOS_ENABLE_TASKDAG ) */
#endif /* #ifndef KOKKOS_EMU_SIMPLETASKSCHEDULER_HPP */

