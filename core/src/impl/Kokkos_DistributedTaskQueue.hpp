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

#ifndef KOKKOS_IMPL_DistributedTaskQueue_HPP
#define KOKKOS_IMPL_DistributedTaskQueue_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_TASKDAG)

#include <Kokkos_TaskScheduler_fwd.hpp>
#include <Kokkos_Core_fwd.hpp>

#include <Kokkos_MemoryPool.hpp>

#include <impl/Kokkos_TaskBase.hpp>
#include <impl/Kokkos_TaskResult.hpp>

#include <impl/Kokkos_TaskQueueMemoryManager.hpp>
#include <impl/Kokkos_TaskQueueCommon.hpp>
#include <impl/Kokkos_SingleTaskQueue.hpp>
#include <impl/Kokkos_Memory_Fence.hpp>
#include <impl/Kokkos_Atomic_Increment.hpp>
#include <impl/Kokkos_OptionalRef.hpp>
#include <impl/Kokkos_LIFO.hpp>

#include <string>
#include <typeinfo>
#include <stdexcept>

//#define ENABLE_TASK_STEALING
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
// Distributed task queue ... manage multiple queues on strided memory space
//----------------------------------------------------------------------------
template <class ExecSpace, class MemorySpace, class TaskQueueTraits,
          class MemoryPool =
              Kokkos::MemoryPool<Kokkos::Device<ExecSpace, MemorySpace>>>
class DistributedTaskQueue final
    : public TaskQueueMemoryManager<ExecSpace, MemorySpace, MemoryPool>,
      public TaskQueueCommonMixin<DistributedTaskQueue<
          ExecSpace, MemorySpace, TaskQueueTraits, MemoryPool>> {
 public:
  using execution_space   = ExecSpace;
  using memory_space      = MemorySpace;
  using memory_pool       = MemoryPool;
  using task_queue_type   = DistributedTaskQueue;  // mark as task_queue concept
  using task_queue_traits = TaskQueueTraits;
  using task_base_type    = TaskNode<TaskQueueTraits>;
  using ready_queue_type =
      typename TaskQueueTraits::template ready_queue_type<task_base_type>;

 private:
  using base_t = TaskQueueMemoryManager<ExecSpace, MemorySpace, MemoryPool>;
  using common_mixin_t = TaskQueueCommonMixin<DistributedTaskQueue>;
  using single_queue_type =
      Impl::SingleTaskQueue<Kokkos::Experimental::CilkPlus,
                            Kokkos::Experimental::EmuStridedSpace,
                            Impl::TaskQueueTraitsLockBased>;

  int league_size   = 1;
  void* m_queue_rep = nullptr;

 public:
  struct SchedulerInfo {
    using team_queue_id_t                             = int32_t;
    static constexpr team_queue_id_t NoAssociatedTeam = -1;
    team_queue_id_t team_association                  = NoAssociatedTeam;

    using scheduler_info_type = SchedulerInfo;

    KOKKOS_INLINE_FUNCTION
    constexpr explicit SchedulerInfo(team_queue_id_t association) noexcept
        : team_association(association) {}

    KOKKOS_INLINE_FUNCTION
    SchedulerInfo() = default;

    KOKKOS_INLINE_FUNCTION
    SchedulerInfo(SchedulerInfo const&) = default;

    KOKKOS_INLINE_FUNCTION
    SchedulerInfo(SchedulerInfo&&) = default;

    KOKKOS_INLINE_FUNCTION
    SchedulerInfo& operator=(SchedulerInfo const&) = default;

    KOKKOS_INLINE_FUNCTION
    SchedulerInfo& operator=(SchedulerInfo&&) = default;

    KOKKOS_INLINE_FUNCTION
    ~SchedulerInfo() = default;
  };

  using task_scheduling_info_type = typename std::conditional<
      TaskQueueTraits::ready_queue_insertion_may_fail,
      FailedQueueInsertionLinkedListSchedulingInfo<TaskQueueTraits>,
      EmptyTaskSchedulingInfo>::type;
  using team_scheduler_info_type = SchedulerInfo;

  using runnable_task_base_type = RunnableTaskBase<TaskQueueTraits>;

  template <class Functor, class Scheduler>
  // requires TaskScheduler<Scheduler> && TaskFunctor<Functor>
  using runnable_task_type =
      RunnableTask<task_queue_traits, Scheduler, typename Functor::value_type,
                   Functor>;

  using aggregate_task_type =
      AggregateTask<task_queue_traits, task_scheduling_info_type>;

 public:
  //----------------------------------------------------------------------------
  // <editor-fold desc="Constructors, destructors, and assignment"> {{{2

  DistributedTaskQueue() = delete;
  DistributedTaskQueue(DistributedTaskQueue const& rhs)
      : base_t(rhs),
        league_size(rhs.league_size),
        m_queue_rep(rhs.m_queue_rep) {}
  DistributedTaskQueue(DistributedTaskQueue&& rhs)
      : base_t(std::move(rhs)),
        league_size(std::move(rhs.league_size)),
        m_queue_rep(std::move(rhs.m_queue_rep)) {}
  DistributedTaskQueue& operator=(DistributedTaskQueue const&) = delete;
  DistributedTaskQueue& operator=(DistributedTaskQueue&&) = delete;

  DistributedTaskQueue(
      typename base_t::execution_space const& arg_execution_space,
      typename base_t::memory_space const& arg_memory_space,
      typename base_t::memory_pool const& arg_memory_pool,
      const int league_size_)
      : base_t(arg_memory_pool), league_size(league_size_) {
    // Ask the single task queue how much space it needs (usually will just be
    // sizeof(task_queue_type), but some queues may need additional storage
    // dependent on runtime conditions or properties of the execution space)
    auto const allocation_size = single_queue_type::task_queue_allocation_size(
        arg_execution_space, arg_memory_space, arg_memory_pool);

    // printf("allocating queue memory %d \n", allocation_size);
    // fflush(stdout);

    m_queue_rep = arg_memory_space.allocate(
        allocation_size * league_size);  // create strided space as big as
                                         // allocation size for each node
    long* refPtr = Kokkos::Experimental::EmuReplicatedSpace::getRefAddr();

    for (int i = 0; i < league_size; i++) {
      cilk_spawn_at(&refPtr[i]) initialize_queue(
          i, arg_execution_space, arg_memory_space, arg_memory_pool);
    }
    cilk_sync;
  }

  // Initialize strided queue
  KOKKOS_INLINE_FUNCTION
  void initialize_queue(int i, execution_space const& arg_execution_space,
                        memory_space const& arg_memory_space,
                        memory_pool const& arg_memory_pool) {
    auto* loc_rec = (single_queue_type*)&(((long**)m_queue_rep)[i][0]);
    new (loc_rec) single_queue_type(arg_execution_space, arg_memory_space,
                                    arg_memory_pool);
    loc_rec->set_queue_id(i);
  }

  // This would be more readable with a lambda, but that comes with
  // all the baggage associated with a lambda (compilation times, bugs with
  // nvcc, etc.), so we'll use a simple little helper functor here.
  template <class task_queue_traits, class DerivedDist, class TeamSchedulerInfo>
  struct _dist_schedule_waiting_tasks_operation {
    TaskNode<task_queue_traits> const& m_predecessor;
    DerivedDist& m_queue;
    TeamSchedulerInfo const& m_info;
    KOKKOS_INLINE_FUNCTION
    void operator()(TaskNode<TaskQueueTraits>&& task) const noexcept
    // requires Same<TaskType, Derived::task_base_type>
    {
      int new_team_assoc = mw_ptrtonodelet(&task);
      // printf("Schedule waiting operation: 0x%lx %d, %d \n", &task,
      // task.node_id, new_team_assoc); fflush(stdout);
      using task_scheduling_info_type =
          typename DerivedDist::task_scheduling_info_type;
      if (task.is_runnable())  // KOKKOS_LIKELY
      {
        // printf("waiting is runnable: %d \n", task.node_id);
        // fflush(stdout);

        // TODO @tasking @optimiazation DSH check this outside of the loop ?
        if (m_predecessor.is_runnable()) {
          // printf("waiting pred is runnable: %d %d\n", task.node_id,
          // m_predecessor.node_id);
          // fflush(stdout);

          m_queue.update_scheduling_info_from_completed_predecessor(
              /* ready_task = */ task.as_runnable_task(),
              /* predecessor = */ m_predecessor.as_runnable_task());
        } else {
          // printf("waiting pred is aggr: %d %d\n", task.node_id,
          // m_predecessor.node_id);
          // fflush(stdout);

          KOKKOS_ASSERT(m_predecessor.is_aggregate());
          m_queue.update_scheduling_info_from_completed_predecessor(
              /* ready_task = */ task.as_runnable_task(),
              /* predecessor = */ m_predecessor
                  .template as_aggregate<task_scheduling_info_type>());
        }
        // printf("dist adding waiting runnable %d to ready queue %d\n",
        // task.node_id, new_team_assoc); fflush(stdout);
        m_queue.schedule_runnable(
            std::move(task).as_runnable_task(),
            m_queue.initial_team_scheduler_info(new_team_assoc));
      } else {
        // printf("waiting is aggregate: %d %d\n", task.node_id);
        // fflush(stdout);

        // The scheduling info update happens inside of schedule_aggregate
        m_queue.schedule_aggregate(
            std::move(task).template as_aggregate<task_scheduling_info_type>(),
            m_queue.initial_team_scheduler_info(new_team_assoc));
      }
    }
  };

  KOKKOS_INLINE_FUNCTION
  int team_ready_count(int team_association) {
    single_queue_type& queue = _get_queue_ref(team_association);
    return queue.ready_count();
  }

  KOKKOS_INLINE_FUNCTION
  bool team_queue_done(int team_association) {
    single_queue_type& queue = _get_queue_ref(team_association);
    return queue.is_done();
  }

  KOKKOS_INLINE_FUNCTION
  single_queue_type* _get_queue_ptr(int i) const {
    KOKKOS_ASSERT(m_queue_rep != nullptr);
    single_queue_type* rec = (single_queue_type*)&(((long**)m_queue_rep)[i][0]);
    return rec;
  }

  KOKKOS_INLINE_FUNCTION
  single_queue_type& _get_queue_ref(int i) const {
    single_queue_type* queue = _get_queue_ptr(i);
    KOKKOS_ASSERT(queue != nullptr);
    return *queue;
  }

  KOKKOS_INLINE_FUNCTION
  bool is_done() const noexcept {
    bool all_done = true;
    for (int i = 0; i < league_size; i++) {
      if (!_get_queue_ref(i).is_done()) {
        all_done = false;
        break;
      }
    }
    return all_done;
  }

  KOKKOS_INLINE_FUNCTION
  bool is_done(int i) const noexcept {
    single_queue_type& queue = _get_queue_ref(i);
    bool queue_done          = queue.is_done();
    // if (!queue_done) {
    // printf("queue is not done: %d, remaining = %d \n", i,
    // queue.ready_count());
    //}
    return queue_done;
  }

  // </editor-fold> end Constructors, destructors, and assignment }}}2
  //----------------------------------------------------------------------------
  KOKKOS_FUNCTION
  void schedule_runnable(runnable_task_base_type&& task,
                         team_scheduler_info_type const& info) {
    auto team_association = info.team_association;
    // Should only not be assigned if this is a host spawn...
    if (team_association == team_scheduler_info_type::NoAssociatedTeam) {
      team_association = 0;
    }
    // single_queue_type & queue = _get_queue_ref(team_association);
    // printf("distributed queue schedule runnable: [%d] %d, %d, %d \n",
    // task.node_id, team_association, queue.get_queue_id(),
    // mw_ptrtonodelet(&task)); fflush(stdout);
    this->do_schedule_runnable(_get_queue_ref(team_association),
                               std::move(task));

    // Task may be enqueued and may be run at any point; don't touch it (hence
    // the use of move semantics)
  }

  KOKKOS_FUNCTION
  OptionalRef<task_base_type> pop_ready_task(
      team_scheduler_info_type const& info) {
    KOKKOS_EXPECTS(info.team_association !=
                   team_scheduler_info_type::NoAssociatedTeam);

    auto return_value     = OptionalRef<task_base_type>{};
    auto team_association = info.team_association;

    // always loop in order of priority first, then prefer team tasks over
    // single tasks
    auto& team_queue = this->_get_queue_ref(team_association);

    return_value = team_queue.pop_ready_task(
        typename single_queue_type::team_scheduler_info_type{});

#ifdef ENABLE_TASK_STEALING
    if (not return_value) {
      // loop through the rest of the teams and try to steal
      for (auto isteal = (team_association + 1) % this->league_size;
           isteal != team_association;
           isteal = (isteal + 1) % this->league_size) {
        return_value =
            this->try_to_steal_ready_task(team_association, isteal, team_queue);
        if (return_value.get()) {
          break;
        }
      }

      // Note that this is where we'd update the task's scheduling info
    }
#endif
    // if nothing was found, return a default-constructed (empty) OptionalRef
    return return_value;
  }

  // TODO @tasking @generalization DSH make this a property-based customization
  // point
  KOKKOS_INLINE_FUNCTION
  team_scheduler_info_type initial_team_scheduler_info(int rank_in_league) const
      noexcept {
    return team_scheduler_info_type{
        typename team_scheduler_info_type::team_queue_id_t(rank_in_league %
                                                           league_size)};
  }

  // Call  TaskQueueCommon schedule_runnable with queue selected
  KOKKOS_INLINE_FUNCTION
  void do_schedule_runnable(single_queue_type& queue,
                            runnable_task_base_type&& task) {
    // First schedule the task
    queue.schedule_runnable(
        std::move(task),
        typename single_queue_type::team_scheduler_info_type{});
  }

  // interrogate the other teams' queue to see if there are any ready tasks
  KOKKOS_INLINE_FUNCTION
  OptionalRef<task_base_type> try_to_steal_ready_task(int iOrig, int iSteal,
                                                      single_queue_type& orig) {
    single_queue_type& stealing_queue = _get_queue_ref(iSteal);
    auto return_value                 = OptionalRef<task_base_type>{};
    return_value                      = stealing_queue.pop_ready_task(
        typename single_queue_type::team_scheduler_info_type{}, true);

    // We took it from one queue and added it to the other...need to update the
    // counts accordingly.
    if (return_value.get()) {
      // printf("Stealing task %d from %d for %d \n", return_value->node_id,
      // iSteal, iOrig); fflush(stdout);
      // stealing_queue.increment_ready_count();
      // orig.decrement_ready_count();
    }
    return return_value;
  }

  template <class task_queue_traits, class TeamSchedulerInfo>
  KOKKOS_FUNCTION void _dist_complete_finished_task(
      TaskNode<task_queue_traits>&& task, TeamSchedulerInfo const& info,
      single_queue_type& queue) {
    // printf("task consuming wait queue: %d\n", task.node_id);
    // fflush(stdout);
    task.consume_wait_queue(
        _dist_schedule_waiting_tasks_operation<
            task_queue_traits, task_queue_type, TeamSchedulerInfo>{task, *this,
                                                                   info});
    bool should_delete = task.decrement_and_check_reference_count();
    if (should_delete) {
      queue.deallocate(std::move(task));
    }
  }

  template <class task_queue_traits, class TeamSchedulerInfo>
  KOKKOS_FUNCTION void complete(RunnableTaskBase<task_queue_traits>&& task,
                                TeamSchedulerInfo const& info) {
    int team_assoc           = info.team_association;
    single_queue_type& queue = _get_queue_ref(team_assoc);

    if (task.get_respawn_flag()) {
      // printf("distibuted rescheduling respawned task: %d\n", task.node_id);
      queue.schedule_runnable(
          std::move(task),
          typename single_queue_type::team_scheduler_info_type{});
    } else {
      // printf("completing task: %d\n", task.node_id);
      _dist_complete_finished_task(std::move(task), info, queue);
    }
    // A runnable task was popped from a ready queue finished executing.
    // If respawned into a ready queue then the ready count was incremented
    // so decrement whether respawned or not.  If finished, all of the
    // tasks waiting on this have been enqueued (either in the ready queue
    // or the next waiting queue, in the case of an aggregate), and the
    // ready count has been incremented for each of those, preventing
    // quiescence.  Thus, it's safe to decrement the ready count here.
    // TODO @tasking @memory_order DSH memory order? (probably release)
    queue.decrement_ready_count();
    /*
       if (queue.ready_count() < 0) {
                   printf("dist task completion [%d] caused ready count to go
       negative: %d ... %d - %d \n", task.node_id, mw_ptrtonodelet(&queue),
       queue.get_queue_id(), queue.ready_count()); fflush(stdout); } else {
                   printf("dist task completion [%d] normally %d ... %d, %d \n",
       task.node_id, mw_ptrtonodelet(&queue), queue.get_queue_id(),
       queue.ready_count()); fflush(stdout);
           }
      */
  }

  template <class task_queue_traits, class SchedulingInfo,
            class TeamSchedulerInfo>
  KOKKOS_FUNCTION void complete(
      AggregateTask<task_queue_traits, SchedulingInfo>&& task,
      TeamSchedulerInfo const& info) {
    int team_assoc           = info.team_association;
    single_queue_type& queue = _get_queue_ref(team_assoc);
    _dist_complete_finished_task(std::move(task), info, queue);
  }
};

} /* namespace Impl */
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #if defined( KOKKOS_ENABLE_TASKDAG ) */
#endif /* #ifndef KOKKOS_IMPL_DistributedTaskQueue_HPP */
