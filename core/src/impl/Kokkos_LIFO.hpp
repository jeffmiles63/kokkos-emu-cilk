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

// Experimental unified task-data parallel manycore LDRD

#ifndef KOKKOS_IMPL_LIFO_HPP
#define KOKKOS_IMPL_LIFO_HPP

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_TASKDAG // Note: implies CUDA_VERSION >= 8000 if using CUDA

#include <Kokkos_Core_fwd.hpp>

#include <Kokkos_PointerOwnership.hpp>
#include <impl/Kokkos_OptionalRef.hpp>
#include <impl/Kokkos_Error.hpp> // KOKKOS_EXPECTS
#include <impl/Kokkos_LinkedListNode.hpp>

#include <Kokkos_Atomic.hpp>  // atomic_compare_exchange, atomic_fence

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

template <class T>
struct LockBasedLIFOCommon
{

  using value_type = T;

  using node_type = SimpleSinglyLinkedListNode<>;

  long queue_lock;  
  static constexpr uintptr_t LockTag = ~uintptr_t(0);
  static constexpr uintptr_t EndTag = ~uintptr_t(1);
  int curQueueCnt; 

  OwningRawPtr<node_type> m_head = (node_type*)EndTag;

  KOKKOS_INLINE_FUNCTION
  bool _try_push_node(node_type& node) {
    KOKKOS_EXPECTS(!node.is_enqueued());
    
    // lock the queue
    //printf("update queue lock II: %08x \n", &(this->queue_lock) );
    long test_lock = Kokkos::atomic_compare_exchange(&(this->queue_lock), (long)0, (long)1);

    if( test_lock == 0) {
       auto* volatile & next = LinkedListNodeAccess::next_ptr(node);		

       // set task->next to the head of the queue
       next = this->m_head;
       
       this->m_head = &node;
       
       this->curQueueCnt++;
       
       // unlock the queue
       ::Kokkos::atomic_exchange(&(this->queue_lock), 0);

       //printf("update head: %08x, queue cnt = %d \n", this->m_head, this->curQueueCnt);
       //fflush(stdout);    
     
       return true;  // successful update        
    }

    // Failed, replace 'task->m_next' value since 'task' remains
    // not a member of a queue.

    // TODO @tasking @memory_order DSH this should have a memory order and not a memory fence
    LinkedListNodeAccess::mark_as_not_enqueued(node);

    // fence to emulate acquire semantics on next
    // Do not proceed until 'next' has been stored.    

    return false;	
/*
    auto* volatile & next = LinkedListNodeAccess::next_ptr(node);

    // store the head of the queue in a local variable
    auto* old_head = m_head;

    // retry until someone locks the queue or we successfully compare exchange
    while (old_head != (node_type*)LockTag) {

      // TODO @tasking @memory_order DSH this should have a memory order and not a memory fence

      // set task->next to the head of the queue
      next = old_head;

      // fence to emulate acquire semantics on next and release semantics on
      // the store of m_head
      // Do not proceed until 'next' has been stored.
      Kokkos::memory_fence();

      // store the old head
      auto* const old_head_tmp = old_head;

      // attempt to swap task with the old head of the queue
      // as if this were done atomically:
      //   if(m_head == old_head) {
      //     m_head = &node;
      //   }
      //   old_head = m_head;
      //printf("update queue head: %08x \n", m_head);
      old_head = ::Kokkos::atomic_compare_exchange(&m_head, old_head, &node);

      if(old_head_tmp == old_head) return true;
    }

    // Failed, replace 'task->m_next' value since 'task' remains
    // not a member of a queue.

    // TODO @tasking @memory_order DSH this should have a memory order and not a memory fence
    LinkedListNodeAccess::mark_as_not_enqueued(node);

    // fence to emulate acquire semantics on next
    // Do not proceed until 'next' has been stored.
    ::Kokkos::memory_fence();

    return false;	  
*/

  }

  bool _is_empty() const noexcept {
    // TODO @tasking @memory_order DSH make this an atomic load with memory order
    return this->m_head == (node_type*)EndTag;
  }

};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

template <class T>
class LockBasedLIFO
  : private LockBasedLIFOCommon<T>
{

private:

  using base_t = LockBasedLIFOCommon<T>;
  using node_type = typename base_t::node_type;

public:

  using value_type = typename base_t::value_type; // = T
  using intrusive_node_base_type = SimpleSinglyLinkedListNode<>;

public:


  LockBasedLIFO() { this->queue_lock = 0; this->curQueueCnt = 0; }
  LockBasedLIFO(LockBasedLIFO const&) = delete;
  LockBasedLIFO(LockBasedLIFO&&) = delete;
  LockBasedLIFO& operator=(LockBasedLIFO const&) = delete;
  LockBasedLIFO& operator=(LockBasedLIFO&&) = delete;

  ~LockBasedLIFO() = default;


  bool empty() const noexcept {
    // TODO @tasking @memory_order DSH memory order
    return this->_is_empty();
  }

  KOKKOS_INLINE_FUNCTION
  OptionalRef<T> pop(bool abort_on_locked = false)
  {
    //printf("pop(): %08x \n", this->m_head);
    //fflush(stdout);
    // Put this in here to avoid requiring value_type to be complete until now.
    static_assert(
      std::is_base_of<intrusive_node_base_type, value_type>::value,
      "Intrusive linked-list value_type must be derived from intrusive_node_base_type"
    );
    
    // just check one time for end tag...pop is allowed to return nothing...
    if ( this->m_head == (node_type*)base_t::EndTag ) {
	   return { };	
	}

    // Retry until the lock is acquired or the queue is empty.
    while(true) {

       // lock the queue
       //printf("update queue lock: %08x \n", &(this->queue_lock) );
       long test_lock = Kokkos::atomic_compare_exchange(&(this->queue_lock), (long)0, (long)1);

       if( test_lock == 0) {
		   
		   auto* rv = this->m_head;
		   
		   if (rv == (node_type*)base_t::EndTag ) {
			   Kokkos::atomic_exchange(&(this->queue_lock), 0);
			   break;
		   }

           // TODO @tasking @memory_order DSH check whether the volatile is needed here
           OwningRawPtr<node_type> next = LinkedListNodeAccess::next_ptr(*rv);

           // This algorithm is not lockfree because a adversarial scheduler could
           // context switch this thread at this point and the rest of the threads
           // calling this method would never make forward progress

           this->m_head = next;
        
          // Mark rv as popped by assigning nullptr to the next
          LinkedListNodeAccess::mark_as_not_enqueued(*rv);

          this->curQueueCnt--;
          
          Kokkos::atomic_exchange(&(this->queue_lock), 0); // unlock the queue
          
          //printf("pop, remaining queue count = %d \n", this->curQueueCnt);
          //fflush(stdout);
        
          //Kokkos::Experimental::print_pointer( 0, this, "queue this ptr");
          //Kokkos::Experimental::print_pointer( 0, rv, "queue task ptr");
          return OptionalRef<T>{ *static_cast<T*>(rv) };
       } else {       
           if(abort_on_locked) {		     
             break;
           }
        }
    }
    
    // Return an empty OptionalRef by calling the default constructor
    return { };
  }

  KOKKOS_INLINE_FUNCTION
  OptionalRef<T>
  steal()
  {
    // TODO @tasking @optimization DSH do this with fewer retries
    return pop(/* abort_on_locked = */ true);
  }
  
  KOKKOS_INLINE_FUNCTION
  bool try_push(node_type& node) {
    KOKKOS_EXPECTS(!node.is_enqueued());
    
    return this->_try_push_node(node); 
  
  }

  KOKKOS_INLINE_FUNCTION
  bool push(node_type& node)
  {
    while(!this->try_push(node)) { /* retry until success */ }
    // for consistency with push interface on other queue types:
    return true;
  }

  KOKKOS_INLINE_FUNCTION
  bool push(node_type&& node)
  {
    // Just forward to the lvalue version
    return push(node);
  }

};


/** @brief A Multiple Producer, Single Consumer Queue with some special semantics
 *
 * This multi-producer, single consumer queue has the following semantics:
 *
 *   - Any number of threads may call `try_emplace`/`try_push`
 *       + These operations are lock-free.
 *   - Exactly one thread calls `consume()`, and the call occurs exactly once
 *     in the lifetime of the queue.
 *       + This operation is lock-free (and wait-free w.r.t. producers)
 *   - Any calls to `try_push` that happen-before the call to
 *     `consume()` will succeed and return an true, such that the `consume()`
 *     call will visit that node.
 *   - Any calls to `try_push` for which the single call to `consume()`
 *     happens-before those calls will return false and the node given as
 *     an argument to `try_push` will not be visited by consume()
 *
 *
 * @tparam T The type of items in the queue
 *
 */
template <class T>
class SingleConsumeOperationLIFO
  : private LockBasedLIFOCommon<T>
{
private:

  using base_t = LockBasedLIFOCommon<T>;
  using node_type = typename base_t::node_type;

  // Allows us to reuse the existing infrastructure for
  static constexpr auto ConsumedTag = base_t::LockTag;

public:

  using value_type = typename base_t::value_type; // = T

  KOKKOS_INLINE_FUNCTION
  SingleConsumeOperationLIFO() noexcept = default;

  SingleConsumeOperationLIFO(SingleConsumeOperationLIFO const&) = delete;
  SingleConsumeOperationLIFO(SingleConsumeOperationLIFO&&) = delete;
  SingleConsumeOperationLIFO& operator=(SingleConsumeOperationLIFO const&) = delete;
  SingleConsumeOperationLIFO& operator=(SingleConsumeOperationLIFO&&) = delete;

  KOKKOS_INLINE_FUNCTION
  ~SingleConsumeOperationLIFO() = default;

  KOKKOS_INLINE_FUNCTION
  bool empty() const noexcept {
    // TODO @tasking @memory_order DSH memory order
    return this->_is_empty();
  }

  KOKKOS_INLINE_FUNCTION
  bool is_consumed() const noexcept {
    // TODO @tasking @memory_order DSH memory order?
    return this->m_head == (node_type*)ConsumedTag;
  }

  KOKKOS_INLINE_FUNCTION
  bool try_push(node_type& node)
  {
	  if (this->m_head == (node_type*)ConsumedTag)
	     return false;
	
	  //printf("try push onto single consume queue: 0x%lx - %d, 0x%lx - %d \n", 
	  //   &node, mw_ptrtonodelet(&node), &(this->m_head), mw_ptrtonodelet(&(this->m_head)) );
      return this->_try_push_node(node);
      // Ensures: (return value is true) || (node.is_enqueued() == false);
  }
  
  KOKKOS_INLINE_FUNCTION
  void push(node_type& node) {
	  while (try_push(node) == false) {
		  RESCHEDULE();
	  }
  }

  template <class Function>
  KOKKOS_INLINE_FUNCTION
  void consume(Function&& f) {
    auto* const consumed_tag = (node_type*)ConsumedTag;

    // lock the queue
    //printf("update queue lock II: %08x \n", &(this->queue_lock) );
    long test_lock = Kokkos::atomic_compare_exchange(&(this->queue_lock), (long)0, (long)1);

    if( test_lock == 0) {

       // Swap the Consumed tag into the head of the queue:

       // (local variable used for assertion only)
       // TODO @tasking @memory_order DSH this should have memory order release, I think
       auto old_head = Kokkos::atomic_exchange(&(this->m_head), consumed_tag);

       // Assert that the queue wasn't consumed before this
       // This can't be an expects clause because the acquire fence on the read
       // would be a side-effect
       KOKKOS_ASSERT(old_head != consumed_tag);

       // We now have exclusive access to the queue; loop over it and call
       // the user function
       while(old_head != (node_type*)base_t::EndTag) {

          // get the Node to make the call with
          auto* call_arg = old_head;

          // advance the head
          old_head = LinkedListNodeAccess::next_ptr(*old_head);

          // Mark as popped before proceeding
          LinkedListNodeAccess::mark_as_not_enqueued(*call_arg);
      
          printf("queue consume call user function: 0x%lx \n", call_arg);
          fflush(stdout);

          // Call the user function
          auto& arg = *static_cast<T*>(call_arg);
          f(std::move(arg));

       }
              // unlock the queue
       ::Kokkos::atomic_exchange(&(this->queue_lock), 0);
     }
  }

};

} // end namespace Impl
} // end namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

struct TaskQueueTraitsLockBased
{

  // TODO @tasking @documentation DSH document what concepts these match

  template <class Task>
  using ready_queue_type = LockBasedLIFO<Task>;

  template <class Task>
  using waiting_queue_type = SingleConsumeOperationLIFO<Task>;

  template <class Task>
  using intrusive_task_base_type =
    typename ready_queue_type<Task>::intrusive_node_base_type;

  static constexpr auto ready_queue_insertion_may_fail = false;

};


} // end namespace Impl
} // end namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* defined KOKKOS_ENABLE_TASKDAG */
#endif /* #ifndef KOKKOS_IMPL_LIFO_HPP */

