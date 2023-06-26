
#ifndef PARLAY_PARALLEL_H_
#define PARLAY_PARALLEL_H_

#include <cassert>
#include <cstdlib>

#include <algorithm>
#include <string>
#include <thread>
#include <type_traits>  // IWYU pragma: keep
#include <utility>

// ----------------------------------------------------------------------------
// All scheduler plugins are required to implement the
// following four functions to integrate with parlay:
//  - num_workers :   return the max number of workers
//  - worker_id :     return the id of the current worker
//  - parallel_for :  execute a given loop body in parallel
//  - par_do :        execute two given functions in parallel
// ----------------------------------------------------------------------------

namespace parlay {

// number of threads available from OS
inline size_t num_workers();

// id of running thread, should be numbered from [0...num-workers)
inline size_t worker_id();

// parallel loop from start (inclusive) to end (exclusive) running
// function f.
//    f should map size_t to void.
//    granularity is the number of iterations to run sequentially
//      if 0 (default) then the scheduler will decide
//    conservative uses a safer scheduler
template <typename F>
inline void parallel_for(size_t start, size_t end, F&& f, long granularity = 0,
                         bool conservative = false);

// runs the thunks left and right in parallel.
//    both left and write should map void to void
//    conservative uses a safer scheduler
template <typename Lf, typename Rf>
inline void par_do(Lf&& left, Rf&& right, bool conservative = false);

// ----------------------------------------------------------------------------
//          Extra functions implemented on top of the four basic ones
//
//  - parallel_do : alias of par_do
//  - blocked_for : parallelize a loop over a sequence of blocks
// ----------------------------------------------------------------------------

// Given two functions left and right, both invocable with no arguments,
// invokes both of them potentially in parallel.
template <typename Lf, typename Rf>
inline void parallel_do(Lf&& left, Rf&& right, bool conservative = false) {
  static_assert(std::is_invocable_v<Lf&&>);
  static_assert(std::is_invocable_v<Rf&&>);
  par_do(std::forward<Lf>(left), std::forward<Rf>(right), conservative);
}

template<typename F>
inline void blocked_for(size_t start, size_t end, size_t block_size, F&& f, bool conservative = false) {
  static_assert(std::is_invocable_v<F&, size_t, size_t, size_t>);
  assert(block_size > 0);
  if (start < end) {
    size_t n_blocks = (end - start + block_size - 1) / block_size;
    parallel_for(0, n_blocks, [&](size_t i) {
      size_t s = start + i * block_size;
      size_t e = (std::min)(s + block_size, end);
      f(i, s, e);
    }, 0, conservative);
  }
}

}  // namespace parlay

//****************************************************
//                Scheduler selection
//****************************************************

// Cilk Scheduler
#if defined(PARLAY_CILKPLUS)
#include "internal/scheduler_plugins/cilkplus.h"    // IWYU pragma: keep, export

#elif defined(PARLAY_OPENCILK)
#include "internal/scheduler_plugins/opencilk.h"    // IWYU pragma: keep, export

// OpenMP Scheduler
#elif defined(PARLAY_OPENMP)
#include "internal/scheduler_plugins/omp.h"         // IWYU pragma: keep, export

// Intel's Thread Building Blocks Scheduler
#elif defined(PARLAY_TBB)
#include "internal/scheduler_plugins/tbb.h"         // IWYU pragma: keep, export

// No Parallelism
#elif defined(PARLAY_SEQUENTIAL)
#include "internal/scheduler_plugins/sequential.h"  // IWYU pragma: keep, export

// Parlay's Homegrown Scheduler
#else

#include "scheduler.h"

namespace parlay {

inline size_t num_workers() {
  assert(IsSchedulerInitialized());
  return GetScheduler()->num_workers();
}

inline size_t worker_id() {
  assert(IsSchedulerInitialized());
  return GetScheduler()->worker_id();
}

template <class F>
inline void parallel_for(size_t start, size_t end, F&& f, long granularity,
                         bool conservative) {
  static_assert(std::is_invocable_v<F&, size_t>);
  // Note: scheduler::parfor copies the function object, so we wrap it in
  // a lambda here in case F is expensive to copy or not copyable at all
  auto loop_body = [&](size_t i) { f(i); };
  if (start + 1 == end) {
    f(start);
  }
  else if ((end - start) <= static_cast<size_t>(granularity)) {
    for (size_t i = start; i < end; i++) loop_body(i);
  }
  else if (end > start) {
    GetScheduler()->parfor(start, end, loop_body, static_cast<size_t>(granularity), conservative);
  }
}

template <typename Lf, typename Rf>
inline void par_do(Lf&& left, Rf&& right, bool conservative) {
  static_assert(std::is_invocable_v<Lf&&>);
  static_assert(std::is_invocable_v<Rf&&>);
  return GetScheduler()->pardo(std::forward<Lf>(left), std::forward<Rf>(right), conservative);
}

}  // namespace parlay

#endif

#endif  // PARLAY_PARALELL_H_
