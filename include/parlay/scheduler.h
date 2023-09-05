
#ifndef PARLAY_SCHEDULER_H_
#define PARLAY_SCHEDULER_H_

#include <cassert>
#include <cstdint>
#include <cstdlib>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>         // IWYU pragma: keep
#include <iostream>
#include <memory>
#include <thread>
#include <type_traits>    // IWYU pragma: keep
#include <utility>
#include <vector>

#include "internal/work_stealing_deque.h"
#include "internal/work_stealing_job.h"

// IWYU pragma: no_include <bits/chrono.h>
// IWYU pragma: no_include <bits/this_thread_sleep.h>



// True if the scheduler should scale the number of awake workers
// proportional to the amount of work to be done. This saves CPU
// time if there is not any parallel work available, but may cause
// some startup lag when more parallelism becomes available.
//
// Default: true
#ifndef PARLAY_ELASTIC_PARALLELISM
#define PARLAY_ELASTIC_PARALLELISM false
#endif


// PARLAY_ELASTIC_STEAL_TIMEOUT sets the number of microseconds
// that a worker will attempt to steal jobs, such that if no
// jobs are successfully stolen, it will go to sleep.
//
// Default: 10000 (10 milliseconds)
#ifndef PARLAY_ELASTIC_STEAL_TIMEOUT
#define PARLAY_ELASTIC_STEAL_TIMEOUT 10000
#endif


#if PARLAY_ELASTIC_PARALLELISM
#include "internal/atomic_wait.h"
#endif

namespace parlay {

// kMaxThreadId represents an invalid thread id. A thread with thread_id equals
// kMaxThreadId is an uninitialized thread from Parlay's perspective.
inline constexpr unsigned int kMaxThreadId =
    std::numeric_limits<unsigned int>::max();

inline unsigned int& GetThreadId() {
  static thread_local unsigned int thread_id = kMaxThreadId;
  return thread_id;
}

template <typename Job>
struct scheduler {
  static_assert(std::is_invocable_r_v<void, Job&>);

  // After YIELD_FACTOR * P unsuccessful steal attempts, a
  // a worker will sleep briefly for SLEEP_FACTOR * P nanoseconds
  // to give other threads a chance to work and save some cycles.
  constexpr static size_t YIELD_FACTOR = 200;
  constexpr static size_t SLEEP_FACTOR = 200;

  // The length of time that a worker must fail to steal anything
  // before it goes to sleep to save CPU time.
  constexpr static std::chrono::microseconds STEAL_TIMEOUT{PARLAY_ELASTIC_STEAL_TIMEOUT};

 public:
  unsigned int num_threads;

  explicit scheduler(size_t num_workers)
      : num_threads(num_workers),
        num_deques(num_threads),
        num_awake_workers(num_threads),
        deques(num_deques),
        attempts(num_deques),
        spawned_threads(),
        finished_flag(false) {

    // Spawn num_threads many threads on startup
    GetThreadId() = 0;  // thread-local write
    for (unsigned int i = 1; i < num_threads; i++) {
      spawned_threads.emplace_back([&, i]() {
        worker(i);
      });
    }
  }

  ~scheduler() {
    shutdown();
  }

  // Push onto local stack.
  void spawn(Job* job) {
    int id = worker_id();
    [[maybe_unused]] bool first = deques[id].push_bottom(job);
#if PARLAY_ELASTIC_PARALLELISM
    if (first) wake_up_a_worker();
#endif
  }

  // Wait until the given condition is true.
  //
  // If conservative, this thread will simply busy wait. Otherwise,
  // it will look for work to steal and keep itself occupied. This
  // can deadlock if the stolen work wants a lock held by the code
  // that is waiting, so avoid that.
  template <typename F>
  void wait_until(F&& done, bool conservative = false) {
    // Conservative avoids deadlock if scheduler is used in conjunction
    // with user locks enclosing a wait.
    if (conservative) {
      while (!done())
        std::this_thread::yield();
    }
    // If not conservative, schedule within the wait.
    // Can deadlock if a stolen job uses same lock as encloses the wait.
    else {
      do_work_until(std::forward<F>(done));
    }
  }

  // Pop from local stack.
  Job* get_own_job() {
    auto id = worker_id();
    return deques[id].pop_bottom();
  }

  unsigned int num_workers() { return num_threads; }
  unsigned int worker_id() { return GetThreadId(); }

  // Sets the amount of sleep delay when a worker finds no work to steal. This
  // controls a tradeoff between latency (how long before a worker wakes up and
  // discovers a pending job to steal) and cpu load when idle (how often a
  // worker tries to find a job to steal). Returns the previous value.
  // NOTE: It is normally not necessary to adjust this, since the initial value
  // is chosen to incur minimal load with only a small latency impact. But this
  // function is available for tuning performance in special cases.
  std::chrono::nanoseconds set_sleep_delay_when_idle(
      std::chrono::nanoseconds new_sleep_delay_when_idle) {
    auto old_value = sleep_delay_when_idle.load();
    sleep_delay_when_idle.store(new_sleep_delay_when_idle);
    return old_value;
  }

  bool finished() const noexcept {
    return finished_flag.load(std::memory_order_acquire);
  }

 private:
  // Align to avoid false sharing.
  struct alignas(128) attempt {
    size_t val;
  };

  int num_deques;
  std::atomic<size_t> num_awake_workers;
  std::vector<internal::Deque<Job>> deques;
  std::vector<attempt> attempts;
  std::vector<std::thread> spawned_threads;
  std::atomic<int> finished_flag;

  std::atomic<size_t> wake_up_counter{0};
  std::atomic<size_t> num_finished_workers{0};

  // When a worker fails to find work to steal, it performs 100 * num_queues
  // sleeps of this duration (checking finished() and work_available after each
  // one) before trying again to find work to steal.
  std::atomic<std::chrono::nanoseconds> sleep_delay_when_idle =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::milliseconds(1));


  // Start an individual worker task, stealing work if no local
  // work is available. May go to sleep if no work is available
  // for a long time, until woken up again when notified that
  // new work is available.
  void worker(size_t id) {
    GetThreadId() = id;  // thread-local write
#if PARLAY_ELASTIC_PARALLELISM
    wait_for_work();
#endif
    while (!finished()) {
      Job* job = get_job([&]() { return finished(); }, PARLAY_ELASTIC_PARALLELISM);
      if (job)(*job)();
#if PARLAY_ELASTIC_PARALLELISM
      else if (!finished()) {
        // If no job was stolen, the worker should go to
        // sleep and wait until more work is available
        wait_for_work();
      }
#endif
    }
    assert(finished());
    num_finished_workers.fetch_add(1);
  }

  // Runs tasks until done(), stealing work if necessary.
  //
  // Does not sleep or time out since this can be called
  // by the main thread and by join points, for which sleeping
  // would cause deadlock, and timing out could cause a join
  // point to resume execution before the job it was waiting
  // on has completed.
  template <typename F>
  void do_work_until(F&& done) {
    while (true) {
      Job* job = get_job(done, false);  // timeout MUST BE false
      if (!job) return;
      (*job)();
    }
    assert(done());
  }

  // Find a job, first trying local stack, then random steals.
  //
  // Returns nullptr if break_early() returns true before a job
  // is found, or, if timeout is true and it takes longer than
  // STEAL_TIMEOUT to find a job to steal.
  template <typename F>
  Job* get_job(F&& break_early, bool timeout) {
    if (break_early()) return nullptr;
    Job* job = get_own_job();
    if (job) return job;
    else job = steal_job(std::forward<F>(break_early), timeout);
    return job;
  }
  
  // Find a job with random steals.
  //
  // Returns nullptr if break_early() returns true before a job
  // is found, or, if timeout is true and it takes longer than
  // STEAL_TIMEOUT to find a job to steal.
  template<typename F>
  Job* steal_job(F&& break_early, bool timeout) {
    size_t id = worker_id();
    const auto start_time = std::chrono::steady_clock::now();
    do {
      // By coupon collector's problem, this should touch all.
      for (size_t i = 0; i <= YIELD_FACTOR * num_deques; i++) {
        if (break_early()) return nullptr;
        Job* job = try_steal(id);
        if (job) return job;
      }
      std::this_thread::sleep_for(std::chrono::nanoseconds(num_deques * 100));
    } while (!timeout || std::chrono::steady_clock::now() - start_time < STEAL_TIMEOUT);
    return nullptr;
  }

  Job* try_steal(size_t id) {
    // use hashing to get "random" target
    size_t target = (hash(id) + hash(attempts[id].val)) % num_deques;
    attempts[id].val++;
    auto [job, empty] = deques[target].pop_top();
#if PARLAY_ELASTIC_PARALLELISM
    if (!empty) wake_up_a_worker();
#endif
    return job;
  }

#if PARLAY_ELASTIC_PARALLELISM

  // Wakes up at least one sleeping worker (more than one
  // worker may be woken up depending on the implementation).
  void wake_up_a_worker() {
    if (num_awake_workers.load(std::memory_order_acquire) < num_threads) {
      wake_up_counter.fetch_add(1);
      parlay::atomic_notify_one(&wake_up_counter);
    }
  }
  
  // Wake up all sleeping workers
  void wake_up_all_workers() {
    if (num_awake_workers.load(std::memory_order_acquire) < num_threads) {
      wake_up_counter.fetch_add(1);
      parlay::atomic_notify_all(&wake_up_counter);
    }
  }
  
  // Wait until notified to wake up
  void wait_for_work() {
    num_awake_workers.fetch_sub(1);
    parlay::atomic_wait(&wake_up_counter, wake_up_counter.load());
    num_awake_workers.fetch_add(1);
  }

#endif

  size_t hash(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return static_cast<size_t>(x);
  }
  
  void shutdown() {
    finished_flag.store(true, std::memory_order_release);
#if PARLAY_ELASTIC_PARALLELISM
    // We must spam wake all workers until they finish in
    // case any of them are just about to fall asleep, since
    // they might therefore miss the flag to finish
    while (num_finished_workers.load() < num_threads - 1) {
      wake_up_all_workers();
      std::this_thread::yield();
    }
#endif
    for (unsigned int i = 1; i < num_threads; i++) {
      spawned_threads[i - 1].join();
    }
    // Reset thread_local thread_id for the main thread before recycling it.
    GetThreadId() = kMaxThreadId;
  }
};

class fork_join_scheduler;

// scheduler_internal_pointer is a data class that defines pointer types used
// internally for scheduler lifecycle management. It is conceptually equivalent
// to `scheduler_pointer` except that the latter is used externally (as a return
// type of `initialize_scheduler`).
//
// The intended use of `scheduler_internal_pointer` is to maintain a
// thread-local instance of this class for all Parlay threads (both main and
// child threads).
struct scheduler_internal_pointer {
  fork_join_scheduler* raw;
  std::weak_ptr<fork_join_scheduler> weak;

  scheduler_internal_pointer() : raw(nullptr) {}
};

// scheduler_pointer is the external interface as the pointer type for the
// scheduler. It supports dereference operation.
struct scheduler_pointer {
  fork_join_scheduler* raw;
  std::shared_ptr<fork_join_scheduler> shared;

  scheduler_pointer() : raw(nullptr) {}

  scheduler_pointer(const scheduler_pointer& other) {
    raw = other.raw;
    shared = other.shared;
  }

  scheduler_pointer& operator=(const scheduler_pointer& other) {
    raw = other.raw;
    shared = other.shared;
    return *this;
  }

  // When dereferencing, the raw pointer must be valid. Do *not* use `shared`,
  // which is for lifecycle management only.
  fork_join_scheduler* operator->() { return raw; }
};

class fork_join_scheduler {
  using Job = WorkStealingJob;

  // Underlying scheduler object
  std::unique_ptr<scheduler<Job>> sched;

  inline static thread_local unsigned int num_workers_to_start = 0;

 public:
  explicit fork_join_scheduler(size_t p) : sched(std::make_unique<scheduler<Job>>(p)) {}

  static void set_num_workers_to_start(unsigned int num_workers) {
    num_workers_to_start = num_workers;
  }

  // Returns a scheduler_pointer instance managing the lifecycle of a underlying
  // Parlay scheduler.
  //
  // Properties of the returned scheduler_pointer instance:
  //
  // (1) `raw` points to a valid scheduler instance.
  //
  // (2) `shared` is nullptr except for the following two cases, in which it
  // points to the same scheduler instance as the raw pointer.
  //
  // (2.1) If the current call creates a scheduler instance.
  //
  // (2.2) If the current call has `increment_ref_count` set to true and is made
  // by the thread which is the root thread of a live scheduler instance. This
  // represents a case where the caller explicitly maintains the shared
  // ownership of the scheduler.
  static scheduler_pointer GetThreadLocalScheduler(
      bool increment_ref_count) {
    auto& scheduler = GetInternalScheduler();
    scheduler_pointer result;
    unsigned int thread_id = GetThreadId();
    if (thread_id == kMaxThreadId) {
      // We have not yet been initialized, so are not part of an existing parlay
      // scheduler. Create a scheduler and perform ref-counting.

      std::cerr << "Currently not expecting to create scheduler instance from GetThreadLocalScheduler()" << std::endl;
      exit(-1);

//      // Create the scheduler instance.
//      result.shared = std::make_shared<fork_join_scheduler>();
//
//      // Assign it to the thread_local weak pointer. The scheduler is stored in
//      // the thread_local weak pointer variable for the root thread only.
//      scheduler.weak = result.shared;
//
//      // Also store the raw pointer. This allows faster access when ref-counting
//      // is unnecessary based on the call context.
//      fork_join_scheduler* raw_pointer = result.shared.get();
//      scheduler.raw = raw_pointer;
//
//      // The raw pointer needs to be stored in child threads' thread_local
//      // variable as well. Thus the need to capture it in the initialization
//      // function.
//      std::function<void(unsigned int)> init_func =
//          [raw_pointer](unsigned int child_thread_id) {
//            GetThreadId() = child_thread_id;
//            SetInternalScheduler(raw_pointer);
//          };
//
//      std::function<void()> cleanup_func = []() {
//        ResetInternalScheduler();
//        GetThreadId() = kMaxThreadId;
//      };
//      raw_pointer->start_workers(std::move(init_func), std::move(cleanup_func));

    } else if (thread_id == 0 && increment_ref_count) {
      // We are in the already-initialized main thread and we are requested to
      // increase the ref-count. This is the case where the external caller
      // would like to explicitly manage the lifecycle of the scheduler.
      result.shared = scheduler.weak.lock();
    }

    // When we arrive here, the raw pointer in the thread_local variable
    // `scheduler` must have already been set properly.
    result.raw = scheduler.raw;

    // Invariants on return value
    // (1) Raw pointer must not be nullptr
    // (2) If (2.1) we create a new scheduler instance in this call or (2.2)
    // `increment_ref_count` is set to true, then the returned shared pointer
    // must not be nullptr and it must point to the same scheduler object as the
    // raw pointer. Otherwise, it must be nullptr.
    assert(result.raw != nullptr);

    // Note that we must use the previously stored thread_id instead of calling
    // GetThreadId again, because the thread-local value in GetThreadId may have
    // been changed within this function. The invariant is checked against the
    // initial thread_id value upon entry.
    if ((thread_id == kMaxThreadId) ||
        (thread_id == 0 && increment_ref_count)) {
      assert(result.shared != nullptr);
      assert(result.shared.get() == result.raw);
    } else {
      assert(result.shared == nullptr);
    }

    return result;
  }

  unsigned int num_workers() { return sched->num_workers(); }
  unsigned int worker_id() { return sched->worker_id(); }
  std::chrono::nanoseconds set_sleep_delay_when_idle(
      std::chrono::nanoseconds new_sleep_delay_when_idle) {
    return sched->set_sleep_delay_when_idle(new_sleep_delay_when_idle);
  }

  // Determine the number of workers to spawn
  unsigned int init_num_workers() {
    if (const auto env_p = std::getenv("PARLAY_NUM_THREADS")) {
      return std::stoi(env_p);
    } else {
      if (num_workers_to_start == 0) {
        std::cerr
            << "WARNING: Initializing parlay scheduler from "
               "parlay/scheduler.h. Expected scheduler to be explicitly "
               "initialized by the client before use. In the future, we may "
               "change the default initialization behavior to result in a "
               "runtime error."
            << std::endl;
        num_workers_to_start = std::thread::hardware_concurrency();
      }
      return num_workers_to_start;
    }
  }

  // Fork two thunks and wait until they both finish.
  template <typename L, typename R>
  void pardo(L&& left, R&& right, bool conservative = false) {
    auto execute_right = [&]() { std::forward<R>(right)(); };
    auto right_job = make_job(right);
    sched->spawn(&right_job);
    std::forward<L>(left)();
    if (const Job* job = sched->get_own_job(); job != nullptr) {
      assert(job == &right_job);
      execute_right();
    }
    else {
      //sched->wait_for(right_job, conservative);
      auto done = [&]() { return right_job.finished(); };
      sched->wait_until(done, conservative);
      assert(right_job.finished());
    }
  }

  template <typename F>
  void parfor(size_t start, size_t end, F f, size_t granularity = 0, bool conservative = false) {
    if (end <= start) return;
    if (granularity == 0) {
      size_t done = get_granularity(start, end, f);
      granularity = std::max(done, (end - start) / static_cast<size_t>(128 * sched->num_threads));
      start += done;
    }
    parfor_(start, end, f, granularity, conservative);
  }

 private:
  template <typename F>
  size_t get_granularity(size_t start, size_t end, F f) {
    size_t done = 0;
    size_t sz = 1;
    unsigned long long int ticks = 0;
    do {
      sz = std::min(sz, end - (start + done));
      auto tstart = std::chrono::steady_clock::now();
      for (size_t i = 0; i < sz; i++) f(start + done + i);
      auto tstop = std::chrono::steady_clock::now();
      ticks = static_cast<unsigned long long int>(std::chrono::duration_cast<
                std::chrono::nanoseconds>(tstop - tstart).count());
      done += sz;
      sz *= 2;
    } while (ticks < 1000 && done < (end - start));
    return done;
  }

  template <typename F>
  void parfor_(size_t start, size_t end, F f, size_t granularity, bool conservative) {
    if ((end - start) <= granularity)
      for (size_t i = start; i < end; i++) f(i);
    else {
      size_t n = end - start;
      // Not in middle to avoid clashes on set-associative caches on powers of 2.
      size_t mid = (start + (9 * (n + 1)) / 16);
      pardo([&]() { parfor_(start, mid, f, granularity, conservative); },
            [&]() { parfor_(mid, end, f, granularity, conservative); },
            conservative);
    }
  }

  static scheduler_internal_pointer& GetInternalScheduler() {
    static thread_local scheduler_internal_pointer scheduler_ptr;
    return scheduler_ptr;
  }

  static void SetInternalScheduler(fork_join_scheduler* input_scheduler) {
    auto& scheduler = GetInternalScheduler();
    scheduler.raw = input_scheduler;  // thread-local write
    // There is no need to populate the weak pointer for child threads.
    scheduler.weak.reset();
  }

  static void ResetInternalScheduler() {
    auto& scheduler = GetInternalScheduler();
    scheduler.raw = nullptr;
    // This is only needed by the main thread. But resetting is harmless for
    // child threads.
    scheduler.weak.reset();
  }

};


inline bool IsSchedulerInitialized() {
  return GetThreadId() != kMaxThreadId;
}

// Returns the thread-local scheduler pointer. Creates the thread-local
// scheduler if necessary.
//
// If `increment_ref_count` is true, then we return the shared pointer in
// addition to the raw pointer if we are called in thread_id==0 (i.e., we are
// in an already-initialized Parlay main thread).
//
// If we are called with thread_id==kMaxThreadId, then a new scheduler instance
// will be created with this call. In this case, we always return the shared
// pointer along with the raw pointer, regardless of the value of
// `increment_ref_count`.
inline scheduler_pointer GetScheduler(bool increment_ref_count = false) {
  return fork_join_scheduler::GetThreadLocalScheduler(increment_ref_count);
}


// Initializes the scheduler to use num_workers, and starts the workers. Calling
// this function is optional, but recommended: if it is not called, the
// scheduler will be initialized with a default num_workers the first time it is
// used via a parallel_for() or pardo(). Since the default num_workers is
// unlikely to be the best choice for the algorithm and hardware environment,
// calling initialize_scheduler is preferred.
//
// `num_workers` will have no effect if the scheduler is already running, either
// from a previous call to initialize_scheduler, or from a call to
// GetScheduler() (via parallel_for() or pardo()). But initialize_scheduler will
// always extend the lifetime of the scheduler to include that of the returned
// scheduler_pointer if it is called wtihin a root thread (i.e., thread id ==
// 0).
//
// Returns a scheduler_pointer instance pointing to the new scheduler instance.
inline scheduler_pointer initialize_scheduler(
    unsigned int num_workers) {
  fork_join_scheduler::set_num_workers_to_start(num_workers);
  return GetScheduler(/*increment_ref_count=*/true);
}



}  // namespace parlay

#endif  // PARLAY_SCHEDULER_H_
