// Copyright 2016 The Gemmlowp Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// multi_thread_gemm.h: Multi-threaded GEMM entry point.
// Readers note: To understand this file, it is useful to first
// read and understand the much simpler single_thread_gemm.h.

#ifndef GEMMLOWP_INTERNAL_PTHREAD_SHIM_H_
#define GEMMLOWP_INTENRAL_PTHREAD_SHIM_H_

#include <condition_variable>
#include <mutex>
#include <thread>

namespace gemmlowp {

typedef void gemmlowp_thread_attr_t;
typedef std::condition_variable gemmlowp_cond_t;
typedef std::mutex gemmlowp_mutex_t;
typedef std::thread gemmlowp_thread_t;

inline int pthread_create(pthread_t* thread, const pthread_attr_t* attr,
                          void *(*start_routine) (void*), void* arg) {
  *thread = std::thread(start_routine, arg);
  return 0;
}

inline int pthread_join(pthread_t& thread) {
  thread.join();
  return 0;
}

inline int pthread_cond_signal(pthread_cond_t* cond) {
  cond->notify_one();
  return 0;
}

inline int pthread_cond_wait(pthread_cond_t* cond, pthread_mutex_t* mu) {
  // `std::condition_variable::wait()` requires a unique lock, but to
  // emulate the pthreads API we are manually calling
  // `std::mutex::lock()` and `std::mutex::unlock()`, so we use adopt/release
  // to let the unique lock borrow the lock for the duration of this call.
  std::unique_lock<std::mutex> l(mu, std::adopt_lock);
  cond->wait(l);
  l.release();
  return 0;
}

inline int pthread_mutex_lock(pthread_mutex_t* mu) {
  mu->lock();
  return 0;
}

inline int pthread_mutex_unlock(pthread_mutex_t* mu) {
  mu->unlock();
  return 0;
}

}

#endif  // GEMMLOWP_INTERNAL_MULTI_THREAD_GEMM_H_
