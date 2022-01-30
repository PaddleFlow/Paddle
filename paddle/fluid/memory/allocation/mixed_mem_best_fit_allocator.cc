// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/memory/allocation/mixed_mem_best_fit_allocator.h"

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"

// PADDLE_DEFINE_EXPORTED_bool(
//     init_allocated_mem, false,
//     "It is a mistake that the values of the memory allocated by "
//     "BuddyAllocator are always zeroed in some op's implementation. "
//     "To find this error in time, we use init_allocated_mem to indicate "
//     "that initializing the allocated memory with a small value "
//     "during unit testing.");

namespace paddle {
namespace memory {
namespace allocation {

Allocation* MixedMemBestFitAllocator::AllocateImpl(size_t size) {
  PADDLE_ENFORCE_NOT_NULL(
      device_allocator_,
      platform::errors::InvalidArgument("Underlying device allocator of "
                                        "MixedMemBestFitAllocator is nullptr"));

  PADDLE_ENFORCE_NOT_NULL(
      host_allocator_,
      platform::errors::InvalidArgument("Underlying host allocator of "
                                        "MixedMemBestFitAllocator is nullptr"));

  if (!device_allocator_->ReachLimit()) {
    try {
      void* ptr = device_allocator_->Alloc(size);
      PADDLE_ENFORCE_NOT_NULL(
          ptr, platform::errors::ResourceExhausted("cudaDeviceAlloc failed"));
      Allocation* tmp_alloc = new Allocation(ptr, size, device_place_);
      platform::MemEvenRecorder::Instance().PushMemRecord(
          static_cast<void*>(tmp_alloc), device_place_, size);
      return tmp_alloc;
    } catch (...) {
      VLOG(1) << "cuda allocation failed";
      throw;
    }
  } else {
    VLOG(2) << "device memory reached limit, try to allocate from host pinned "
               "memory";
    try {
      size_t index = 0;
      void* host_ptr = host_allocator_->Alloc(&index, size);
      PADDLE_ENFORCE_NOT_NULL(host_ptr, platform::errors::ResourceExhausted(
                                            "cudaHostAlloc failed"));

      void* dev_ptr;
      PADDLE_ENFORCE_CUDA_SUCCESS(
          cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0));
      PADDLE_ENFORCE_NOT_NULL(dev_ptr, platform::errors::ResourceExhausted(
                                           "cudaHostGetDevicePointer failed"));
      VLOG(10) << "system allocator converted host_ptr " << host_ptr
               << " to dev_ptr: " << dev_ptr << ", index: " << index
               << ", size: " << size;

      devptr2hostptr_.insert({dev_ptr, {host_ptr, index, size}});
      Allocation* tmp_alloc = new Allocation(dev_ptr, size, device_place_);
      return tmp_alloc;
    } catch (...) {
      VLOG(1) << "Still allocation failed using host memory";
      throw;
    }
  }

  return nullptr;
}

void MixedMemBestFitAllocator::FreeImpl(Allocation* allocation) {
  auto place = allocation->place();
  bool succ = false;

  auto it = devptr2hostptr_.find(allocation->ptr());
  if (it == devptr2hostptr_.end()) {
    device_allocator_->Free(allocation->ptr());
    succ = true;
    platform::MemEvenRecorder::Instance().PopMemRecord(
        static_cast<void*>(allocation), place);
  } else {
    host_allocator_->Free(it->second.host_ptr_, it->second.size_,
                          it->second.index_);
    devptr2hostptr_.erase(it);
    succ = true;
  }

  VLOG(9) << "FreeImpl called, place: " << place
          << ", addr: " << allocation->ptr()
          << ", size: " << allocation->size();

  if (succ) {
    // platform::MemEvenRecorder::Instance().PopMemRecord(
    //     static_cast<void*>(allocation), place);
    delete allocation;
  }
  return;
}

uint64_t MixedMemBestFitAllocator::ReleaseImpl(const platform::Place& place) {
  VLOG(9) << "ReleaseImpl called, place: " << place;
  uint64_t ret = 0;
  if (platform::is_gpu_place(place)) {
    ret = device_allocator_->Release();
  } else if (platform::is_cuda_pinned_place(place)) {
    // ignore
  }
  return ret;
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
