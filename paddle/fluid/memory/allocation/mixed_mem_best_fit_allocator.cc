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

#include "paddle/fluid/platform/cpu_info.h"
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

  if (!reach_limit_) {
    try {
      return device_allocator_->Allocate(size).release();
    } catch (const BadAlloc& exp) {
      const size_t host_max_size = paddle::platform::CpuMaxAllocSize();
      VLOG(1) << "Not enough GPU memory, try to use cuda pinned memory as "
                 "supplement, max host memory: "
              << host_max_size << ", required size: " << size;
      reach_limit_ = true;
    } catch (...) {
      throw;
    }
  }

  if (reach_limit_) {
    try {
      void* ptr = host_allocator_->Alloc(size);
      if (ptr == nullptr) {
        LOG(WARNING) << "cudaHostAlloc Cannot allocate " << size
                     << " bytes in CUDAPinnedPlace";
      }
      PADDLE_ENFORCE_NOT_NULL(
          ptr, platform::errors::ResourceExhausted("cudaHostAlloc failed"));
      // if (FLAGS_init_allocated_mem) {
      //   memset(ptr, 0xEF, size);
      // }

      Allocation* tmp_alloc =
          new Allocation(ptr, size, platform::CUDAPinnedPlace());
      platform::MemEvenRecorder::Instance().PushMemRecord(
          static_cast<void*>(tmp_alloc), platform::CUDAPinnedPlace(), size);
      return tmp_alloc;
    } catch (...) {
      VLOG(1) << "Still allocation failed using host memory";
      throw;
    }
  }

  return nullptr;
}

void MixedMemBestFitAllocator::FreeImpl(Allocation* allocation) {
  const auto place = allocation->place();
  VLOG(9) << "FreeImpl called, place: " << place
          << ", addr: " << allocation->ptr()
          << ", size: " << allocation->size();
  if (platform::is_gpu_place(place)) {
    device_allocator_->Free(allocation);
  } else if (platform::is_cuda_pinned_place(place)) {
    host_allocator_->Free(allocation->ptr());
    platform::MemEvenRecorder::Instance().PopMemRecord(
        static_cast<void*>(allocation), place);
    delete allocation;
  }
  return;
}

uint64_t MixedMemBestFitAllocator::ReleaseImpl(const platform::Place& place) {
  VLOG(9) << "ReleaseImpl called, place: " << place;
  uint64_t ret = 0;
  if (platform::is_gpu_place(place)) {
    ret = device_allocator_->Release(place);
  } else if (platform::is_cuda_pinned_place(place)) {
    ret = host_allocator_->Release();
  }
  return ret;
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
