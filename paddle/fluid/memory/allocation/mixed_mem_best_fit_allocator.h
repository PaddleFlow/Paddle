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

#pragma once

#include <atomic>
#include <mutex>
#include <unordered_set>

#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/detail/buddy_allocator.h"
#include "paddle/fluid/memory/detail/system_allocator.h"
#include "paddle/fluid/platform/cpu_info.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace memory {
namespace allocation {

struct MappedAddr {
  void* host_ptr_;
  size_t index_;
  size_t size_;
};

// MixedMemBestFitAllocator combines GPU memory and host pinned memory.
class MixedMemBestFitAllocator : public Allocator {
 public:
  explicit MixedMemBestFitAllocator(int device_id,
                                    const platform::CUDAPlace& place)
      : device_id_(device_id), device_place_(place) {
    device_allocator_ = std::make_shared<detail::BuddyAllocator>(
        std::unique_ptr<detail::SystemAllocator>(
            new detail::GPUAllocator(device_id)),
        platform::GpuMinChunkSize(), platform::GpuMaxChunkSize());

    // host_allocator_ = std::make_shared<detail::BuddyAllocator>(
    //     std::unique_ptr<detail::SystemAllocator>(
    //         new detail::CUDAPinnedAllocator()),
    //     platform::CUDAPinnedMinChunkSize(),
    //     platform::CUDAPinnedMaxChunkSize());
    host_allocator_ = std::make_shared<detail::CUDAPinnedAllocator>();
    VLOG(2) << "MixedMemBestFitAllocator created, device_id: " << device_id;
  }

  virtual ~MixedMemBestFitAllocator() {}

  bool IsAllocThreadSafe() const override { return true; }

  std::shared_ptr<detail::BuddyAllocator> GetDeviceAllocator() {
    VLOG(10) << "GetDeviceAllocator";
    return device_allocator_;
  }

 protected:
  Allocation* AllocateImpl(size_t size) override;
  void FreeImpl(Allocation* allocation) override;
  uint64_t ReleaseImpl(const platform::Place& place) override;

 private:
  int device_id_;

  platform::CUDAPlace device_place_;
  std::shared_ptr<detail::BuddyAllocator> device_allocator_;
  // std::shared_ptr<detail::BuddyAllocator> host_allocator_;
  std::shared_ptr<detail::SystemAllocator> host_allocator_;

  std::unordered_map<void*, MappedAddr> devptr2hostptr_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
