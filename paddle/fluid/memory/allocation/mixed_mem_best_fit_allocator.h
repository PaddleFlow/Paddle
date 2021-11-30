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

#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/detail/buddy_allocator.h"
#include "paddle/fluid/memory/detail/system_allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

// MixedMemBestFitAllocator combines GPU memory and host pinned memory.
class MixedMemBestFitAllocator : public Allocator {
 public:
  explicit MixedMemBestFitAllocator(int device_id,
                                    std::shared_ptr<Allocator> device)
      : device_id_(device_id), device_allocator_(std::move(device)) {
    host_allocator_ = std::make_unique<detail::BuddyAllocator>(
        std::unique_ptr<detail::SystemAllocator>(
            new detail::CUDAPinnedAllocator),
        platform::CUDAPinnedMinChunkSize(), platform::CUDAPinnedMaxChunkSize());
    VLOG(9) << "MixedMemBestFitAllocator created, device_id: " << device_id;
  }

  virtual ~MixedMemBestFitAllocator() {}

  bool IsAllocThreadSafe() const override { return true; }

 protected:
  Allocation* AllocateImpl(size_t size) override;
  void FreeImpl(Allocation* allocation) override;
  uint64_t ReleaseImpl(const platform::Place& place) override;

 private:
  // std::mutex mutex_;
  std::atomic<bool> reach_limit_{false};
  int device_id_;
  std::shared_ptr<Allocator> device_allocator_;
  std::unique_ptr<detail::BuddyAllocator> host_allocator_;
};
}  // namespace allocation
}  // namespace memory
}  // namespace paddle
