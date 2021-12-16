/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "paddle/fluid/memory/detail/buddy_allocator.h"

namespace paddle {
namespace memory {
namespace management {

// using BuddyAllocator = detail::BuddyAllocator;
// using CUDAAllocatorAdjustor = allocation::CUDAAllocatorAdjustor;

struct GPUResourceLimitInfo {
  size_t mem_limit_;  // Mb
  // For recording VGPU_MEMORY_LIMIT
  size_t initial_mem_limit_;
  // uint32 sm_util;
};

class CUDAAllocatorAdjustor {
 public:
  size_t AdjustMemoryLimit(std::shared_ptr<detail::BuddyAllocator> allocator,
                           int device_id, size_t new_memory_limit);

  void GetMemPoolStats(std::shared_ptr<detail::BuddyAllocator> allocator,
                       int device_id, int64_t& deviceMemPoolSize,
                       int64_t& deviceMemStable);

 private:
  size_t FreeEmptyMemory(std::shared_ptr<detail::BuddyAllocator> allocator,
                         int device_id, size_t target_memory_bytes);
};

// For performing the adjustment of the usage of both the GPU memory and
// the SM (streaming multiprocessor).
class GPUUsageAdjustment {
 public:
  // Adjust the memory limit of the giving GPU.
  bool AdjustMemLimit(int device_id, size_t new_mem_limt);

 private:
  std::shared_ptr<detail::BuddyAllocator> GetGPUAllocator(int device_id);

  // Acquire this mutex before adjusting the GPU usage.
  std::mutex mutex_;

  struct GPUUsageInfo {
    std::shared_ptr<detail::BuddyAllocator> gpu_allocator_;
    GPUResourceLimitInfo cur_limit_;
  };

  // For recording current GPU usage info.
  // key: device_id
  std::unordered_map<int, GPUUsageInfo> cur_usage_info_;
};

}  // namespace management
}  // namespace memory
}  // namespace paddle
