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

#include "paddle/fluid/memory/management/gpu_usage_adjustment.h"

#include <climits>

#include "glog/logging.h"
#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/fluid/memory/allocation/mixed_mem_best_fit_allocator.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace memory {
namespace management {

size_t CUDAAllocatorAdjustor::AdjustMemoryLimit(
    std::shared_ptr<detail::BuddyAllocator> allocator, int device_id,
    size_t new_memory_limit) {
  PADDLE_ENFORCE_NOT_NULL(allocator,
                          platform::errors::NotFound("Invliad parameter"));

  VLOG(0) << "try to adjust device_id " << device_id << " to "
          << new_memory_limit << ", current allocated "
          << allocator->TotalAllocated();
  size_t new_limit = new_memory_limit;
  if (new_memory_limit >= allocator->TotalAllocated()) {
    // bool ok =
    // paddle::platform::RecordedLimitResize(device_id, new_memory_limit);
    bool ok = allocator->ResizeLimit(new_memory_limit);
    VLOG(2) << "RecordedLimitResize: " << ok;
    if (ok) {
      new_limit = new_memory_limit;
    } else {
      // TODO
    }
  } else {
    // total_region_allocated_bytes_ > new_memory_limit:
    // shrink, need to free memory
    size_t free_res = FreeEmptyMemory(allocator, device_id, new_memory_limit);
    VLOG(2) << "after free, alloc size " << free_res;
    if (free_res <= new_memory_limit) {
      VLOG(2) << "successful";
      new_limit = new_memory_limit;
    } else {
      VLOG(2) << "failed";
      new_limit = free_res;
    }
    // bool ok = paddle::platform::RecordedLimitResize(device_id, new_limit);
    bool ok = allocator->ResizeLimit(free_res);
    VLOG(2) << "RecordedLimitResize: " << ok;
  }

  return new_limit;
}

void CUDAAllocatorAdjustor::GetMemPoolStats(
    std::shared_ptr<detail::BuddyAllocator> allocator, int device_id,
    int64_t& deviceMemPoolSize, int64_t& deviceMemStable) {
  // TODO
  return;
}

size_t CUDAAllocatorAdjustor::FreeEmptyMemory(
    std::shared_ptr<detail::BuddyAllocator> allocator, int device_id,
    size_t target_memory_bytes) {
  size_t free_res = allocator->Release();
  VLOG(2) << "Release: " << free_res;
  return allocator->TotalAllocated();
}

bool GPUUsageAdjustment::AdjustMemLimit(int device_id, size_t new_mem_limit) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto cur_info = cur_usage_info_.find(device_id);
  if (cur_info == cur_usage_info_.cend()) {
    auto allo = GetGPUAllocator(device_id);
    if (allo == nullptr) {
      LOG(ERROR) << "Failed to get the allocator of device_id: " << device_id;
      return false;
    }

    GPUUsageInfo usage_info;
    usage_info.gpu_allocator_ = allo;
    usage_info.cur_limit_.mem_limit_ = ULONG_MAX;

    auto ret = cur_usage_info_.emplace(device_id, usage_info);
    if (ret.second == false) {
      return false;
    }

    cur_info = ret.first;
  }

  if (cur_info->second.cur_limit_.mem_limit_ == new_mem_limit) {
    // unchanged
    VLOG(1) << "limit unchanged.";
    return false;
  }

  std::unique_ptr<CUDAAllocatorAdjustor> adj =
      std::make_unique<CUDAAllocatorAdjustor>();

  VLOG(2) << "Start to manage the mem size limit to " << new_mem_limit
          << "b of device: " << device_id;

  size_t cur_mem_limit = adj->AdjustMemoryLimit(cur_info->second.gpu_allocator_,
                                                device_id, new_mem_limit);

  if (cur_mem_limit > new_mem_limit) {
    VLOG(2) << "Failed to manage the mem size limit to " << new_mem_limit
            << " of device device_id: " << device_id;
    return false;
  }

  return true;
}

std::shared_ptr<detail::BuddyAllocator> GPUUsageAdjustment::GetGPUAllocator(
    int device_id) {
  // get allocator from allocator_facade
  // down cast to mix
  platform::CUDAPlace place(device_id);
  auto allocator = allocation::AllocatorFacade::Instance().GetAllocator(place);
  PADDLE_ENFORCE_NOT_NULL(
      allocator,
      platform::errors::NotFound("Allocator not found for device %s", place));
  VLOG(2) << "allocator type " << typeid(*allocator.get()).name();
  auto mixed_allocator =
      std::dynamic_pointer_cast<allocation::MixedMemBestFitAllocator>(
          allocator);
  PADDLE_ENFORCE_NOT_NULL(mixed_allocator,
                          platform::errors::NotFound(
                              "MixedAllocator not found for device %s", place));
  PADDLE_ENFORCE_NOT_NULL(
      mixed_allocator->GetDeviceAllocator(),
      platform::errors::NotFound("MixedDeviceAllocator not found for device %s",
                                 place));
  return mixed_allocator->GetDeviceAllocator();
}

}  // namespace management
}  // namespace memory
}  // namespace paddle
