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

#include <atomic>
#include <memory>
#include <string>

#include "gflags/gflags.h"
#include "json/json.h"
#include "paddle/fluid/memory/management/file_listener.h"
#include "paddle/fluid/memory/management/gpu_usage_adjustment.h"

DECLARE_string(job_name);

namespace paddle {
namespace memory {
namespace management {

// Determine if we need to adjust the GPU resource at the end of
// each RunStep.
class GPUResourceManagement {
 public:
  GPUResourceManagement();
  ~GPUResourceManagement();

  // For adjusting the limit of each resource after each SessionRun.
  // Status RunAction(const SessionRunActionOptions& options) override;
  int Run();

  // Get the new gpu resource limit from the json string.
  bool ParseManageInfoFromJson(const std::string& json_str);

  // Disable the GPUResourceManagement feature.
  void DisableGPUResourceManagement() { enable_gpu_resource_manage_ = false; }

  // Get the estimated total idle time.
  uint64_t GetEstimatedIdleTime() { return estimated_total_idle_time_; }

  // Get the current total idle time.
  void SetEstimatedIdleTime(uint64_t idle_time) {
    // mutex_lock l(idle_time_mu_);
    estimated_total_idle_time_ = idle_time;
  }

  // Get the total number of queued GPU op running in
  // the giving executor.
  uint64_t GetExecutorQueuedOpNum(const void* executor_ptr);

  // Set the total number of queued GPU op running in
  // the giving executor.
  void SetExecutorQueuedOpNum(const void* executor_ptr, uint64_t queued_op_num);

 private:
  // Adjust the GPU usage limit of this job.
  // void AdjustUsage();

  // Get the new GPU memory limit from the json string.
  void ParseMemoryLimitFromJson(const Json::Value& json);

  // Get the new GPU usage limit from the json string.
  void ParseUsageLimitFromJson(const Json::Value& json);

  // Sleep a specific time after each SessionRun or Suspend this job.
  void DoSleepOrSuspend(uint64_t sess_duration_us);

  // mutable std::mutex mutex_lock;
  mutable std::mutex mutex_;
  mutable std::mutex manage_mu_;
  mutable std::mutex usage_mu_;
  mutable std::mutex idle_time_mu_;

  // Mark whether the GPUResourceManagement feature
  // is enabled.
  std::atomic<bool> enable_gpu_resource_manage_{false};

  // For recording the parsed new gpu resource limit.
  std::unordered_map<int, GPUResourceLimitInfo> gpu_resource_management_info_;

  // For recording the parsed new gpu performance limitation
  // (if the value is 0, then it means to suspend this job).
  std::atomic<int> gpu_perf_control_{100};

  // For recording the total time of all inserted time slot.
  uint64_t total_time_slot_{0};

  // For recording the estimated total idle time.
  uint64_t estimated_total_idle_time_{0};

  // For recording the total number of queued GPU op running in
  // the specified executor.
  std::unordered_map<const void*, uint64_t> executor_queued_op_num_;

  // Determine if we need to adjust the GPU usage limit.
  std::atomic<bool> need_to_adjust_memory_;

  // For performing the adjustment.
  std::unique_ptr<GPUUsageAdjustment> gpu_usage_adjustment_;

  std::string gpu_resource_manage_file_path_;
  const std::string FILE_LISTENER_NAME = "GPUResourceManage";

  const std::string job_name_;
};

}  // namespace management
}  // namespace memory
}  // namespace paddle
