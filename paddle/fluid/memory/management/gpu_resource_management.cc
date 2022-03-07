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

#include "paddle/fluid/memory/management/gpu_resource_management.h"

#include <string>

#include "glog/logging.h"

namespace paddle {
namespace memory {
namespace management {

int ReadStringFromEnvVar(const std::string& env_var_name,
                         const std::string& default_val, std::string& value) {
  const char* env_var_val = getenv(env_var_name.c_str());
  if (env_var_val != nullptr) {
    value = env_var_val;
  } else {
    value = default_val;
  }
  return 0;
}

GPUResourceManagement::GPUResourceManagement() : task_name_(FLAGS_task_name) {
  ReadStringFromEnvVar("GPU_CONFIG_FILE", "", gpu_resource_manage_file_path_);
  if (gpu_resource_manage_file_path_.empty()) {
    enable_gpu_resource_manage_ = false;
    VLOG(1) << "no GPU_CONFIG_FILE provided";
    return;
  }

  if (task_name_.empty()) {
    VLOG(1) << "no task_name";
  }

  enable_gpu_resource_manage_ = true;
  gpu_usage_adjustment_ = std::make_unique<GPUUsageAdjustment>();

  // Register a handler that will be triggered when the file named
  FileListener::GlobalFileListener()->RegisterFileListener(
      gpu_resource_manage_file_path_, FILE_LISTENER_NAME,
      [&](const std::string& str) {
        VLOG(2) << "GPU resource management registered target file.";
        this->ParseManageInfoFromJson(str);
        this->Run();
      });
}

GPUResourceManagement::~GPUResourceManagement() {
  if (enable_gpu_resource_manage_) {
    FileListener::GlobalFileListener()->UnregisterFileListener(
        gpu_resource_manage_file_path_, FILE_LISTENER_NAME);
  }
}

int GPUResourceManagement::Run() {
  VLOG(2) << "GPUResourceManagement::Run(), adjust: " << need_to_adjust_memory_
          << ", size: " << gpu_resource_management_info_.size();
  if (!need_to_adjust_memory_) {
    return 0;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& it : gpu_resource_management_info_) {
    gpu_usage_adjustment_->AdjustMemLimit(it.first, it.second.mem_limit_);
  }
  need_to_adjust_memory_ = false;  // done

  DoSleepOrSuspend(100);

  return 0;
}

bool GPUResourceManagement::ParseManageInfoFromJson(
    const std::string& json_str) {
  Json::Reader reader;
  Json::Value json;

  if (!reader.parse(json_str, json)) {
    LOG(INFO) << "Failed to parse the json string";
    return false;
  }

  ParseMemoryLimitFromJson(json);
  ParseUsageLimitFromJson(json);
  VLOG(1) << "gpu resource config updated.";
  return true;
}

uint64_t GPUResourceManagement::GetExecutorQueuedOpNum(
    const void* executor_ptr) {
  // TODO
  return 0;
}

void GPUResourceManagement::ParseMemoryLimitFromJson(const Json::Value& json) {
  if (json["gpuConfigInfo"].isNull()) {
    return;
  }
  // mutex_lock l(manage_mu_);
  std::lock_guard<std::mutex> lock(mutex_);

  auto gpu_infos = json["gpuConfigInfo"];
  if (!gpu_infos.isMember(task_name_)) {
    VLOG(1) << "not gpu config info provided for " << task_name_;
    return;
  }
  auto selected_info = gpu_infos[task_name_];

  int device_id = 0;
  if (selected_info["device_id"].isNull() ||
      !selected_info["device_id"].isInt()) {
    VLOG(1) << "invliad device_id field";
    return;
  }
  if (selected_info["maxDeviceMemMb"].isNull() ||
      !selected_info["maxDeviceMemMb"].isUInt64()) {
    VLOG(1) << "invliad maxDeviceMemMb field";
    return;
  }
  device_id = selected_info["device_id"].asInt();

  GPUResourceLimitInfo limit_info;
  limit_info.mem_limit_ = selected_info["maxDeviceMemMb"].asUInt64() << 20;
  auto res = gpu_resource_management_info_.emplace(device_id, limit_info);
  if (res.second == false) {
    res.first->second.mem_limit_ = limit_info.mem_limit_;
  }

  VLOG(0) << "Parse GPU device_id: " << device_id
          << " maxDeviceMem: " << limit_info.mem_limit_;

  if (gpu_resource_management_info_.size() != 0) {
    need_to_adjust_memory_ = true;
  }
}

void GPUResourceManagement::ParseUsageLimitFromJson(const Json::Value& json) {
  // TODO perfControl
}

void GPUResourceManagement::SetExecutorQueuedOpNum(const void* executor_ptr,
                                                   uint64_t queued_op_num) {
  // TODO
  return;
}

void GPUResourceManagement::DoSleepOrSuspend(uint64_t sess_duration_us) {
  // TODO
  usleep(sess_duration_us);
  return;
}

}  // namespace management
}  // namespace memory
}  // namespace paddle
