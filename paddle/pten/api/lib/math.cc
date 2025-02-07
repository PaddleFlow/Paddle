/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/pten/api/include/math.h"

#include <memory>

#include "glog/logging.h"

#include "paddle/pten/api/lib/api_registry.h"
#include "paddle/pten/api/lib/kernel_dispatch.h"
#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/include/core.h"
#include "paddle/pten/include/infermeta.h"
#include "paddle/pten/infermeta/unary.h"

PT_DECLARE_MODULE(MathCPU);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PT_DECLARE_MODULE(MathCUDA);
#endif

namespace paddle {
namespace experimental {

PD_DLL_DECL Tensor mean(const Tensor& x,
                        const std::vector<int64_t>& axis,
                        bool keep_dim) {
  // 1. Get kernel signature and kernel
  auto kernel_key_set = ParseKernelKeyByInputArgs(x);
  auto kernel_key = kernel_key_set.GetHigestPriorityKernelKey();
  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
      "reduce_mean", kernel_key);

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());
  auto kernel_context = pten::KernelContext(dev_ctx);

  // 3. Auto data transform
  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());
  kernel_context.EmplaceBackInput(dense_x);

  // The real value of reduce_all will be get in kernel
  // so use default value(false) is OK.
  bool reduce_all = false;

  DataType out_dtype = DataType::UNDEFINED;

  kernel_context.EmplaceBackAttr(axis);
  kernel_context.EmplaceBackAttr(keep_dim);
  kernel_context.EmplaceBackAttr(reduce_all);
  kernel_context.EmplaceBackAttr(dense_x->dtype());
  kernel_context.EmplaceBackAttr(out_dtype);

  // 4. InferShape
  auto out_meta = ReduceInferMeta(dense_x->meta(), axis, keep_dim);

  // 5. Prepare outputs
  Tensor out;
  const auto allocator =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          pten::TransToFluidPlace(kernel_key.backend()));
  auto dense_out = std::make_shared<pten::DenseTensor>(allocator, out_meta);
  kernel_context.EmplaceBackOutput(dense_out);
  out.set_impl(dense_out);

  // 6. Call kernel
  kernel(&kernel_context);

  return out;
}

PD_DLL_DECL Tensor sum(const Tensor& x,
                       const std::vector<int64_t>& axis,
                       DataType dtype,
                       bool keep_dim) {
  // 1. Get kernel signature and kernel
  auto kernel_key_set = ParseKernelKeyByInputArgs(x);
  auto kernel_key = kernel_key_set.GetHigestPriorityKernelKey();
  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
      "reduce_sum", kernel_key);

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());
  auto kernel_context = pten::KernelContext(dev_ctx);

  // 3. Auto data transform
  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());
  kernel_context.EmplaceBackInput(dense_x);

  // The real value of reduce_all will be get in kernel
  // so use default value(false) is OK.
  bool reduce_all = false;

  DataType out_dtype = DataType::UNDEFINED;
  if (dense_x->dtype() == DataType::BOOL ||
      dense_x->dtype() == DataType::INT32 ||
      dense_x->dtype() == DataType::INT64) {
    out_dtype = DataType::INT64;
  }

  kernel_context.EmplaceBackAttr(axis);
  kernel_context.EmplaceBackAttr(keep_dim);
  kernel_context.EmplaceBackAttr(reduce_all);
  kernel_context.EmplaceBackAttr(dense_x->dtype());
  kernel_context.EmplaceBackAttr(out_dtype);

  // 4. InferMeta
  auto out_meta = ReduceInferMeta(dense_x->meta(), axis, keep_dim);

  // 5. Prepare outputs
  Tensor out;
  const auto allocator =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          pten::TransToFluidPlace(kernel_key.backend()));
  auto dense_out = std::make_shared<pten::DenseTensor>(allocator, out_meta);
  kernel_context.EmplaceBackOutput(dense_out);
  out.set_impl(dense_out);

  // 6. Call kernel
  kernel(&kernel_context);

  return out;
}

PD_DLL_DECL Tensor add(const Tensor& x, const Tensor& y) {
  // 1. Get kernel signature and kernel
  auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
  auto kernel_key = kernel_key_set.GetHigestPriorityKernelKey();
  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
      "elementwise_add", kernel_key);

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());
  auto kernel_context = pten::KernelContext(dev_ctx);

  // 3. Auto data transform
  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());
  kernel_context.EmplaceBackInput(dense_x);
  auto dense_y = std::dynamic_pointer_cast<pten::DenseTensor>(y.impl());
  kernel_context.EmplaceBackInput(dense_y);
  kernel_context.EmplaceBackAttr(-1);

  // 4. InferMeta
  auto out_meta = ElementwiseInferMeta(dense_x->meta(), dense_y->meta(), -1);

  // 5. Prepare outputs
  Tensor out;
  const auto allocator = std::make_shared<DefaultAllocator>(
      pten::TransToFluidPlace(kernel_key.backend()));
  auto dense_out = std::make_shared<pten::DenseTensor>(allocator, out_meta);
  kernel_context.EmplaceBackOutput(dense_out);
  out.set_impl(dense_out);

  // 6. Call kernel
  kernel(&kernel_context);

  return out;
}

PD_DLL_DECL Tensor subtract(const Tensor& x, const Tensor& y) {
  // 1. Get kernel signature and kernel
  auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
  auto kernel_key = kernel_key_set.GetHigestPriorityKernelKey();
  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
      "elementwise_sub", kernel_key);

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());
  auto kernel_context = pten::KernelContext(dev_ctx);

  // 3. Auto data transform
  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());
  kernel_context.EmplaceBackInput(dense_x);
  auto dense_y = std::dynamic_pointer_cast<pten::DenseTensor>(y.impl());
  kernel_context.EmplaceBackInput(dense_y);
  kernel_context.EmplaceBackAttr(-1);

  // 4. InferMeta
  auto out_meta = ElementwiseInferMeta(dense_x->meta(), dense_y->meta(), -1);

  // 5. Prepare outputs
  Tensor out;
  const auto allocator = std::make_shared<DefaultAllocator>(
      pten::TransToFluidPlace(kernel_key.backend()));
  auto dense_out = std::make_shared<pten::DenseTensor>(allocator, out_meta);
  kernel_context.EmplaceBackOutput(dense_out);
  out.set_impl(dense_out);

  // 6. Call kernel
  kernel(&kernel_context);

  return out;
}

PD_DLL_DECL Tensor divide(const Tensor& x, const Tensor& y) {
  // 1. Get kernel signature and kernel
  auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
  auto kernel_key = kernel_key_set.GetHigestPriorityKernelKey();
  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
      "elementwise_div", kernel_key);

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());
  auto kernel_context = pten::KernelContext(dev_ctx);

  // 3. Auto data transform
  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());
  kernel_context.EmplaceBackInput(dense_x);
  auto dense_y = std::dynamic_pointer_cast<pten::DenseTensor>(y.impl());
  kernel_context.EmplaceBackInput(dense_y);
  kernel_context.EmplaceBackAttr(-1);

  // 4. InferMeta
  auto out_meta = ElementwiseInferMeta(dense_x->meta(), dense_y->meta(), -1);

  // 5. Prepare outputs
  Tensor out;
  const auto allocator = std::make_shared<DefaultAllocator>(
      pten::TransToFluidPlace(kernel_key.backend()));
  auto dense_out = std::make_shared<pten::DenseTensor>(allocator, out_meta);
  kernel_context.EmplaceBackOutput(dense_out);
  out.set_impl(dense_out);

  // 6. Call kernel
  kernel(&kernel_context);

  return out;
}

PD_DLL_DECL Tensor multiply(const Tensor& x, const Tensor& y) {
  // 1. Get kernel signature and kernel
  auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
  auto kernel_key = kernel_key_set.GetHigestPriorityKernelKey();
  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
      "elementwise_mul", kernel_key);

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());
  auto kernel_context = pten::KernelContext(dev_ctx);

  // 3. Auto data transform
  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());
  kernel_context.EmplaceBackInput(dense_x);
  auto dense_y = std::dynamic_pointer_cast<pten::DenseTensor>(y.impl());
  kernel_context.EmplaceBackInput(dense_y);
  kernel_context.EmplaceBackAttr(-1);

  // 4. InferMeta
  auto out_meta = ElementwiseInferMeta(dense_x->meta(), dense_y->meta(), -1);

  // 5. Prepare outputs
  Tensor out;
  const auto allocator = std::make_shared<DefaultAllocator>(
      pten::TransToFluidPlace(kernel_key.backend()));
  auto dense_out = std::make_shared<pten::DenseTensor>(allocator, out_meta);
  kernel_context.EmplaceBackOutput(dense_out);
  out.set_impl(dense_out);

  // 6. Call kernel
  kernel(&kernel_context);

  return out;
}

PD_DLL_DECL Tensor scale(const Tensor& x,
                         const Scalar& scale,
                         float bias,
                         bool bias_after_scale) {
  // 1. Get kernel signature and kernel
  auto kernel_key_set = ParseKernelKeyByInputArgs(x);
  auto kernel_key = kernel_key_set.GetHigestPriorityKernelKey();
  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
      "scale", kernel_key);

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());
  auto kernel_context = pten::KernelContext(dev_ctx);

  // 3. Auto data transform
  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());
  kernel_context.EmplaceBackInput(dense_x);
  kernel_context.EmplaceBackAttr(pten::Scalar(scale));
  kernel_context.EmplaceBackAttr(bias);
  kernel_context.EmplaceBackAttr(bias_after_scale);

  // 4. InferMeta
  auto out_meta = UnchangedInferMeta(dense_x->meta());

  // 5. Prepare outputs
  Tensor out;
  const auto allocator = std::make_shared<DefaultAllocator>(
      pten::TransToFluidPlace(kernel_key.backend()));
  auto dense_out = std::make_shared<pten::DenseTensor>(allocator, out_meta);
  kernel_context.EmplaceBackOutput(dense_out);
  out.set_impl(dense_out);

  // 6. Call kernel
  kernel(&kernel_context);

  return out;
}

}  // namespace experimental
}  // namespace paddle

PT_REGISTER_API(Math);
