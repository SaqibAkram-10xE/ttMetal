// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "example_device_operation.hpp"

namespace ttnn::operations::examples {

ExampleDeviceOperation::program_factory_t ExampleDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    bool some_condition_based_on_operation_attributes_and_or_tensor_args = true;
    if (some_condition_based_on_operation_attributes_and_or_tensor_args) {
        return SingleCore{};
    }
    return MultiCore{};
}

void ExampleDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

void ExampleDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

ExampleDeviceOperation::spec_return_value_t ExampleDeviceOperation::compute_output_specs(
    //My output would be same as ColIdx_tensor
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& ColIdx_tensor = tensor_args.ColIdx_tensor;
    return TensorSpec(
        ColIdx_tensor.logical_shape(),
        tt::tt_metal::TensorLayout(
            ColIdx_tensor.dtype(), tt::tt_metal::PageConfig(ColIdx_tensor.layout()), MemoryConfig{}));
}

ExampleDeviceOperation::tensor_return_value_t ExampleDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.ColIdx_tensor.device());
}

std::tuple<ExampleDeviceOperation::operation_attributes_t, ExampleDeviceOperation::tensor_args_t>
ExampleDeviceOperation::invoke(const Tensor& RowIdx_tensor,
                               const Tensor& CodeBook_tensor,
                               const Tensor& ColIdx_tensor) {
    return {operation_attributes_t{true, 42}, 
            tensor_args_t{RowIdx_tensor, CodeBook_tensor, ColIdx_tensor}};
}

}  // namespace ttnn::operations::examples