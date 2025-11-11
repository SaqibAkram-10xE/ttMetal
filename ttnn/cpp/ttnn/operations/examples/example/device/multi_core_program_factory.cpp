// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "example_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

uint32_t n_tiles_colIdx = 3;              // 16 or 64
uint32_t tile_size_bytes_codebook = 256;  //

namespace ttnn::operations::examples {
ExampleDeviceOperation::MultiCore::cached_program_t ExampleDeviceOperation::MultiCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& RowIdx_tensor = tensor_args.RowIdx_tensor;
    const auto& CodeBook_tensor = tensor_args.CodeBook_tensor;
    const auto& ColIdx_tensor = tensor_args.ColIdx_tensor;
    const auto& Scales_tensor = tensor_args.Scales_tensor;
    auto& output_tensor = tensor_return_value;

    auto colIdx_dram_buffer = ColIdx_tensor.buffer();
    auto codeBook_dram_buffer = CodeBook_tensor.buffer();
    auto rowIdx_dram_buffer = RowIdx_tensor.buffer();
    auto scales_dram_buffer = Scales_tensor.buffer();
    auto dst_dram_buffer = output_tensor.buffer();
    // fmt::print("{}\n", ColIdx_tensor.tensor_spec());

    tt::tt_metal::Program program{};

    tt::DataFormat cb_data_format_colIdx = tt::tt_metal::datatype_to_dataformat_converter(ColIdx_tensor.dtype());
    uint32_t tile_size_bytes_colIdx = tt::tile_size(cb_data_format_colIdx);

    tt::DataFormat cb_data_format_codebook = tt::tt_metal::datatype_to_dataformat_converter(CodeBook_tensor.dtype());
    // tile_size_bytes_codebook = tt::tile_size(cb_data_format_codebook);
    tile_size_bytes_codebook = tile_size_bytes_codebook * 2;  // bfloat16 uses 2 bytes

    tt::DataFormat cb_data_format_rowIdx = tt::tt_metal::datatype_to_dataformat_converter(RowIdx_tensor.dtype());
    uint32_t tile_size_bytes_rowIdx = tt::tile_size(cb_data_format_rowIdx);

    tt::DataFormat cb_data_format_scales = tt::tt_metal::datatype_to_dataformat_converter(Scales_tensor.dtype());
    uint32_t tile_size_bytes_scales = tt::tile_size(cb_data_format_scales);

    tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    uint32_t tile_size_bytes_output = tt::tile_size(cb_data_format_output);

    uint32_t n_tiles_colIdx = ColIdx_tensor.physical_volume() / tt::constants::TILE_HW;

    tt::tt_metal::IDevice* device = ColIdx_tensor.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core1, work_per_core2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, n_tiles_colIdx);

    fmt::print("\n=== Core & Work Distribution Info ===\n");
    fmt::print("Grid size (x, y): ({}, {})\n", compute_with_storage_grid_size.x, compute_with_storage_grid_size.y);
    fmt::print("num_cores_y                : {}\n", num_cores_y);
    fmt::print("num_cores                  : {}\n", num_cores);
    fmt::print("work_per_core1             : {}\n", work_per_core1);
    fmt::print("work_per_core2             : {}\n", work_per_core2);
    fmt::print("all_cores.size()           : {}\n", all_cores.size());
    fmt::print("core_group_1.size()        : {}\n", core_group_1.size());
    fmt::print("core_group_2.size()        : {}\n", core_group_2.size());

    // Optionally print core coordinates if you want details:
    // fmt::print("\n-- core_group_1 cores --\n");
    // for (const auto& core : core_group_1)
    //     fmt::print("({},{}) ", core.x, core.y);
    // fmt::print("\n");

    // fmt::print("\n-- core_group_2 cores --\n");
    // for (const auto& core : core_group_2)
    //     fmt::print("({},{}) ", core.x, core.y);
    // fmt::print("\n\n");

    tt::tt_metal::distributed::DeviceLocalBufferConfig dram_config_colIdx{
        .page_size = tile_size_bytes_colIdx,  // The page size of the buffer in bytes. Unlike the `loopback` example, we
                                              //  need the page size to be the same as the tile size for a large portion
                                              //  of the NoC transfer APIs to work.
        .buffer_type = BufferType::DRAM};     // This is a DRAM buffer.
    tt::tt_metal::distributed::DeviceLocalBufferConfig dram_config_rowIdx{
        .page_size = tile_size_bytes_rowIdx, .buffer_type = BufferType::DRAM};
    tt::tt_metal::distributed::DeviceLocalBufferConfig dram_config_codebook{
        .page_size = tile_size_bytes_codebook, .buffer_type = BufferType::DRAM};
    tt::tt_metal::distributed::DeviceLocalBufferConfig dram_config_scales{
        .page_size = tile_size_bytes_scales, .buffer_type = BufferType::DRAM};
    tt::tt_metal::distributed::DeviceLocalBufferConfig dram_config_output{
        .page_size = tile_size_bytes_output, .buffer_type = BufferType::DRAM};

    constexpr uint32_t tiles_per_cb = 2;
    tt::CBIndex colIdx_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(
            /*total_size=*/tiles_per_cb * tile_size_bytes_colIdx,  // The total size of the circular buffer in bytes
                                                                   /*data_format_spec=*/
            {{colIdx_cb_index, cb_data_format_colIdx}})  // The circular buffer index and data format it'll hold
            .set_page_size(colIdx_cb_index, tile_size_bytes_colIdx));  // Since we will be sending one tile at a time,
                                                                       // we set the page size to the tile size (and
                                                                       // thus total_size / page_size = tiles_per is the
                                                                       // number of entries in the circular buffer)
    tt::CBIndex rowIdx_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(
            /*total_size=*/tiles_per_cb * tile_size_bytes_rowIdx,
            /*data_format_spec=*/{{rowIdx_cb_index, cb_data_format_rowIdx}})
            .set_page_size(rowIdx_cb_index, tile_size_bytes_rowIdx));

    tt::CBIndex codeBook_cb_index = tt::CBIndex::c_2;
    tt::tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(
            /*total_size=*/tiles_per_cb * tile_size_bytes_codebook,
            /*data_format_spec=*/{{codeBook_cb_index, cb_data_format_codebook}})
            .set_page_size(codeBook_cb_index, tile_size_bytes_codebook));

    tt::CBIndex scales_cb_index = tt::CBIndex::c_3;
    tt::tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(
            /*total_size=*/tiles_per_cb * tile_size_bytes_scales,
            /*data_format_spec=*/{{scales_cb_index, cb_data_format_scales}})
            .set_page_size(scales_cb_index, tile_size_bytes_scales));

    tt::CBIndex dst_cb_index = tt::CBIndex::c_16;
    tt::tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(
            /*total_size=*/tiles_per_cb * tile_size_bytes_output,
            /*data_format_spec=*/{{dst_cb_index, cb_data_format_output}})
            .set_page_size(dst_cb_index, tile_size_bytes_output));

    std::vector<uint32_t> reader_compile_time_args;
    tt::tt_metal::TensorAccessorArgs(*colIdx_dram_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*rowIdx_dram_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*codeBook_dram_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*scales_dram_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*dst_dram_buffer).append_to(reader_compile_time_args);

    auto reader = CreateKernel(
        program,
        "tt_metal/programming_examples/eltwise_binary/kernels/dataflow/read_tiles.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    // std::vector<uint32_t> compute_kernel_args_group_1 = {
    //     num_tiles_per_core_group_1,  // per_core_block_cnt
    //     1                            // per_core_block_size
    // };

    // if (!core_group_2.ranges().empty()) {
    //     std::vector<uint32_t> compute_kernel_args_group_2 = {
    //         num_tiles_per_core_group_2,  // per_core_block_cnt
    //         1                            // per_core_block_size
    //     };

    //     tt::tt_metal::CreateKernel(
    //         program,
    //         "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp",
    //         core_group_2,
    //         tt::tt_metal::ComputeConfig{
    //             .math_fidelity = MathFidelity::HiFi4,
    //             .math_approx_mode = math_approx_mode,
    //             .compile_args = compute_kernel_args_group_2});
    // }

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = work_per_core1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = work_per_core2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        tt::tt_metal::SetRuntimeArgs(
            program,
            reader,
            core,
            {colIdx_dram_buffer->address(),
             rowIdx_dram_buffer->address(),
             codeBook_dram_buffer->address(),
             scales_dram_buffer->address(),
             dst_dram_buffer->address(),
             num_tiles_per_core,
             tile_size_bytes_codebook,
             num_tiles_written});

        num_tiles_written += num_tiles_per_core;
    }

    return {
        std::move(program),
        {.reader = reader,
         //  .writer = writer,
         .num_cores = num_cores,
         .num_cores_y = num_cores_y}};
}

void ExampleDeviceOperation::MultiCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& reader = cached_program.shared_variables.reader;
    // auto& writer = cached_program.shared_variables.writer;
    auto& num_cores = cached_program.shared_variables.num_cores;
    auto& num_cores_y = cached_program.shared_variables.num_cores_y;

    const auto& RowIdx_tensor = tensor_args.RowIdx_tensor;
    const auto& CodeBook_tensor = tensor_args.CodeBook_tensor;
    const auto& ColIdx_tensor = tensor_args.ColIdx_tensor;
    const auto& Scales_tensor = tensor_args.Scales_tensor;
    auto& output_tensor = tensor_return_value;

    auto colIdx_dram_buffer = ColIdx_tensor.buffer();
    auto codeBook_dram_buffer = CodeBook_tensor.buffer();
    auto rowIdx_dram_buffer = RowIdx_tensor.buffer();
    auto scales_dram_buffer = Scales_tensor.buffer();
    auto dst_dram_buffer = output_tensor.buffer();

    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        {
            // auto& runtime_args = GetRuntimeArgs(program, reader, core);
            // runtime_args[0] = src_buffer->address();

            auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, reader, core);
            runtime_args[0] = colIdx_dram_buffer->address();
            runtime_args[1] = rowIdx_dram_buffer->address();
            runtime_args[2] = codeBook_dram_buffer->address();
            runtime_args[3] = scales_dram_buffer->address();
            runtime_args[4] = dst_dram_buffer->address();
            runtime_args[5] = n_tiles_colIdx;
            runtime_args[6] = tile_size_bytes_codebook;
            // runtime_args[7] = src_buffer->address();
        }

        // {
        //     auto& runtime_args = GetRuntimeArgs(program, writer, core);
        //     runtime_args[0] = dst_buffer->address();
        // }
    }
}

}  // namespace ttnn::operations::examples
