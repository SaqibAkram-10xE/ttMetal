// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "example_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

constexpr uint32_t n_tiles_colIdx = 3;   // 16 or 64


namespace ttnn::operations::examples {
ExampleDeviceOperation::SingleCore::cached_program_t ExampleDeviceOperation::SingleCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& RowIdx_tensor = tensor_args.RowIdx_tensor;
    const auto& CodeBook_tensor = tensor_args.CodeBook_tensor;
    const auto& ColIdx_tensor = tensor_args.ColIdx_tensor;
    auto& output_tensor = tensor_return_value;

    // auto src_buffer = input_tensor.buffer();
    // auto dst_buffer = output_tensor.buffer();

    auto colIdx_dram_buffer = ColIdx_tensor.buffer();
    auto codeBook_dram_buffer = CodeBook_tensor.buffer();
    auto rowIdx_dram_buffer = RowIdx_tensor.buffer();
    auto dst_dram_buffer = output_tensor.buffer();


    tt::tt_metal::Program program{};

    constexpr CoreCoord core = {0, 0};

    // we are sure to use the uint8_t now so,

    tt::DataFormat cb_data_format_colIdx = tt::tt_metal::datatype_to_dataformat_converter(ColIdx_tensor.dtype());
    uint32_t tile_size_bytes_colIdx = tt::tile_size(cb_data_format_colIdx);
    
    tt::DataFormat cb_data_format_codebook = tt::tt_metal::datatype_to_dataformat_converter(CodeBook_tensor.dtype());
    uint32_t tile_size_bytes_codebook = tt::tile_size(cb_data_format_codebook);
    
    tt::DataFormat cb_data_format_rowIdx = tt::tt_metal::datatype_to_dataformat_converter(RowIdx_tensor.dtype());
    uint32_t tile_size_bytes_rowIdx = tt::tile_size(cb_data_format_rowIdx);
    
    tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    uint32_t tile_size_bytes_output = tt::tile_size(cb_data_format_output);


    // TODO: pass it through the method once 
    // we have a send to end model setup.
    // constexpr uint32_t n_tiles_colIdx = 3;   // 16 or 64

    // constexpr uint32_t n_tiles_codebook = 1; //unused

    // constexpr uint32_t n_tiles_rowIdx = 1;    // 1 or 16
    constexpr uint32_t elements_per_tile_colIdx = tt::constants::TILE_HW;
    constexpr uint32_t elements_per_tile_rowIdx = tt::constants::TILE_HW;
    // constexpr uint32_t elements_per_tile_codebook = 64;
    // constexpr uint32_t tile_size_bytes_colIdx = sizeof(uint8_t) * elements_per_tile_colIdx;
    // constexpr uint32_t tile_size_bytes_codebook = sizeof(uint8_t) * elements_per_tile_codebook;
    // constexpr uint32_t tile_size_bytes_rowIdx = sizeof(uint8_t) * elements_per_tile_rowIdx;

    uint32_t n_tiles_colIdx = ColIdx_tensor.physical_volume() / tt::constants::TILE_HW;
    uint32_t n_tiles_rowIdx = RowIdx_tensor.physical_volume() / tt::constants::TILE_HW;
    // uint32_t n_tiles_codebook = CodeBook_tensor.physical_volume() / tt::constants::TILE_HW;

    // CoreCoord compute_with_storage_grid_size = {1, 1};
    // uint32_t num_cores_y = compute_with_storage_grid_size.y;
    // auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
    //     tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, n_tiles_colIdx);
    
    // auto core_grid = device->compute_with_storage_grid_size();
    // auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
    //     tt::tt_metal::split_work_to_cores(core_grid, n_tiles_colIdx);



    tt::tt_metal::distributed::DeviceLocalBufferConfig dram_config_colIdx{
        .page_size = tile_size_bytes_colIdx, //The page size of the buffer in bytes. Unlike the `loopback` example, we
                                      // need the page size to be the same as the tile size for a large portion of the NoC transfer APIs to work.
        .buffer_type = BufferType::DRAM}; // This is a DRAM buffer.
    tt::tt_metal::distributed::DeviceLocalBufferConfig dram_config_rowIdx{
        .page_size = tile_size_bytes_rowIdx, 
        .buffer_type = BufferType::DRAM};
    tt::tt_metal::distributed::DeviceLocalBufferConfig dram_config_codebook{
        .page_size = tile_size_bytes_codebook,
        .buffer_type = BufferType::DRAM};
    tt::tt_metal::distributed::DeviceLocalBufferConfig dram_config_output{
        .page_size = tile_size_bytes_output,
        .buffer_type = BufferType::DRAM};

    //Unused

    // tt::tt_metal::distributed::ReplicatedBufferConfig buffer_config{
    //     .size = n_tiles_colIdx * tile_size_bytes_colIdx // Total bytes per device (replicated across the mesh).
    // };
    // tt::tt_metal::distributed::ReplicatedBufferConfig buffer_config_rowIdx{
    //     .size = n_tiles_rowIdx * tile_size_bytes_rowIdx // Total bytes per device (replicated across the mesh).
    // };
    // tt::tt_metal::distributed::ReplicatedBufferConfig buffer_config_codebook{
    //     .size = n_tiles_codebook * tile_size_bytes_codebook // Total bytes per device (replicated across the mesh).
    // };
    // tt::tt_metal::distributed::ReplicatedBufferConfig buffer_config_output{
    //     .size = n_tiles_colIdx * tile_size_bytes_output // Total bytes per device (replicated across the mesh).
    // };


    /*
    distributed::EnqueueWriteMeshBuffer(cq, colIdx_dram_buffer, colIdx_data, false);
    distributed::EnqueueWriteMeshBuffer(cq, rowIdx_dram_buffer, rowIdx_data, false);
    distributed::EnqueueWriteMeshBuffer(cq, codeBook_dram_buffer, codeBook_data, false);
    */

/*
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t output_cb_index = tt::CBIndex::c_2;
    uint32_t num_output_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            num_output_tiles * single_tile_size_output, {{output_cb_index, cb_data_format_output}})
            .set_page_size(output_cb_index, single_tile_size_output);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

*/

    constexpr uint32_t tiles_per_cb = 2;
    tt::CBIndex colIdx_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CreateCircularBuffer(program, core, CircularBufferConfig(
        /*total_size=*/tiles_per_cb * tile_size_bytes_colIdx,                    // The total size of the circular buffer in bytes
        /*data_format_spec=*/{{colIdx_cb_index, cb_data_format_colIdx}})// The circular buffer index and data format it'll hold
        .set_page_size(colIdx_cb_index, tile_size_bytes_colIdx));                  // Since we will be sending one tile at a time, we set
                                                                          // the page size to the tile size (and thus
                                                                          // total_size / page_size = tiles_per is the number of
                                                                          // entries in the circular buffer)
    tt::CBIndex rowIdx_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CreateCircularBuffer(program, core, CircularBufferConfig(
        /*total_size=*/tiles_per_cb * tile_size_bytes_rowIdx,
        /*data_format_spec=*/{{rowIdx_cb_index, cb_data_format_rowIdx}})
        .set_page_size(rowIdx_cb_index, tile_size_bytes_rowIdx));

    tt::CBIndex codeBook_cb_index = tt::CBIndex::c_2;
    tt::tt_metal::CreateCircularBuffer(program, core, CircularBufferConfig(
        /*total_size=*/tiles_per_cb * tile_size_bytes_codebook,
        /*data_format_spec=*/{{codeBook_cb_index, cb_data_format_codebook}})
        .set_page_size(codeBook_cb_index, tile_size_bytes_codebook));

    tt::CBIndex dst_cb_index = tt::CBIndex::c_16;
    tt::tt_metal::CreateCircularBuffer(program, core, CircularBufferConfig(
        /*total_size=*/tiles_per_cb * tile_size_bytes_output,
        /*data_format_spec=*/{{dst_cb_index, cb_data_format_output}})
        .set_page_size(dst_cb_index, tile_size_bytes_output));


    std::vector<uint32_t> reader_compile_time_args;
    tt::tt_metal::TensorAccessorArgs(*colIdx_dram_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*rowIdx_dram_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*codeBook_dram_buffer).append_to(reader_compile_time_args);
    
    auto reader = CreateKernel(
        program,
        "tt_metal/programming_examples/eltwise_binary/kernels/dataflow/read_tiles.cpp",
        core,
        tt::tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_compile_time_args});
        //tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);

    auto writer = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/eltwise_binary/kernels/dataflow/write_tile.cpp",
        core,
        tt::tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = writer_compile_time_args});
    
    auto compute = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/eltwise_binary/kernels/compute/tiles_add.cpp",
        core,
        tt::tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4});   // There's different math fidelity modes (for the tensor engine)
                                                            // that trade off performance for accuracy. HiFi4 is the most accurate
                                                            // mode. The other modes are HiFi3, HiFi2, HiFi1 and LoFi. The
                                                            // difference between them is the number of bits used during computation.
/*
    std::vector<uint32_t> reader_compile_time_args;
    tt::tt_metal::TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index};
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> compute_kernel_args_group_1 = {
        num_tiles_per_core_group_1,  // per_core_block_cnt
        1                            // per_core_block_size
    };

    bool math_approx_mode = false;
    tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp",
        core_group_1,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args_group_1});

    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_kernel_args_group_2 = {
            num_tiles_per_core_group_2,  // per_core_block_cnt
            1                            // per_core_block_size
        };

        tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp",
            core_group_2,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_kernel_args_group_2});
    }
*/

    SetRuntimeArgs(program, reader, core, {colIdx_dram_buffer->address(),
                                             rowIdx_dram_buffer->address(),
                                              codeBook_dram_buffer->address(), 
                                              n_tiles_colIdx});
    SetRuntimeArgs(program, writer, core, {dst_dram_buffer->address(), n_tiles_colIdx});
    SetRuntimeArgs(program, compute, core, {n_tiles_colIdx, elements_per_tile_colIdx,
                                            n_tiles_rowIdx, elements_per_tile_rowIdx});

/*
    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        tt::tt_metal::SetRuntimeArgs(
            program, unary_reader_kernel_id, core, {src_buffer->address(), num_tiles_per_core, num_tiles_written});

        tt::tt_metal::SetRuntimeArgs(
            program, unary_writer_kernel_id, core, {dst_buffer->address(), num_tiles_per_core, num_tiles_written});
        num_tiles_written += num_tiles_per_core;
    }
*/

    return {
        std::move(program),
        {.reader = reader, .writer = writer}};
}







void ExampleDeviceOperation::SingleCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {

    auto& program = cached_program.program;
    auto& reader = cached_program.shared_variables.reader;
    auto& writer = cached_program.shared_variables.writer;

    // const auto& input_tensor = tensor_args.input_tensor;
    const auto& RowIdx_tensor = tensor_args.RowIdx_tensor;
    const auto& CodeBook_tensor = tensor_args.CodeBook_tensor;
    const auto& ColIdx_tensor = tensor_args.ColIdx_tensor;
    
    // auto& output_tensor = tensor_return_value;
    auto& output_tensor = tensor_return_value;

    // auto src_buffer = input_tensor.buffer();
    // auto dst_buffer = output_tensor.buffer();
    auto colIdx_dram_buffer = ColIdx_tensor.buffer();
    auto codeBook_dram_buffer = CodeBook_tensor.buffer();
    auto rowIdx_dram_buffer = RowIdx_tensor.buffer();
    auto dst_dram_buffer = output_tensor.buffer();

    {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, reader, CoreCoord{0, 0});
        runtime_args[0] = colIdx_dram_buffer->address();
        runtime_args[1] = rowIdx_dram_buffer->address();
        runtime_args[2] = codeBook_dram_buffer->address();
        runtime_args[3] = n_tiles_colIdx;
        // runtime_args[0] = src_buffer->address();
    }

    {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, writer, CoreCoord{0, 0});
        runtime_args[0] = dst_dram_buffer->address();
        runtime_args[1] = n_tiles_colIdx;
    }
}

}  // namespace ttnn::operations::examples
