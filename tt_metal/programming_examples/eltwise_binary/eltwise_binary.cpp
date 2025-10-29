// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string_view>
#include <vector>
#include "tt-metalium/base_types.hpp"

using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

inline float bfloat16_to_float(uint16_t bf16_val) {
    union {
        uint32_t u32;
        float f32;
    } u;
    u.u32 = static_cast<uint32_t>(bf16_val) << 16;
    return u.f32;
}

int main(int argc, char** argv) {
    bool pass = true;

    try {
        constexpr int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
        distributed::MeshWorkload workload;
        auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());
        Program program = CreateProgram();
        constexpr CoreCoord core = {0, 0};

        constexpr uint32_t n_tiles_colIdx = 4;   // 16 or 64
        constexpr uint32_t n_tiles_codebook = 1;
        constexpr uint32_t n_tiles_rowIdx = 1;    // 1 or 16
        constexpr uint32_t elements_per_tile_colIdx = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
        constexpr uint32_t elements_per_tile_rowIdx = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
        constexpr uint32_t elements_per_tile_codebook = 64;

        constexpr uint32_t tile_size_bytes_colIdx = sizeof(uint8_t) * elements_per_tile_colIdx;
        constexpr uint32_t tile_size_bytes_codebook = sizeof(bfloat16) * elements_per_tile_codebook;
        constexpr uint32_t tile_size_bytes_rowIdx = sizeof(uint8_t) * elements_per_tile_rowIdx;
        constexpr uint32_t tile_size_bytes_output = sizeof(bfloat16) * elements_per_tile_colIdx;

        distributed::DeviceLocalBufferConfig dram_config_colIdx{
            .page_size = tile_size_bytes_colIdx,
            .buffer_type = BufferType::DRAM};
        distributed::DeviceLocalBufferConfig dram_config_rowIdx{
            .page_size = tile_size_bytes_rowIdx, 
            .buffer_type = BufferType::DRAM};
        distributed::DeviceLocalBufferConfig dram_config_codebook{
            .page_size = tile_size_bytes_codebook,
            .buffer_type = BufferType::DRAM};
        distributed::DeviceLocalBufferConfig dram_config_output{
            .page_size = tile_size_bytes_output,
            .buffer_type = BufferType::DRAM};
        

        //setting to the largest one
        distributed::ReplicatedBufferConfig buffer_config{
            .size = n_tiles_colIdx * tile_size_bytes_colIdx // Total bytes per device (replicated across the mesh).
        };
        distributed::ReplicatedBufferConfig buffer_config_rowIdx{
            .size = n_tiles_rowIdx * tile_size_bytes_rowIdx // Total bytes per device (replicated across the mesh).
        };
        distributed::ReplicatedBufferConfig buffer_config_codebook{
            .size = n_tiles_codebook * tile_size_bytes_codebook // Total bytes per device (replicated across the mesh).
        };
        distributed::ReplicatedBufferConfig buffer_config_output{
            .size = n_tiles_colIdx * tile_size_bytes_output // Total bytes per device (replicated across the mesh).
        };

        auto colIdx_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config_colIdx, mesh_device.get());
        auto codeBook_dram_buffer = distributed::MeshBuffer::create(buffer_config_codebook, dram_config_codebook, mesh_device.get());
        auto rowIdx_dram_buffer = distributed::MeshBuffer::create(buffer_config_rowIdx, dram_config_rowIdx, mesh_device.get());
        auto dst_dram_buffer = distributed::MeshBuffer::create(buffer_config_output, dram_config_output, mesh_device.get());
        // Each handle represents a mesh-wide replicated buffer; on a unit mesh this is a single device allocation.
        
        // Initialize vectors with fixed (non-random) values

        // std::vector<uint8_t> colIdx_data(elements_per_tile_colIdx * n_tiles_colIdx);
        // for (size_t i = 0; i < colIdx_data.size(); ++i)
        //     colIdx_data[i] = static_cast<uint8_t>(10); //i % 16   // values 0–15 repeating

        // std::vector<bfloat16> codeBook_data(n_tiles_codebook * elements_per_tile_codebook);
        // for (size_t i = 0; i < codeBook_data.size(); ++i)
        //     codeBook_data[i] = static_cast<float>(5);   // values 0–63 repeating

        // std::vector<uint8_t> rowIdx_data(elements_per_tile_rowIdx * n_tiles_rowIdx);
        // for (size_t i = 0; i < rowIdx_data.size(); ++i)
        //     rowIdx_data[i] = static_cast<uint8_t>(1); //i % 4   // values 0–3 repeating
//-----------------------------------------------------------------------------------//
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dist_col(0, 15);

        std::vector<uint8_t> colIdx_data(elements_per_tile_colIdx * n_tiles_colIdx);
        for (auto& val : colIdx_data)
            val = static_cast<uint8_t>(dist_col(rng));
        
        std::vector<bfloat16> codeBook_data(n_tiles_codebook * elements_per_tile_codebook);
        for (size_t i = 0; i < codeBook_data.size(); ++i)
            codeBook_data[i] = static_cast<float>(i % 64);

        std::uniform_int_distribution<int> dist_row(0, 3);
        std::vector<uint8_t> rowIdx_data(elements_per_tile_rowIdx * n_tiles_rowIdx);
        for (auto& val : rowIdx_data)
            val = static_cast<uint8_t>(dist_row(rng));

        fmt::print("colIdx_data size (bytes): {}\n", colIdx_data.size() * sizeof(uint8_t));
        fmt::print("rowIdx_data size (bytes): {}\n", rowIdx_data.size() * sizeof(uint8_t));
        fmt::print("codeBook_data size (bytes): {}\n", codeBook_data.size() * sizeof(bfloat16));

        fmt::print("colIdx_data elements: {}\n", colIdx_data.size());
        fmt::print("rowIdx_data elements: {}\n", rowIdx_data.size());
        fmt::print("codeBook_data elements: {}\n", codeBook_data.size());

        fmt::print("codeBook first 16: ");
        for (int i = 0; i < 16 && i < (int)codeBook_data.size(); ++i)
            fmt::print("{:.2f} ", static_cast<float>(codeBook_data[i]));
        fmt::print("\n");

        fmt::print("colIdx first 16: ");
        for (int i = 0; i < 16 && i < (int)colIdx_data.size(); ++i)
            fmt::print("{} ", colIdx_data[i]);
        fmt::print("\n");

        fmt::print("rowIdx first 16: ");
        for (int i = 0; i < 16 && i < (int)rowIdx_data.size(); ++i)
            fmt::print("{}:{} ",i , rowIdx_data[i]);
        fmt::print("\n");

        // Upload host vectors into the mesh buffers.
        distributed::EnqueueWriteMeshBuffer(cq, colIdx_dram_buffer, colIdx_data, false);
        distributed::EnqueueWriteMeshBuffer(cq, rowIdx_dram_buffer, rowIdx_data, false);
        distributed::EnqueueWriteMeshBuffer(cq, codeBook_dram_buffer, codeBook_data, false);

        uint32_t tiles_per_cb = 2;
        tt::CBIndex colIdx_cb_index = tt::CBIndex::c_0;
        CreateCircularBuffer(program, core, CircularBufferConfig(
            /*total_size=*/tiles_per_cb * tile_size_bytes_colIdx,
            /*data_format_spec=*/{{colIdx_cb_index, tt::DataFormat::UInt8}})
            .set_page_size(colIdx_cb_index, tile_size_bytes_colIdx));

        tt::CBIndex rowIdx_cb_index = tt::CBIndex::c_1;
        CreateCircularBuffer(program, core, CircularBufferConfig(
            /*total_size=*/tiles_per_cb * tile_size_bytes_rowIdx,
            /*data_format_spec=*/{{rowIdx_cb_index, tt::DataFormat::UInt8}})
            .set_page_size(rowIdx_cb_index, tile_size_bytes_rowIdx));

        tt::CBIndex codeBook_cb_index = tt::CBIndex::c_2;
        CreateCircularBuffer(program, core, CircularBufferConfig(
            /*total_size=*/tiles_per_cb * tile_size_bytes_codebook,
            /*data_format_spec=*/{{codeBook_cb_index, tt::DataFormat::Float16}})
            .set_page_size(codeBook_cb_index, tile_size_bytes_codebook));

            tiles_per_cb = n_tiles_colIdx;
        tt::CBIndex dst_cb_index = tt::CBIndex::c_16;
        CreateCircularBuffer(program, core, CircularBufferConfig(
            /*total_size=*/tiles_per_cb * tile_size_bytes_output,
            /*data_format_spec=*/{{dst_cb_index, tt::DataFormat::Float16}})
            .set_page_size(dst_cb_index, tile_size_bytes_output));

        std::vector<uint32_t> reader_compile_time_args;
        TensorAccessorArgs(*colIdx_dram_buffer).append_to(reader_compile_time_args);
        TensorAccessorArgs(*rowIdx_dram_buffer).append_to(reader_compile_time_args);
        TensorAccessorArgs(*codeBook_dram_buffer).append_to(reader_compile_time_args);
        auto reader = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "eltwise_binary/kernels/dataflow/read_tiles.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_compile_time_args});
        std::vector<uint32_t> writer_compile_time_args;
        
        TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);
        auto writer = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "eltwise_binary/kernels/dataflow/write_tile.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = writer_compile_time_args});
        auto compute = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "eltwise_binary/kernels/compute/tiles_add.cpp",
            core,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4});
        
        SetRuntimeArgs(program, reader, core, {colIdx_dram_buffer->address(),
                                                 rowIdx_dram_buffer->address(),
                                                  codeBook_dram_buffer->address(), 
                                                  n_tiles_colIdx, tile_size_bytes_codebook});
        SetRuntimeArgs(program, writer, core, {dst_dram_buffer->address(), n_tiles_colIdx});
        SetRuntimeArgs(program, compute, core, {n_tiles_colIdx, elements_per_tile_colIdx,
                                                n_tiles_rowIdx, elements_per_tile_rowIdx});

        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);
        // Equivalently:
        // distributed::EnqueueMeshWorkload(cq, workload, true);

        // Read the output buffer (from shard at mesh coordinate {0,0} on a unit mesh) and validate.
        std::vector<bfloat16> result_vec;
        distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, true);

        fmt::print("\n\nresult_vec : ");
        for (int i = 0; i < 40 && i < (int)result_vec.size(); ++i)
            fmt::print("{:.0f} ", static_cast<float>(result_vec[i]));
        fmt::print("\n");
        for (int i = 1024; i < 1024+40 && i < (int)result_vec.size(); ++i)
            fmt::print("{:.0f} ", static_cast<float>(result_vec[i]));
        fmt::print("\n");
        for (int i = 2048; i < 2048+40 && i < (int)result_vec.size(); ++i)
            fmt::print("{:.0f} ", static_cast<float>(result_vec[i]));
        fmt::print("\n");
        for (int i = 3072; i < 3072+40 && i < (int)result_vec.size(); ++i)
            fmt::print("{:.0f} ", static_cast<float>(result_vec[i]));
        fmt::print("\n");

        constexpr float eps = 1e-2f; // loose tolerance because of the nature of bfloat16
        fmt::print("result_vec.size() : {}\n",result_vec.size());
        fmt::print("colIdx_data.size() : {}\n",colIdx_data.size());
        
        static int count = 0;
        static int count1 = 0;

        for(size_t idx_b = 0; idx_b < result_vec.size(); idx_b++) {
            int rowidx = (int)(rowIdx_data[(int)( idx_b / 16) ]);
            int colidx = (int)(colIdx_data[idx_b]);
            int codebook_index = (int)(colidx + (rowidx * 16.0f));

            // cout << " idx_b / 16 ="<< idx_b / 16 << " idxbCount=" << idxbCount << " sizeof(row_indices) "<< row_indices.size() << "\n";
            if(codebook_index >= codeBook_data.size()) {
                fmt::print(stderr, "codebook_index out of bounds\n");
                fmt::print(stderr, "codebook_index = {}, rowidx = {}, colidx = {}\n", (int)codebook_index, (int)rowidx, (int)colidx);
            }else {
                const bfloat16 expected = (codeBook_data[codebook_index]);
                const bfloat16 result = (bfloat16)(result_vec[idx_b]);
                if (std::abs(expected - result) > eps) {
                    pass = false;
                    // fmt::print(stderr, "colIdx_data {}, rowIdx_data {}\n", static_cast<float>(colIdx_data[idx_b]), static_cast<float>(rowIdx_data[idx_b / 16]));
                    if (count < 20)
                    {
                        fmt::print(stderr,
                                    "Result mismatch at index {}: expected {:.2f}, got {:.2f}\n",
                                    idx_b,
                                    static_cast<float>(expected),
                                    static_cast<float>(result)
                                );
                        fmt::print(stderr, "codebook_index = {}, rowidx = {}, colidx = {}\n", (int)codebook_index, (int)rowidx, (int)colidx);
                        // for(int j = codebook_index-1 ; j < codebook_index+1; j++) {
                        //     fmt::print(stderr, "codeBook_data[{}] = {}\n", j, int(codeBook_data[j]));
                        // }
                        count++;
                    }
                    
                    // fmt::print(".");
                }else{
                    if (count1 < 64){
                        // fmt::print(stderr, "Correct Result index {}:\t expected {}, got {}\n", idx_b, expected, actual);
                        // fmt::print(".");

                        count1++;
                    }
                }
            }
        }



        // Finally, we close the device.
        pass &= mesh_device->close();
    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception!\n");
        fmt::print(stderr, "{}\n", e.what());

        throw;
    }
    // clang-format on

    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        // TT_THROW("Test Failed");
        fmt::print("Test Failed\n");
    }

    return 0;
}