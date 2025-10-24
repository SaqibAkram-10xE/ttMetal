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
int main(int argc, char** argv) {
    bool pass = true;

    // clang-format off
    try {
        // Create a 1x1 mesh on device 0. The same API scales to multi-device meshes.
        constexpr int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        // Submit work via a mesh command queue: data uploads/downloads and program execution.
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
        // A program is a collection of kernels. Note that unlike OpenCL/CUDA where every core must run the
        // same kernel at a given time. Metalium allows you to run different kernels on different cores
        // simultaneously.
        distributed::MeshWorkload workload;
        // Execute across this device range. Here it spans the whole mesh (1x1).
        auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());
        Program program = CreateProgram();

        // This example program will only use 1 Tensix core. So we set the core to {0, 0}.
        constexpr CoreCoord core = {0, 0};

        // Define some constants that will be used throughout the program.
        // * Processing 64 tiles
        // * Each tile is 32x32 elements
        // * Each element is a bfloat16 (2 bytes)
        constexpr uint32_t n_tiles_colIdx = 12;   // 16 or 64
        constexpr uint32_t n_tiles_codebook = 1;
        constexpr uint32_t n_tiles_rowIdx = 1;    // 1 or 16
        constexpr uint32_t elements_per_tile_colIdx = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
        constexpr uint32_t elements_per_tile_rowIdx = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
        constexpr uint32_t elements_per_tile_codebook = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;//64;

        constexpr uint32_t tile_size_bytes_colIdx = sizeof(bfloat16) * elements_per_tile_colIdx;
        constexpr uint32_t tile_size_bytes_codebook = sizeof(bfloat16) * elements_per_tile_codebook;
        constexpr uint32_t tile_size_bytes_rowIdx = sizeof(bfloat16) * elements_per_tile_rowIdx;

        // Create 3 DRAM-backed mesh buffers: two inputs (src0, src1) and one output (dst).
        distributed::DeviceLocalBufferConfig dram_config_colIdx{
            .page_size = tile_size_bytes_colIdx, //The page size of the buffer in bytes. Unlike the `loopback` example, we
                                          // need the page size to be the same as the tile size for a large portion of the NoC transfer APIs to work.
            .buffer_type = BufferType::DRAM}; // This is a DRAM buffer.
        distributed::DeviceLocalBufferConfig dram_config_rowIdx{
            .page_size = tile_size_bytes_rowIdx, 
            .buffer_type = BufferType::DRAM};
        distributed::DeviceLocalBufferConfig dram_config_codebook{
            .page_size = tile_size_bytes_codebook,
            .buffer_type = BufferType::DRAM};
        // distributed::DeviceLocalBufferConfig dram_config{
        //     .page_size = tile_size_bytes_colIdx,
        //     .buffer_type = BufferType::DRAM};
        

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


        auto colIdx_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config_colIdx, mesh_device.get());
        auto codeBook_dram_buffer = distributed::MeshBuffer::create(buffer_config_codebook, dram_config_codebook, mesh_device.get());
        auto rowIdx_dram_buffer = distributed::MeshBuffer::create(buffer_config_rowIdx, dram_config_rowIdx, mesh_device.get());
        auto dst_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config_colIdx, mesh_device.get());
        // Each handle represents a mesh-wide replicated buffer; on a unit mesh this is a single device allocation.

        // // Initialize the input buffers with random data. For this example, src0 is a random vector of bfloat16 values
        // std::vector<bfloat16> colIdx_data(elements_per_tile_colIdx * n_tiles_colIdx);
        // for (size_t i = 0; i < colIdx_data.size(); ++i)
        //     colIdx_data[i] = bfloat16(static_cast<float>(15));
        
        //  std::vector<bfloat16> rowIdx_data(elements_per_tile_rowIdx * n_tiles_rowIdx);//, bfloat16(val_to_add));
        // for (size_t i = 0; i < rowIdx_data.size(); ++i)
        //     rowIdx_data[i] = bfloat16(static_cast<float>(3));

        // std::vector<bfloat16> codeBook_data(n_tiles_codebook * elements_per_tile_codebook);
        // for (size_t i = 0; i < codeBook_data.size(); ++i)
        //     codeBook_data[i] = bfloat16(static_cast<float>(6));


        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dist_col(0, 15);
        std::vector<bfloat16> colIdx_data(elements_per_tile_colIdx * n_tiles_colIdx);
        for (auto& val : colIdx_data)
            val = bfloat16(static_cast<float>(dist_col(rng)));

        std::vector<bfloat16> codeBook_data(n_tiles_codebook * elements_per_tile_codebook);
        for (size_t i = 0; i < codeBook_data.size(); ++i)
            codeBook_data[i] = bfloat16(static_cast<float>(i % 64));  

        std::uniform_int_distribution<int> dist_row(0, 3);
        std::vector<bfloat16> rowIdx_data(elements_per_tile_rowIdx * n_tiles_rowIdx);
        for (auto& val : rowIdx_data)
            val = bfloat16(static_cast<float>(dist_row(rng)));



        fmt::print("codeBook first 16: ");
        for (int i = 0; i < 16 && i < (int)codeBook_data.size(); ++i)
            fmt::print("{} ", float(codeBook_data[i]));
        fmt::print("\n");

        fmt::print("colIdx first 16: ");
        for (int i = 0; i < 16 && i < (int)colIdx_data.size(); ++i)
            fmt::print("{} ", float(colIdx_data[i]));
        fmt::print("\n");

        fmt::print("rowIdx first 16: ");
        for (int i = 0; i < 16 && i < (int)rowIdx_data.size(); ++i)
            fmt::print("{} ", float(rowIdx_data[i]));
        fmt::print("\n");




        // fmt::print("First 96 elements of colIdx_data: ");
        // for (int i = 0; i < 96; i++) fmt::print("{:.1f} ", float(colIdx_data[i]));
        // fmt::print("\n");

        // Upload host vectors into the mesh buffers.
        distributed::EnqueueWriteMeshBuffer(cq, colIdx_dram_buffer, colIdx_data, false);
        distributed::EnqueueWriteMeshBuffer(cq, rowIdx_dram_buffer, rowIdx_data, false);
        distributed::EnqueueWriteMeshBuffer(cq, codeBook_dram_buffer, codeBook_data, false);

        // Create 3 circular buffers. Think them like pipes moving data from one core to another. cb_src0 and cb_src1 are used to
        // move data from the reader kernel to the compute kernel. cb_dst is used to move data from the compute kernel to the writer
        // kernel. Each circular buffer is made up of 2 tiles. Thus when one tile is pushed and being used by the receiving end, the
        // sending end can get the next piece of data ready to be pushed. Overlapping the operations. Leading to better performance.
        // However there is a trade off, The more tiles in a circular buffer, the more memory is used. And Circular buffers are
        // backed by L1(SRAM) memory and L1 is a precious resource.
        // The hardware supports up to 32 circular buffers and they all act the same.
        constexpr uint32_t tiles_per_cb = 2;
        tt::CBIndex colIdx_cb_index = tt::CBIndex::c_0;
        CreateCircularBuffer(program, core, CircularBufferConfig(
            /*total_size=*/tiles_per_cb * tile_size_bytes_colIdx,                    // The total size of the circular buffer in bytes
            /*data_format_spec=*/{{colIdx_cb_index, tt::DataFormat::Float16_b}})// The circular buffer index and data format it'll hold
            .set_page_size(colIdx_cb_index, tile_size_bytes_colIdx));                  // Since we will be sending one tile at a time, we set
                                                                              // the page size to the tile size (and thus
                                                                              // total_size / page_size = tiles_per is the number of
                                                                              // entries in the circular buffer)
        tt::CBIndex rowIdx_cb_index = tt::CBIndex::c_1;
        CreateCircularBuffer(program, core, CircularBufferConfig(
            /*total_size=*/tiles_per_cb * tile_size_bytes_rowIdx,
            /*data_format_spec=*/{{rowIdx_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(rowIdx_cb_index, tile_size_bytes_rowIdx));

        tt::CBIndex codeBook_cb_index = tt::CBIndex::c_2;
        CreateCircularBuffer(program, core, CircularBufferConfig(
            /*total_size=*/tiles_per_cb * tile_size_bytes_codebook,
            /*data_format_spec=*/{{codeBook_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(codeBook_cb_index, tile_size_bytes_codebook));

        tt::CBIndex dst_cb_index = tt::CBIndex::c_16;
        CreateCircularBuffer(program, core, CircularBufferConfig(
            /*total_size=*/tiles_per_cb * tile_size_bytes_colIdx,
            /*data_format_spec=*/{{dst_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(dst_cb_index, tile_size_bytes_colIdx));

        // Create the reader, writer and compute kernels. The kernels do the following:
        // * Reader: Reads data from the DRAM buffer and pushes it into the circular buffer.
        // * Compute: Waits for data to be available in the circular buffer, pops it, adds the two inputs together and pushes the result
        //   into the output circular buffer.
        // * Writer: Waits for data to be available in the output circular buffer, pops it and writes it back into DRAM.
        // These kernels work together to form a pipeline. The reader reads data from the DRAM buffer and makes them available in the
        // compute kernel. The compute kernel does math and pushes the result into the writer kernel. The writer kernel writes the result
        // back to DRAM.
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
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4});   // There's different math fidelity modes (for the tensor engine)
                                                                // that trade off performance for accuracy. HiFi4 is the most accurate
                                                                // mode. The other modes are HiFi3, HiFi2, HiFi1 and LoFi. The
                                                                // difference between them is the number of bits used during computation.

        // Set the runtime arguments for the kernels. This also registers
        // the kernels with the program.
        SetRuntimeArgs(program, reader, core, {colIdx_dram_buffer->address(),
                                                 rowIdx_dram_buffer->address(),
                                                  codeBook_dram_buffer->address(), 
                                                  n_tiles_colIdx});
        SetRuntimeArgs(program, writer, core, {dst_dram_buffer->address(), n_tiles_colIdx});
        SetRuntimeArgs(program, compute, core, {n_tiles_colIdx, elements_per_tile_colIdx,
                                                n_tiles_rowIdx, elements_per_tile_rowIdx});

        // We have setup the program. Now we queue the kernel for execution. The final argument is set to false. This indicates
        // to Metalium that the operation is non-blocking. The function is allowed to return upon the kernel being queued. We must
        // ensure that the kernel is finished before we read the output buffer. This is done by calling distributed::Finish(cq) which waits until
        // all operations in the command queue are finished. This is equivalent to calling EnqueueMeshWorkload(cq, program, true); telling
        // Metalium to wait until the program is finished before returning.
        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);
        // Equivalently:
        // distributed::EnqueueMeshWorkload(cq, workload, true);

        // Read the output buffer (from shard at mesh coordinate {0,0} on a unit mesh) and validate.
        std::vector<bfloat16> result_vec;
        distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, true);

        constexpr float eps = 1e-2f; // loose tolerance because of the nature of bfloat16
        TT_FATAL(result_vec.size() == colIdx_data.size(), "Result vector size mismatch");
        
        // for (size_t i = 0; i < result_vec.size(); ++i) {
        //     const float expected = static_cast<float>(colIdx_data[i]) + val_to_add;
        //     // const float expected = static_cast<float>(a_data[i]) + b_data[i];
        //     const float actual = static_cast<float>(result_vec[i]);

        //     if (std::abs(expected - actual) > eps) {
        //         pass = false;
        //         fmt::print(stderr, "colIdx_data {}\n", static_cast<float>(colIdx_data[i]));
        //         fmt::print(stderr, "Result mismatch at index {}: expected {}, got {}\n", i, expected, actual);
        //     }
        // }

        static int count = 0;
        static int count1 = 0;


        // // Calculate the Pearson correlation coefficient (PCC) between the golden vector and the result vector
        // // This is a measure of how similar the two vectors are.
        // // A PCC close to 1 indicates that the two vectors are very similar.
        // float pearson = check_bfloat16_vector_pcc(golden_vec, result_vec);
        // fmt::print("Metalium vs Golden -- PCC = {}\n", pearson);
        // TT_ASSERT(pearson > 0.97, "PCC not high enough. Result PCC: {}, Expected PCC: 0.97", pearson);



        for(size_t idx_b = 0; idx_b < result_vec.size(); idx_b++) {
            float rowidx = float(rowIdx_data[ idx_b / 16 ]);
            float colidx = float(colIdx_data[idx_b]);
            uint8_t codebook_index = (uint8_t)(colidx + (rowidx * 16.0f));

            // cout << " idx_b / 16 ="<< idx_b / 16 << " idxbCount=" << idxbCount << " sizeof(row_indices) "<< row_indices.size() << "\n";
            if(codebook_index >= codeBook_data.size()) {
                fmt::print(stderr, "codebook_index out of bounds\n");
                fmt::print(stderr, "codebook_index = {}, rowidx = {}, colidx = {}\n", (int)codebook_index, (int)rowidx, (int)colidx);
            }else {
                const float expected = static_cast<float>(codeBook_data[codebook_index]);
                const float actual = static_cast<float>(result_vec[idx_b]);
                if (std::abs(expected - actual) > eps) {
                    pass = false;
                    // fmt::print(stderr, "colIdx_data {}, rowIdx_data {}\n", static_cast<float>(colIdx_data[idx_b]), static_cast<float>(rowIdx_data[idx_b / 16]));
                    if (count < 10)
                    {
                        fmt::print(stderr, "Result mismatch at index {}: expected {}, got {}\n", idx_b, expected, actual);
                        fmt::print(stderr, "codebook_index = {}, rowidx = {}, colidx = {}\n", (int)codebook_index, (int)rowidx, (int)colidx);
                        for(int j = codebook_index-1 ; j < codebook_index+1; j++) {
                            fmt::print(stderr, "codeBook_data[{}] = {:.1f}\n", j, static_cast<float>(codeBook_data[j]));
                        }
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