// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
// #include <tt-metalium/command_queue.hpp>
#include <tt-metalium/distributed.hpp>
#include <bmm_op.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "tt-metalium/core_coord.hpp"
#include <chrono>

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

// Reference implementation of matrix multiplication.
// Array A is of size MxK, Array B is of size KxN, and the output C is of size MxN.
// The implementation is bare bones and does not include optimizations such as tiling or vectorization.
// This is intended to be used as a golden reference for testing the Metalium implementation.
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

#define GROUP_SIZE 16

void golden_matmul(
    const std::vector<bfloat16>& a,
    const std::vector<bfloat16>& b,
    const std::vector<bfloat16>& row_indices,
    const std::vector<bfloat16>& codebook,
    std::vector<bfloat16>& output,
    uint32_t M,
    uint32_t N,
    uint32_t K) {
    std::vector<bfloat16> c_bf(M * N, 0);
    std::vector<bfloat16> deQuant(M * N, 0);    
    
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < K; j++) {
            std::uint32_t idx_b = j + (i * K);

            bfloat16 rowidx = static_cast<bfloat16>(row_indices[ idx_b / 16 ]);
            bfloat16 colidx = static_cast<bfloat16>(b[idx_b]);
            uint8_t codebook_index = (uint8_t)(colidx + (rowidx * (bfloat16)16.0));
            
            // cout << " idx_b / 16 ="<< idx_b / 16 << " idxbCount=" << idxbCount << " sizeof(row_indices) "<< row_indices.size() << "\n";
            if(codebook_index >= codebook.size()) {
                cout <<"codebook_index out of bounds" << "\n";
                cout <<"codebook_index="<< (int)codebook_index << " rowidx=" << (int)rowidx << " colidx=" << (int)colidx << "\n";
            }else {
                // deQuant[idx_b] = codebook[codebook_index];
                output[idx_b] = codebook[codebook_index];

            }
        }
    }

    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < N; j++) {
    //         std::uint32_t idx_c = j + (i * N);
    //         std::uint32_t idx_a = i * K;
    //         std::uint32_t idx_b = j;
    //         float c_f = 0;
    //         for (int k_m = 0; k_m < K; k_m++) {
    //             c_f += static_cast<float>(a[idx_a]) * static_cast<float>(b[idx_b]);
    //             idx_a += 1;
    //             idx_b += N;
    //         }
    //         output.at(idx_c) = bfloat16(c_f);
    //     }
    // }
}

// Matrix multiplication using the accelerator.
// Input a and b as well as output are vectors of bfloat16. But in the tiled layout.
// The input a is of size MxK, input b is of size KxN, and the output c is of size MxN.
// For this function, M, N and N must be divisible by TILE_HEIGHT and TILE_WIDTH respectively as that is the native unit
// of computation on the accelerator.
void matmul_single_core(
    const std::vector<bfloat16>& a,
    const std::vector<bfloat16>& b,
    const std::vector<bfloat16>& row_indices,
    const std::vector<bfloat16>& codebook,
    std::vector<bfloat16>& output,
    bool bcast_batch,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    size_t rowidx_num_elements,
    size_t codebook_num_elements) {
    

    // Set up mesh command queue, workload, device range, and program. This is a single-core example using core {0,0}.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program{};
    // Core range from x: [0, 0] to y: [0, 0] (single core at {0, 0})
    CoreCoord core({0, 0});

    // Calcaulate the number of tiles for each dimension.
    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;


    // Create DRAM buffers for the input and output data.
    uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
    uint32_t Num_tiles = (M * N) / (TILE_HEIGHT * TILE_WIDTH); //16
    // uint32_t single_tile_size = sizeof(bfloat16) * b.size();
    // uint32_t ColIdx_Num_ele = b.size();
    // uint32_t Num_tiles = 1; //16

    // uint32_t rowIdx_single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH / 16;
    uint32_t rowIdx_single_tile_size = sizeof(bfloat16) * (TILE_HEIGHT * TILE_WIDTH) / GROUP_SIZE;
    uint32_t rowIdx_Num_tiles = ((K / 16) * (N)) / ((TILE_HEIGHT * TILE_WIDTH) / GROUP_SIZE); //16
    // uint32_t rowIdx_single_tile_size = sizeof(bfloat16) * row_indices.size();
    // uint32_t rowIdx_Num_tiles = 1; //16

    uint32_t codebook_single_tile_size = sizeof(bfloat16) * codebook_num_elements;

    // We allocate DRAM buffers for the input and output data (replicated per device across the mesh).
    // Setting page_size to single_tile_size is the most common configuration for memory buffers in Metalium
    // as it is generic, works for most cases and achieves good performance.
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};

    distributed::DeviceLocalBufferConfig rowIdx_dram_config{
        .page_size = rowIdx_single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};
    
    distributed::DeviceLocalBufferConfig codebook_dram_config{
        .page_size = codebook_single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};
    
    // distributed::ReplicatedBufferConfig buffer_config_A{.size = sizeof(bfloat16) * a.size()};
    distributed::ReplicatedBufferConfig buffer_config_CK{.size = sizeof(bfloat16) * codebook.size()};
    distributed::ReplicatedBufferConfig buffer_config_B{.size = sizeof(bfloat16) * b.size()};
    distributed::ReplicatedBufferConfig buffer_config_C{.size = sizeof(bfloat16) * output.size()};
    distributed::ReplicatedBufferConfig buffer_config_R{.size = sizeof(bfloat16) * row_indices.size()};

    // auto src0_dram_buffer = distributed::MeshBuffer::create(buffer_config_A, dram_config, mesh_device.get());
    auto Codebook_dram_buffer = distributed::MeshBuffer::create(buffer_config_CK, codebook_dram_config, mesh_device.get());
    auto src1_dram_buffer = distributed::MeshBuffer::create(buffer_config_B, dram_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(buffer_config_C, dram_config, mesh_device.get());
    auto RowIdx_dram_buffer = distributed::MeshBuffer::create(buffer_config_R, rowIdx_dram_config, mesh_device.get());

   
    // Create circular buffers for the input and output data.
    // Using 2 tiles per circular buffer to allow for double buffering (data movement can be reading from one tile while
    // the compute kernel is using the other tile). This number can be adjusted based on the use case. But geberally
    // diminishing returns observed after several tiles.
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    
    uint32_t num_input_tiles = 2;
    uint32_t num_output_tiles = 2;
    uint32_t codebook_num_input_tiles = 1;
        
    // uint32_t src0_cb_index = CBIndex::c_0;
    // CircularBufferConfig cb_src0_config =
    //     CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
    //         .set_page_size(src0_cb_index, single_tile_size);
    // tt_metal::CreateCircularBuffer(program, core, cb_src0_config);
        
    uint32_t codebook_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_codebook_config =
        CircularBufferConfig(codebook_num_input_tiles * codebook_single_tile_size, {{codebook_cb_index, cb_data_format}})
            .set_page_size(codebook_cb_index, codebook_single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_codebook_config);

    uint32_t src1_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src1_config);
  
  
    uint32_t rowidx_cb_index = CBIndex::c_2;
    CircularBufferConfig cb_rowidx_config =
        CircularBufferConfig(num_input_tiles * rowIdx_single_tile_size, {{rowidx_cb_index, cb_data_format}})
            .set_page_size(rowidx_cb_index, rowIdx_single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_rowidx_config);



    uint32_t output_cb_index = tt::CBIndex::c_16;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, cb_data_format}})
            .set_page_size(output_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    // Create the data movement kernels and the compute kernel
    std::vector<uint32_t> reader_compile_time_args;
    // TensorAccessorArgs(*src0_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*Codebook_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*RowIdx_dram_buffer).append_to(reader_compile_time_args);
    
    
    auto reader_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "matmul/matmul_single_core/kernels/dataflow/reader_single_core_mm.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);
    auto writer_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "matmul/matmul_single_core/kernels/dataflow/writer_single_core_mm.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    // Compile time arguments for the kernels
    // Note that these take effect at the kernel's compile time. Chaning these values will require recompilation of the
    // kernel. Having arguments at compile time allows the compiler to optimize the kernel for the specific use case.
    // Like applying loop unrolling, constant folding, etc.. resulting in a more efficient kernel.
    std::vector<uint32_t> compute_compile_time_args = {
        Mt,  // Mt
        Kt,  // Kt
        Nt,  // Nt
        codebook_num_elements,
        rowIdx_Num_tiles,
        ColIdx_Num_ele
        // Num_tiles,
         
    };
    tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "matmul/matmul_single_core/kernels/compute/mm.cpp",
        core,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_compile_time_args});

    // Set kernel arguments
    // uint32_t src0_addr = src0_dram_buffer->address();
    uint32_t codebook_addr = Codebook_dram_buffer->address();
    uint32_t ColIdx_addr = src1_dram_buffer->address();
    uint32_t dst_addr = dst_dram_buffer->address();
    uint32_t rowIdx_addr = RowIdx_dram_buffer->address();

    // tt_metal::SetRuntimeArgs(program, reader_id, core, {src0_addr, src1_addr, Mt, Kt, Nt});
   
    // tt_metal::SetRuntimeArgs(program, reader_id, core,
    // { src0_addr, src1_addr, rowIdx_addr, codebook_addr, Mt, Kt, Nt,
    //   /*rowidx_size_bytes=*/ sizeof(bfloat16) * rowidx_num_elements,
    //   /*codebook_size_bytes=*/ sizeof(bfloat16) * codebook_num_elements });
    
    tt_metal::SetRuntimeArgs(program, reader_id, core,
    { codebook_addr, ColIdx_addr, rowIdx_addr, Mt, Kt, Nt, Num_tiles, rowIdx_Num_tiles });


    tt_metal::SetRuntimeArgs(program, writer_id, core, {dst_addr, rowIdx_Num_tiles, Kt, Nt});
    // NOTE: Note that we never set the runtime arguments for the compute kernel. This is because everything needed has
    // been set at compile time. The compute kernel does not need any runtime arguments to execute. And so we can skip
    // this step.

    // Upload the input data to the DRAM buffers, execute the kernels, wait for the result to be read into the output
    // buffer
    // distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, a, false);
    distributed::EnqueueWriteMeshBuffer(cq, Codebook_dram_buffer, codebook, false);
    distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, b, false);
    distributed::EnqueueWriteMeshBuffer(cq, RowIdx_dram_buffer, row_indices, false);

    workload.add_program(device_range, std::move(program));
    
    auto start_device_time = std::chrono::high_resolution_clock::now();
    
    distributed::EnqueueMeshWorkload(cq, workload, false);

    distributed::EnqueueReadMeshBuffer(cq, output, dst_dram_buffer, true);
    
    auto end_device_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> device_duration_ms = end_device_time - start_device_time;
    fmt::print("Device execution time: {:.3f} ms\n", device_duration_ms.count());

}

///////////////////////////////////////

int main() {
    bool pass = true;

    try {
        // Open device
        constexpr int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

                // parameters for the matrix multiplication
        constexpr uint32_t M = 128;  // user-defined
        constexpr uint32_t N = 128;  // user-defined
        constexpr uint32_t K = 128;  // user-defined

        static_assert(M % TILE_HEIGHT == 0, "M must be divisible by TILE_HEIGHT");
        static_assert(N % TILE_WIDTH == 0, "N must be divisible by TILE_WIDTH");
        static_assert(K % TILE_WIDTH == 0, "K must be divisible by TILE_WIDTH");

        // std::mt19937 rng(42); // Fixed seed for reproducibility
        // std::uniform_real_distribution<float> dist(0.f, 1.0f);

        std::vector<bfloat16> activations(M * K);
        std::vector<bfloat16> centroidIndices(K * N);
        std::vector<bfloat16> codebook(64);
        std::vector<bfloat16> RowIndices((K / 16) * (N ));// 16 = group size; 8*16
        size_t rowidx_num_elements = RowIndices.size();
        size_t codebook_num_elements = codebook.size();


        for (size_t i = 0; i < activations.size(); ++i)
            activations[i] = bfloat16(static_cast<float>(i));

        for (size_t i = 0; i < centroidIndices.size(); ++i)
            centroidIndices[i] = bfloat16(static_cast<float>((i % 16))); 

        for (size_t i = 0; i < codebook.size(); ++i)
            codebook[i] = bfloat16(static_cast<float>(10));  

        for (size_t i = 0; i < RowIndices.size(); ++i)
            RowIndices[i] = bfloat16(static_cast<float>(i % 4)); 



        // for (size_t i = 0; i < activations.size(); ++i)
        //     activations[i] = bfloat16(static_cast<float>(i));

        // for (size_t i = 0; i < centroidIndices.size(); ++i)
        //     centroidIndices[i] = bfloat16(static_cast<float>((i ))); 

        // for (size_t i = 0; i < codebook.size(); ++i)
        //     codebook[i] = bfloat16(static_cast<float>(10));  

        // for (size_t i = 0; i < RowIndices.size(); ++i)
        //     RowIndices[i] = bfloat16(static_cast<float>(i )); 




        // === Print first 96 elements of each ===
        // fmt::print("First 96 elements of activations: ");
        // for (int i = 0; i < 96; i++) fmt::print("{:.4f} ", float(activations[i]));
        // fmt::print("\n");

        // fmt::print("First 5 elements of centroidIndices: ");
        // for (int i = 0; i < 5; i++) fmt::print("{:.4f} ", float(centroidIndices[i]));
        // fmt::print("\n");

        // fmt::print("First 5 elements of RowIndices: ");
        // for (int i = 0; i < 5; i++) fmt::print("{:.4f} ", float(RowIndices[i]));
        // fmt::print("\n");

        // fmt::print("First 10 elements of Codebook: ");
        // for (int i = 0; i < 10; i++) fmt::print("{:.4f} ", float(codebook[i]));
        // fmt::print("\n");


        std::vector<bfloat16> golden_vec(M * N, 0);
        golden_matmul(activations, centroidIndices, RowIndices, 
                            codebook, golden_vec, M, N, K);

        // Tilize the input vectors to match the expected tiled layout for the device
        // The Tenstorrent hardware operates on data in 32x32 tiles rather than standard row-major format.
        // tilize_nfaces() converts the input matrices from row-major layout to the tiled layout expected by the device.
        // This transformation groups elements into 32x32 blocks and reorders them in memory so that each tile
        // (32x32 elements) is stored contiguously. This matches the native data access patterns of the matrix engine
        // and enables efficient operations on the accelerator.
        
        // activations = tilize_nfaces(activations, M, K);
        // centroidIndices = tilize_nfaces(centroidIndices, K, N);
        
        // RowIndices = tilize_nfaces(RowIndices, K / 16, N / 16);
        // codebook typically small; tilization depends on how reader expects it. If reader expects
        // linear codebook in DRAM then you can keep it as-is. If it expects tiled layout, tilize it:
        // codebook = tilize_nfaces(codebook, codebook_rows, codebook_cols);
        // In this example we leave codebook linear and pass the buffer address; adjust if needed for your reader.

        std::vector<bfloat16> result_vec(M * N, 0);
        auto start_time = std::chrono::high_resolution_clock::now();
        matmul_single_core(activations, centroidIndices, RowIndices, codebook,
                             result_vec, false, M, N, K, mesh_device,
                              rowidx_num_elements, codebook_num_elements);
        auto end_time = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> duration_ms = end_time - start_time;
        fmt::print("matmul_single_core execution time: {:.3f} ms\n", duration_ms.count());

        // Reverse the tilization to get the result in the row-major format that the CPU expects
        // result_vec = untilize_nfaces(result_vec, M, N);

        fmt::print("Output vector of size {}\n", result_vec.size());
        fmt::print("First 10 elements of Output: ");
        for (int i = 0; i < 10; i++) fmt::print("{:.2f} ", float(result_vec[i]));
        fmt::print("\n");
        fmt::print("First 10 elements of GOLDEN Output: ");
        for (int i = 0; i < 10; i++) fmt::print("{:.2f} ", float(golden_vec[i]));
        fmt::print("\n");
       
        // Calculate the Pearson correlation coefficient (PCC) between the golden vector and the result vector
        // This is a measure of how similar the two vectors are.
        // A PCC close to 1 indicates that the two vectors are very similar.
        float pearson = check_bfloat16_vector_pcc(golden_vec, result_vec);
        fmt::print("Metalium vs Golden -- PCC = {}\n", pearson);
        TT_ASSERT(pearson > 0.97, "PCC not high enough. Result PCC: {}, Expected PCC: 0.97", pearson);

        pass &= mesh_device->close();

    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception!\n");
        fmt::print(stderr, "{}\n", e.what());

        throw;
    }

    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}