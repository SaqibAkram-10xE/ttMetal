// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

inline float bfloat16_to_float(uint16_t bf16_val) {
    uint32_t temp = static_cast<uint32_t>(bf16_val) << 16;  // upper bits
    float result;
    std::memcpy(&result, &temp, sizeof(result));
    return result;
}

void kernel_main() {

    // DPRINT << "I am at the start of kernel" << ENDL();

    // Expected runtime arg order (host must match):
    // 0: src0_addr
    // 1: src1_addr
    // 2: rowidx_addr
    // 3: codebook_addr
    // 4: Mt
    // 5: Kt
    // 6: Nt
    // 7: rowidx_size_bytes   <-- NEW (total bytes of row_indices buffer)
    // 8: codebook_size_bytes <-- NEW (total bytes of codebook buffer)
    
    // uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t codebook_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t rowidx_addr = get_arg_val<uint32_t>(2);
    uint32_t Mt        = get_arg_val<uint32_t>(3);
    uint32_t Kt        = get_arg_val<uint32_t>(4);
    uint32_t Nt        = get_arg_val<uint32_t>(5);
    uint32_t Num_tiles = get_arg_val<uint32_t>(6);
    uint32_t rowIdx_Num_tiles = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_id_codeBook = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t cb_id_rowidx = 2;
    // constexpr uint32_t cb_id_codebook = 3;

    // Get tile sizes for these CB ids (as configured on host)
    uint32_t tile_size_codebook = sizeof(bfloat16) * 64;//get_tile_size(cb_id_codeBook);         // typically 32*32*2 = 2048
    uint32_t tile_size_in1 = ; //get_tile_size(cb_id_in1);         // typically 2048
    uint32_t tile_size_rowidx = get_tile_size(cb_id_rowidx);   // host configured page size for rowidx CB
    // uint32_t tile_size_codebook = get_tile_size(cb_id_codebook);// host configured page size for codebook CB

    // === Tensor accessors (interleaved layout) ===
    constexpr auto s0_args = TensorAccessorArgs<0>();
    const auto s0 = TensorAccessor(s0_args, codebook_addr, tile_size_codebook);

    constexpr auto s1_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>();
    const auto s1 = TensorAccessor(s1_args, src1_addr, tile_size_in1);

    constexpr auto s2_args = TensorAccessorArgs<s1_args.next_compile_time_args_offset()>();
    const auto s2 = TensorAccessor(s2_args, rowidx_addr, tile_size_rowidx);

    // constexpr auto s3_args = TensorAccessorArgs<s2_args.next_compile_time_args_offset()>();
    // const auto s3 = TensorAccessor(s3_args, codebook_addr, tile_size_codebook);

    // DPRINT << "I am at mid" << ENDL();
    // DPRINT_DATA0(DPRINT << "Hello, host, I am running a void data movement kernel on Data Movement core 0." << ENDL());
    // DPRINT_DATA1(DPRINT << "Hello, host, I am running a void data movement kernel on Data Movement core 1." << ENDL());
    
   
    
    for(uint32_t RIt = 0; RIt < rowIdx_Num_tiles; RIt++) {
        
        cb_reserve_back(cb_id_in1, 1);
        uint32_t l1_write_addr_colIdx = get_write_ptr(cb_id_in1);
        noc_async_read_tile(RIt, s1, l1_write_addr_colIdx);   // Only one tile or chunk
        noc_async_read_barrier();
        const uint16_t* ptr16 = reinterpret_cast<const uint16_t*>(l1_write_addr_colIdx);
        DPRINT << "l1_write_addr_colIdx from 1020 to 1028 elements (bf16->f32): ";
        for (uint32_t i = 1020; i < 1028; i++) {
            float val = bfloat16_to_float(ptr16[i]);
            DPRINT << val << " ";
        }
        DPRINT << ENDL();
        cb_push_back(cb_id_in1, 1);
        
        cb_reserve_back(cb_id_rowidx, 1);
        uint32_t l1_write_addr_rowIdx = get_write_ptr(cb_id_rowidx);
        noc_async_read_tile(RIt, s2, l1_write_addr_rowIdx);   // Only one tile or chunk
        noc_async_read_barrier();
        const uint16_t* ptr16R = reinterpret_cast<const uint16_t*>(l1_write_addr_rowIdx);
        DPRINT << "l1_write_addr_rowIdx first 20 elements (bf16->f32): ";
        for (uint32_t i = 0; i < 20; i++) {
            float val = bfloat16_to_float(ptr16R[i]);
            DPRINT << val << " ";
        }
        DPRINT << ENDL();
        cb_push_back(cb_id_rowidx, 1);
        DPRINT << "RIt: " << RIt << ENDL();
    }
    DPRINT << "col_Idx matrix loaded into L1" << ENDL();

    DPRINT << "RowIndex table loaded into L1" << ENDL();

     //cb_id_codeBook
    {
        cb_reserve_back(cb_id_codeBook, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id_codeBook);
        noc_async_read_tile(0, s0, l1_write_addr);   // Only one tile or chunk
        noc_async_read_barrier();
       
        const uint16_t* ptr16 = reinterpret_cast<const uint16_t*>(l1_write_addr);
        DPRINT << "Codebook first 20 elements (bf16->f32): ";
        for (uint32_t i = 0; i < 20; i++) {
            float val = bfloat16_to_float(ptr16[i]);
            DPRINT << val << " ";
        }
        DPRINT << ENDL();
        cb_push_back(cb_id_codeBook, 1);
        DPRINT << "Codebook loaded into L1" << ENDL();
    }
    

    
    // Loop through the dimensions of the matrices. Read them and push to the circular buffers.
    // Dimension names are called M, N and K. `t` in `mt` means tile.
    /*
    for (uint32_t mt = 0; mt < Mt; mt++) {
        uint32_t itileB = 0;
        for (uint32_t nt = 0; nt < Nt; nt++) {
            for (uint32_t kt = 0; kt < Kt; kt++) {
                {                                          // Read A's tile at (mt, kt)
                    uint32_t a_tile_index = mt * Kt + kt;  // A is MK, so we stride by Kt
                    cb_reserve_back(cb_id_in0, 1);
                    uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
                    noc_async_read_tile(a_tile_index, s0, l1_write_addr_in0);
                    noc_async_read_barrier();
                    cb_push_back(cb_id_in0, 1);
                }

                {                                          // Read B's tile at (kt, nt)
                    uint32_t b_tile_index = kt * Nt + nt;  // B is KN, so we stride by Nt
                    cb_reserve_back(cb_id_in1, 1);
                    uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
                    noc_async_read_tile(b_tile_index, s1, l1_write_addr_in1);
                    noc_async_read_barrier();
                    cb_push_back(cb_id_in1, 1);
                }

                {    
                    uint32_t Row_tile_index = kt * Nt + nt;
                    cb_reserve_back(cb_id_rowidx, 1);
                    uint32_t l1_write_addr = get_write_ptr(cb_id_rowidx);
                    noc_async_read_tile(Row_tile_index, s2, l1_write_addr);   // Only one tile or chunk
                    noc_async_read_barrier();
                    cb_push_back(cb_id_rowidx, 1);
                }

            }  // Kt loop
        }  // Nt loop
    }  // Mt loop
*/

    DPRINT << "I am at end of data_flow kernel" << ENDL();
}