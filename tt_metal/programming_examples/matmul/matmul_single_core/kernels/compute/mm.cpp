// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
// #include <stdint.h>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "hostdevcommon/kernel_structs.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api.h"
#include "debug/dprint.h"
// #include <tt-metalium/bfloat16.hpp>
using std::uint32_t;

inline float bfloat16_to_float(uint16_t bf16_val) {
    union {
        uint32_t u32;
        float f32;
    } u;
    u.u32 = static_cast<uint32_t>(bf16_val) << 16;
    return u.f32;
}

// inline float bfloat16_to_float(uint16_t bf16_val) {
//     uint32_t temp = static_cast<uint32_t>(bf16_val) << 16;
//     return *(float*)&temp;
// }

namespace NAMESPACE {
/**
 * @brief Main kernel function for batched matrix multiplication (BMM).
 *
 * This function performs a blocked outer product matrix multiplication using tiles.
 * It initializes the matrix engine (FPU) and sets up circular buffers for input and output.
 * For each output tile (indexed by mt and nt), it:
 *   - Acquires the destination buffer.
 *   - Iterates over the K dimension (kt), waiting for input tiles to be available in the circular buffers.
 *   - Performs a tile-wise matrix multiplication using `matmul_tiles`.
 *   - Pops the used tiles from the input buffers.
 *   - After processing all K tiles, reserves space in the output buffer, packs the result tile, and pushes it to the
 * output buffer.
 *   - Releases the destination buffer.
 *
 * Compile-time arguments:
 *   - Mt: Number of output tile rows.
 *   - Kt: Number of tiles in the reduction dimension.
 *   - Nt: Number of output tile columns.
 *
 * Circular buffers:
 *   - cb_in0: Input buffer for matrix A tiles.
 *   - cb_in1: Input buffer for matrix B tiles.
 *   - cb_out: Output buffer for result tiles.
 *
 * Assumes that input tiles are provided in the correct order and that the reader is responsible for supplying
 * the appropriate tiles for each output tile computation.
 */static bool once = false;
void MAIN {
    
    const uint32_t Mt = get_compile_time_arg_val(0);
    const uint32_t Kt = get_compile_time_arg_val(1);
    const uint32_t Nt = get_compile_time_arg_val(2);
    const uint32_t codebook_num_elements = get_compile_time_arg_val(3);
    const uint32_t rowIdx_Num_tiles = get_compile_time_arg_val(4);
    const uint32_t ColIdx_Num_ele = get_compile_time_arg_val(5);

    constexpr tt::CBIndex cb_codebook = tt::CBIndex::c_0; // Codebook
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_rowidx = tt::CBIndex::c_2; // RowIndices
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    volatile uint16_t* cbCodebook_tile_ptr_f32    = nullptr;
    volatile uint16_t* cbIn1_tile_ptr_f32    = nullptr;
    volatile uint16_t* cbRowIdx_tile_ptr_f32    = nullptr;
    volatile uint16_t* cbOut_tile_ptr_f32    = nullptr;

    // mm_init(cb_in1, cb_in1, cb_out);
    // cb_wait_front(cb_out, 1);
    cb_get_tile(cb_out, 0 , &cbOut_tile_ptr_f32);
    

    cb_wait_front(cb_codebook, 1);
    cb_get_tile(cb_codebook, 0 , &cbCodebook_tile_ptr_f32);
    DPRINT << "CodeBook 8+64 elements (int16->f32): ";
    for (uint32_t i = 0; i < 8 + codebook_num_elements; i++) {
        float val = bfloat16_to_float(cbCodebook_tile_ptr_f32[i]);
        DPRINT << val << " ";
    }
    DPRINT << ENDL();
    cb_pop_front(cb_codebook, 1);


    for(uint32_t Row_tile_index = 0; Row_tile_index < rowIdx_Num_tiles; Row_tile_index++) {
        // Make sure registers can be used for the output tile. This also sets the registers to zero.
        tile_regs_acquire();
        
        cb_wait_front(cb_in1, 1);
        cb_wait_front(cb_rowidx, 1);

        // matmul_tiles(cb_in1, cb_in1, 0, 0, 0, false);

        cb_get_tile(cb_in1, Row_tile_index, &cbIn1_tile_ptr_f32);
        cb_get_tile(cb_rowidx, Row_tile_index, &cbRowIdx_tile_ptr_f32);

        DPRINT << "ColIdx 8+70 elements (int16->f32): ";
        for (uint32_t i = 0; i < 8+70; i++) {
            float val = bfloat16_to_float(cbIn1_tile_ptr_f32[i]);
            DPRINT << val << " ";
        }
        DPRINT << ENDL();

        DPRINT << "RowIdx 8+70 elements (int16->f32): ";
        for (uint32_t i = 0; i < 8+70; i++) {
            float val = bfloat16_to_float(cbRowIdx_tile_ptr_f32[i]);
            DPRINT << val << " ";
        }
        DPRINT << ENDL();

        // for(uint32_t ColIdx = 0; ColIdx < ColIdx_Num_ele; ColIdx++){
        for(uint32_t ColIdx = 1023; ColIdx < 1026/*ColIdx_Num_ele*/; ColIdx++){

            float rowidx = bfloat16_to_float(cbRowIdx_tile_ptr_f32[ 8 + (ColIdx / 16) ]);
            float colidx = bfloat16_to_float(cbIn1_tile_ptr_f32[8 + ColIdx]);
            uint8_t codebook_index = (uint8_t)(colidx + (rowidx * (float)16.0));
            
            DPRINT <<"Compute_codebook_index out of bounds" << "\n";
            DPRINT <<"Compute_codebook_index="<< (int)codebook_index 
                << " rowidx=" << (int)rowidx 
                << " colidx=" << (int)colidx << "\n";



            if(codebook_index >= codebook_num_elements) {
                // DPRINT <<"Compute_codebook_index out of bounds" << "\n";
                // DPRINT <<"Compute_codebook_index="<< (int)codebook_index 
                //         << " rowidx=" << (int)rowidx 
                //         << " colidx=" << (int)colidx << "\n";
            } else {
                // deQuant[idx_b] = codebook[codebook_index];
                cbOut_tile_ptr_f32[ColIdx] = cbCodebook_tile_ptr_f32[codebook_index + 8];
            }
        }
             
        // Mark the input tiles as used by popping them from the front of the circular buffers.
        cb_pop_front(cb_in1, 1);
        cb_pop_front(cb_rowidx, 1);

    }
    // Need to check this
    //cb_pop_front(cb_out, 1);





    tile_regs_commit();
    tile_regs_wait();

    // Ensure the output circular buffer has space for the result tile.
    cb_reserve_back(cb_out, 1);
    // Pack the result tile into the output circular buffer.
    pack_tile(0, cb_out);
    // Mark the output tile as ready so the writer can read it.
    cb_push_back(cb_out, 1);

    // We don't need the registers anymore, so we can release them and prepare for the next output tile.
    tile_regs_release();

    DPRINT << "Compute complete" << ENDL();


    // Setup the FPU (matrix engine) for the matmul operation. And specify the input
    // and output circular buffers.

    // mm_init(cb_in0, cb_in1, cb_out);

    // init_sfpu(cb_in1, cb_out);
    // the simplest possible version of outer product blocked matmul
    // the reader is expected to read the A's and B's tile rows and tile columns for each output tile
    



    // for (uint32_t mt = 0; mt < Mt; ++mt) {
    //     for (uint32_t nt = 0; nt < Nt; ++nt) {
    //         // Make sure registers can be used for the output tile. This also sets the registers to zero.
    //         tile_regs_acquire();
    //         for (uint32_t kt = 0; kt < Kt; kt++) {
    //             // Wait for the input tiles to be available in the input circular buffers.
    //             cb_wait_front(cb_in0, 1);
    //             cb_wait_front(cb_in1, 1);
    //             cb_wait_front(cb_rowidx, 1);

    //             uint32_t Row_tile_index = kt * Nt + nt;
    //             cb_get_tile(cb_in1, Row_tile_index, &cbIn1_tile_ptr_f32);
    //             cb_get_tile(cb_rowidx, Row_tile_index, &cbRowIdx_tile_ptr_f32);


    //             DPRINT << "ColIdx 8+70 elements (int16->f32): ";
    //             for (uint32_t i = 0; i < 8+70; i++) {
    //                 float val = bfloat16_to_float(cbIn1_tile_ptr_f32[i]);
    //                 DPRINT << val << " ";
    //             }
    //             DPRINT << ENDL();

    //             DPRINT << "RowIdx 8+70 elements (int16->f32): ";
    //             for (uint32_t i = 0; i < 8+70; i++) {
    //                 float val = bfloat16_to_float(cbRowIdx_tile_ptr_f32[i]);
    //                 DPRINT << val << " ";
    //             }
    //             DPRINT << ENDL();

    //           // Mark the input tiles as used by popping them from the front of the circular buffers.
    //             cb_pop_front(cb_in0, 1);
    //             cb_pop_front(cb_in1, 1);
    //         }

    //         // Commit and wait for the registers are populated with the results from the FPU
    //         tile_regs_commit();
    //         tile_regs_wait();

    //         // Ensure the output circular buffer has space for the result tile.
    //         cb_reserve_back(cb_out, 1);
    //         // Pack the result tile into the output circular buffer.
    //         pack_tile(0, cb_out);
    //         // Mark the output tile as ready so the writer can read it.
    //         cb_push_back(cb_out, 1);

    //         // We don't need the registers anymore, so we can release them and prepare for the next output tile.
    //         tile_regs_release();
    //     }
    // }
    





}
}  // namespace NAMESPACE