// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api.h"

inline float bfloat16_to_float(uint16_t bf16_val) {
    union {
        uint32_t u32;
        float f32;
    } u;
    u.u32 = static_cast<uint32_t>(bf16_val) << 16;
    return u.f32;
}

inline uint16_t float_to_bfloat16(float fval) {
    union {
        uint32_t u32;
        float f32;
    } u;
    u.f32 = fval;
    // Round-to-nearest-even (optional, for accuracy)
    uint32_t rounding_bias = ((u.u32 >> 16) & 1) + 0x7FFF;
    return static_cast<uint16_t>((u.u32 + rounding_bias) >> 16);
}


static int count = 0;

namespace NAMESPACE {
void MAIN {

    uint32_t n_tiles_colIdx = get_arg_val<uint32_t>(0);
    uint32_t elements_per_tile_colIdx = get_arg_val<uint32_t>(1);
    uint32_t n_tiles_rowIdx = get_arg_val<uint32_t>(2);
    uint32_t elements_per_tile_row = get_arg_val<uint32_t>(3);
    
    constexpr auto cb_colIdx = tt::CBIndex::c_0;
    constexpr auto cb_rowIdx = tt::CBIndex::c_1;
    constexpr auto cb_codeBook = tt::CBIndex::c_2;
    constexpr auto cb_scale = tt::CBIndex::c_3;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    
    volatile uint8_t* colIdx_tile_ptr = nullptr;
    volatile uint8_t* rowIdx_tile_ptr = nullptr;
    volatile uint8_t* scale_tile_ptr = nullptr;
    volatile uint16_t* codeBook_tile_ptr = nullptr;
    volatile uint16_t* out_tile_ptr = nullptr;
 
    uint32_t current_row_tile_id = 0;
    int codebook_index = 0;
    int rowidx = 0;
    int colidx = 0;
    float scale_value = 0.0f;
    float codebook_value = 0.0f;
    
    cb_wait_front(cb_codeBook, 1);
    cb_get_tile(cb_codeBook, 0 , &codeBook_tile_ptr);
    codeBook_tile_ptr = &codeBook_tile_ptr[8]; // in fp16
    
    cb_wait_front(cb_rowIdx, 1);
    cb_get_tile(cb_rowIdx, 0 , &rowIdx_tile_ptr);
    rowIdx_tile_ptr = &rowIdx_tile_ptr[16]; 

    cb_wait_front(cb_scale, 1);
    cb_get_tile(cb_scale, 0 , &scale_tile_ptr);
    scale_tile_ptr = &scale_tile_ptr[8]; // in bfloat16

    for (uint32_t colIdx_tile_id = 0;
                  colIdx_tile_id < n_tiles_colIdx;
                  colIdx_tile_id++) {
        
        if(colIdx_tile_id >> 4 != current_row_tile_id) {
            cb_pop_front(cb_rowIdx, 1);
            cb_pop_front(cb_scale, 1);
            current_row_tile_id ++;

            cb_wait_front(cb_rowIdx, 1);
            cb_get_tile(cb_rowIdx, 0, &rowIdx_tile_ptr);
            rowIdx_tile_ptr = &rowIdx_tile_ptr[16];
            cb_wait_front(cb_scale, 1);
            cb_get_tile(cb_scale, 0, &scale_tile_ptr);
            scale_tile_ptr = &scale_tile_ptr[8];
        }

        cb_wait_front(cb_colIdx, 1);
        cb_get_tile(cb_colIdx, 0 , &colIdx_tile_ptr);
        colIdx_tile_ptr = &colIdx_tile_ptr[16];
          
        cb_reserve_back(cb_out0, 1); // seems not working
        cb_get_tile(cb_out0, colIdx_tile_id , &out_tile_ptr);
        out_tile_ptr = &out_tile_ptr[8]; //in bfloat16 
        
        for(uint16_t idx_b = 0; idx_b < elements_per_tile_colIdx; idx_b++) {
            rowidx = (rowIdx_tile_ptr[idx_b >> 4]);
            colidx = (colIdx_tile_ptr[idx_b]);
            codebook_index = (colidx + (rowidx << 6)); // for next codebook row, 64 entries per row
            codebook_value = bfloat16_to_float(codeBook_tile_ptr[codebook_index]);
            scale_value = bfloat16_to_float(scale_tile_ptr[idx_b >> 4]); // for a group
            out_tile_ptr[idx_b] = float_to_bfloat16(codebook_value * scale_value);
        }
        rowIdx_tile_ptr = &rowIdx_tile_ptr[64]; // elements_per_tile_colIdx / 16 = 64
        scale_tile_ptr = &scale_tile_ptr[64]; // elements_per_tile_colIdx / 16 = 64
        
        cb_push_back(cb_out0, 1);
        cb_pop_front(cb_colIdx, 1);

    }
    cb_pop_front(cb_rowIdx, 1);
    cb_pop_front(cb_codeBook, 1);

}  // namespace NAMESPACE

}