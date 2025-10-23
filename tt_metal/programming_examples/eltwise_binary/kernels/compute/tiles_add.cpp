inline float bfloat16_to_float(uint16_t bf16_val) {
    union {
        uint32_t u32;
        float f32;
    } u;
    u.u32 = static_cast<uint32_t>(bf16_val) << 16;
    return u.f32;
}

namespace NAMESPACE {
void MAIN {

    // args:
    // 0: n_tiles_colIdx
    // 1: elements_per_tile_colIdx
    // 2: n_tiles_rowIdx
    // 3: elements_per_tile_row
    uint32_t n_tiles_colIdx = get_arg_val<uint32_t>(0);
    uint32_t elements_per_tile_colIdx = get_arg_val<uint32_t>(1);
    uint32_t n_tiles_rowIdx = get_arg_val<uint32_t>(2);
    uint32_t elements_per_tile_row = get_arg_val<uint32_t>(3);

    constexpr auto cb_colIdx = tt::CBIndex::c_0;
    constexpr auto cb_rowIdx = tt::CBIndex::c_1;
    constexpr auto cb_codeBook = tt::CBIndex::c_2;
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    volatile uint16_t* colIdx_tile_ptr_f32    = nullptr;
    volatile uint16_t* rowIdx_tile_ptr_f32     = nullptr;
    volatile uint16_t* codeBook_tile_ptr_f32 = nullptr;
    volatile uint16_t* out_tile_ptr_f32 = nullptr;

    // Get codeBook (assumed small, single tile)
    cb_wait_front(cb_codeBook, 1);
    cb_get_tile(cb_codeBook, 0 , &codeBook_tile_ptr_f32);
    // your original code used +8 offset
    codeBook_tile_ptr_f32 = &codeBook_tile_ptr_f32[8];

    // We'll fetch row tiles on demand; initialize current id to invalid
    uint32_t current_row_tile_id = UINT32_MAX;

    // Reserve/pop management: we'll pop row tiles when we no longer need them.
    // Note: depending on Tenstorrent runtime semantics you might need different
    // pop counts; this implementation pops a row tile when we advance past it.

    for (uint32_t i = 0; i < n_tiles_colIdx; i++) {
        // Wait for and fetch the current colIdx tile (tile index = i)
        cb_wait_front(cb_colIdx, 1);
        cb_get_tile(cb_colIdx, i , &colIdx_tile_ptr_f32);
        colIdx_tile_ptr_f32 = &colIdx_tile_ptr_f32[8];

        // prepare out tile for this col tile
        cb_reserve_back(cb_out0, 1);
        cb_get_tile(cb_out0, i , &out_tile_ptr_f32);
        out_tile_ptr_f32 = &out_tile_ptr_f32[8];

        // For each element in the col tile compute global indices and map to row tile/index
        for(size_t idx_b = 0; idx_b < elements_per_tile_colIdx; idx_b++) {
            // global column index (across entire colIdx array)
            uint64_t global_col_idx = (uint64_t)i * (uint64_t)elements_per_tile_colIdx + (uint64_t)idx_b;

            // the row index corresponding to this column: each rowIdx covers 16 colIdx values
            uint64_t row_global_idx = global_col_idx / 16ULL;

            // determine which row tile and index-within-row-tile hold this row value
            uint32_t row_tile_id = (uint32_t)(row_global_idx / (uint64_t)elements_per_tile_row);
            uint32_t row_idx_in_tile = (uint32_t)(row_global_idx % (uint64_t)elements_per_tile_row);

            // sanity check tile id
            if (row_tile_id >= n_tiles_rowIdx) {
                // out of range: report and continue (or you could clamp)
                if (idx_b < 10) {
                    DPRINT << "Row tile out of range: row_tile_id=" << (int)row_tile_id
                           << " n_tiles_rowIdx=" << (int)n_tiles_rowIdx
                           << " global_col_idx=" << (unsigned long long)global_col_idx << "\n";
                }
                //break;
                // set some safe fallback (e.g., 0)
                row_tile_id = n_tiles_rowIdx ? (n_tiles_rowIdx - 1) : 0;
            }

            // fetch the needed row tile if it's not already loaded
            if (row_tile_id != current_row_tile_id) {
                // if we had a previous row tile we can pop it (release it)
                if (current_row_tile_id != UINT32_MAX) {
                    cb_pop_front(cb_rowIdx, 1);
                }
                // wait for the next row tile to be available then get it
                cb_wait_front(cb_rowIdx, 1);
                cb_get_tile(cb_rowIdx, row_tile_id, &rowIdx_tile_ptr_f32);
                rowIdx_tile_ptr_f32 = &rowIdx_tile_ptr_f32[8];
                current_row_tile_id = row_tile_id;
            }

            // load row and col bfloat16 values and create codebook index
            float rowidx = bfloat16_to_float(rowIdx_tile_ptr_f32[row_idx_in_tile]);
            float colidx = bfloat16_to_float(colIdx_tile_ptr_f32[idx_b]);

            uint8_t codebook_index = (uint8_t)(colidx + (rowidx * 16.0f));

            if(codebook_index >= 64) {
                if(idx_b < 10) {
                    DPRINT <<"Error idx_b: " << (int)idx_b << " col_tile #: " << (int)i << ".\t";
                    DPRINT <<"codebook_index="<< (int)codebook_index << " rowidx=" << (float)rowidx << " colidx=" << (float)colidx << "\n";
                }
                // optionally clamp:
                codebook_index = (uint8_t)(codebook_index & 0x3F);
            }

            // write out the bfloat16 codebook entry (copied directly)
            out_tile_ptr_f32[idx_b] = (codeBook_tile_ptr_f32[codebook_index]);

            // if you need to inspect as float, compute:
            // float output = bfloat16_to_float(out_tile_ptr_f32[idx_b]);
        } // end idx_b loop

        // push and pack the output tile, and release the processed col tile
        pack_tile(i, cb_out0);
        cb_push_back(cb_out0, 1);
        cb_pop_front(cb_colIdx, 1);
    } // end col tile loop

    // clean up: pop any remaining row/codebook tiles we held
    if (current_row_tile_id != UINT32_MAX) {
        cb_pop_front(cb_rowIdx, 1);
    }
    cb_pop_front(cb_codeBook, 1);
}
}  // namespace NAMESPACE





// -----------------OLD Version----------------//
/*
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

namespace NAMESPACE {
void MAIN {

    uint32_t n_tiles_colIdx = get_arg_val<uint32_t>(0);
    uint32_t elements_per_tile_colIdx = get_arg_val<uint32_t>(1);

    constexpr auto cb_colIdx = tt::CBIndex::c_0;
    constexpr auto cb_rowIdx = tt::CBIndex::c_1;
    constexpr auto cb_codeBook = tt::CBIndex::c_2;
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    volatile uint16_t* colIdx_tile_ptr_f32    = nullptr;
    volatile uint16_t* rowIdx_tile_ptr_f32     = nullptr;
    volatile uint16_t* codeBook_tile_ptr_f32 = nullptr;
    volatile uint16_t* out_tile_ptr_f32 = nullptr;
   
    cb_wait_front(cb_codeBook, 1);
    cb_get_tile(cb_codeBook, 0 , &codeBook_tile_ptr_f32);
    codeBook_tile_ptr_f32 = &codeBook_tile_ptr_f32[8];

    cb_wait_front(cb_rowIdx, 1);
    cb_get_tile(cb_rowIdx, 0 , &rowIdx_tile_ptr_f32);
    rowIdx_tile_ptr_f32 = &rowIdx_tile_ptr_f32[8]; 

    for (uint32_t i = 0; i < n_tiles_colIdx; i++) {
        cb_wait_front(cb_colIdx, 1);
    
        cb_get_tile(cb_colIdx, 0 , &colIdx_tile_ptr_f32);
        colIdx_tile_ptr_f32 = &colIdx_tile_ptr_f32[8];
    
        cb_reserve_back(cb_out0, 1);
        cb_get_tile(cb_out0, i , &out_tile_ptr_f32);
        out_tile_ptr_f32 = &out_tile_ptr_f32[8];

        for(size_t idx_b = 0; idx_b < elements_per_tile_colIdx; idx_b++) {
            float rowidx = bfloat16_to_float(rowIdx_tile_ptr_f32[ (idx_b) / 16 ]);
            float colidx = bfloat16_to_float(colIdx_tile_ptr_f32[ (idx_b)]);
            uint8_t codebook_index = (uint8_t)(colidx + (rowidx * 16.0f));
    
            if(codebook_index >= 64) {
                if(idx_b < 10)
                {
                    DPRINT <<"Error idx_b: " << (int)idx_b << " tile #: " << (int)i << ".\t";
                    DPRINT <<"codebook_index="<< (int)codebook_index << " rowidx=" << (float)rowidx << " colidx=" << (float)colidx << "\n";
                }
            } else {
                out_tile_ptr_f32[idx_b] = (codeBook_tile_ptr_f32[codebook_index]);
                float output = bfloat16_to_float(out_tile_ptr_f32[idx_b]);
            }
        }
        pack_tile(i, cb_out0);
        cb_push_back(cb_out0, 1);
        cb_pop_front(cb_colIdx, 1);
    }

    cb_pop_front(cb_codeBook, 1);
    cb_pop_front(cb_rowIdx, 1);
}
}  // namespace NAMESPACE
*/