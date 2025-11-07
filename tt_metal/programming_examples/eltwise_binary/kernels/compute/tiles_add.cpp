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
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    volatile uint8_t* colIdx_tile_ptr = nullptr;
    volatile uint8_t* rowIdx_tile_ptr = nullptr;
    volatile uint16_t* codeBook_tile_ptr = nullptr;
    volatile uint16_t* out_tile_ptr = nullptr;

    uint32_t current_row_tile_id = 0;
    int codebook_index = 0;
    int rowidx = 0;
    int colidx = 0;

    cb_wait_front(cb_codeBook, 1);
    cb_get_tile(cb_codeBook, 0 , &codeBook_tile_ptr);
    codeBook_tile_ptr = &codeBook_tile_ptr[8]; // in fp16

    cb_wait_front(cb_rowIdx, 1);
    cb_get_tile(cb_rowIdx, 0 , &rowIdx_tile_ptr);
    rowIdx_tile_ptr = &rowIdx_tile_ptr[16];

    for (uint32_t colIdx_tile_id = 0; colIdx_tile_id < n_tiles_colIdx; colIdx_tile_id++) {
        if(colIdx_tile_id >> 4 != current_row_tile_id) {
            cb_pop_front(cb_rowIdx, 1);
            current_row_tile_id ++;
            cb_wait_front(cb_rowIdx, 1);
            cb_get_tile(cb_rowIdx, 0, &rowIdx_tile_ptr);
            rowIdx_tile_ptr = &rowIdx_tile_ptr[16];
        }

        cb_wait_front(cb_colIdx, 1);
        cb_get_tile(cb_colIdx, 0 , &colIdx_tile_ptr);
        colIdx_tile_ptr = &colIdx_tile_ptr[16];
        // DPRINT << "COMPUTE: colIdx_tile_ptr=" << (int)colIdx_tile_ptr <<" tile_index=" << colIdx_tile_id << ENDL();

        // tile_regs_acquire();
        cb_reserve_back(cb_out0, 1); // seems not working
        cb_get_tile(cb_out0, colIdx_tile_id, &out_tile_ptr);
        out_tile_ptr += 8;

        // Alternate ±2048 depending on tile id
        // if (colIdx_tile_id > 0) {
        //     int offset = (colIdx_tile_id % 2 == 1) ? -2048 : +2048;
        //     out_tile_ptr += offset;
        // }
        // #ifdef TRISC_MATH
        for(uint16_t idx_b = 0; idx_b < elements_per_tile_colIdx; idx_b++) {
            rowidx = (rowIdx_tile_ptr[idx_b >> 4]);
            colidx = (colIdx_tile_ptr[idx_b]);
            codebook_index = (colidx + (rowidx << 4));
            out_tile_ptr[idx_b] = (codeBook_tile_ptr[codebook_index]);
            // #ifdef TRISC_MATH
            // if ((idx_b == 0) && (colIdx_tile_id <= 2)) {
            //     DPRINT << "COMPUTE: out_cb_addr=" << (int)out_tile_ptr <<" tile_index=" << colIdx_tile_id << ENDL();
            // DPRINT << "out[0]: " << bfloat16_to_float(out_tile_ptr[idx_b]) << ENDL();
            // DPRINT << "colidx: " << (int)colidx << ENDL();
            // DPRINT << "rowidx: " << (int)rowidx << ENDL();
            // DPRINT << "codebook_index: " << (int)codebook_index << ENDL();
            // DPRINT << "out[1]: " << (int)out_tile_ptr[idx_b+1] << ENDL();
            // }
            // #endif
        }
        // #endif
        rowIdx_tile_ptr = &rowIdx_tile_ptr[64];
        // out_tile_ptr -= 2048; // bfloat16 uses 2 bytes

        cb_pop_front(cb_colIdx, 1);
        cb_push_back(cb_out0, 1);

        // tile_regs_commit();
        // tile_regs_wait();

        // tile_regs_release();
    }
    cb_pop_front(cb_rowIdx, 1);
    cb_pop_front(cb_codeBook, 1);

}  // namespace NAMESPACE

}  // namespace NAMESPACE
