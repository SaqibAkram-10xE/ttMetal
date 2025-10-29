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
static int count1 = 0;
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

    cb_wait_front(cb_codeBook, 1);
    cb_get_tile(cb_codeBook, 0 , &codeBook_tile_ptr);
    codeBook_tile_ptr = &codeBook_tile_ptr[8]; // in fp16
    
    cb_wait_front(cb_rowIdx, 1);
    cb_get_tile(cb_rowIdx, 0 , &rowIdx_tile_ptr);
    rowIdx_tile_ptr = &rowIdx_tile_ptr[16]; 

    static uint16_t outIdx = 0;
    
    for (uint32_t colIdx_tile_id = 0; colIdx_tile_id < n_tiles_colIdx; colIdx_tile_id++) {
        cb_wait_front(cb_colIdx, 1);
        cb_get_tile(cb_colIdx, 0 , &colIdx_tile_ptr);
        colIdx_tile_ptr = &colIdx_tile_ptr[16];
    
        
        cb_reserve_back(cb_out0, 1); // seems not working
        cb_get_tile(cb_out0, colIdx_tile_id , &out_tile_ptr);
        out_tile_ptr = &out_tile_ptr[8]; //in bfloat16
        // if(outIdx == 1){
        //     outIdx = 0;
        // }else{
        //     outIdx++;
        // }
        DPRINT << "PRODUCER: col_tile_id=" << (int)colIdx_tile_id
                << " reserved_out_ptr=" << (int)out_tile_ptr << ENDL();

        // DPRINT << "PRODUCER: out_ptr_offset=" << (uintptr_t)out_ptr_offset << ENDL();


        int rowChunkId = 1024 * colIdx_tile_id;
        for(size_t idx_b = 0; idx_b < elements_per_tile_colIdx; idx_b++) {
            int rowidx = (int)(rowIdx_tile_ptr[(int)((idx_b + rowChunkId) / 16) ]);
            int colidx = (int)(colIdx_tile_ptr[idx_b]);
            int codebook_index = (int)(colidx + (rowidx << 4));
            if((idx_b == 0) && (colIdx_tile_id < n_tiles_colIdx)) {
                DPRINT << "idx: " << (int)idx_b << " tile # " << colIdx_tile_id;
                DPRINT << " codebook_index="<< (int)codebook_index << " rowidx=" << (int)rowidx << " colidx=" << (int)colidx << "\n";
                float value = bfloat16_to_float(codeBook_tile_ptr[codebook_index]);
                DPRINT << value << ENDL();
            }
            if(codebook_index >= 64) {
                if(count < 3) {
                    DPRINT <<"Error idx_b: " << (int)idx_b << " col_tile #: " << (int)colIdx_tile_id << ".\t";
                    DPRINT <<"codebook_index="<< (int)codebook_index << " rowidx=" << (int)rowidx << " colidx=" << (int)colidx << "\n";
                }
                count++;
            } else {
                out_tile_ptr[idx_b] = (codeBook_tile_ptr[codebook_index]);

                if((count1 < 40) && (colIdx_tile_id < n_tiles_colIdx)) {
                    DPRINT << bfloat16_to_float(out_tile_ptr[idx_b]) << " ";   
                    // DPRINT << "Tile:" << (int)colIdx_tile_id << ", index:" << (int)idx_b;// ENDL();
                    // DPRINT << ", codebook_index="<< (int)codebook_index << " rowidx=" << (int)rowidx << " colidx=" << (int)colidx << "\n";
                }
                if(colIdx_tile_id < n_tiles_colIdx){
                    count1++;
                }
            }
        }
        DPRINT << ENDL();
        count1 = 0;
       
        cb_push_back(cb_out0, 1);
        DPRINT << "PRODUCER: pushed back tile for col_tile_id=" << (int)colIdx_tile_id << ENDL();

        cb_pop_front(cb_colIdx, 1);
    }
    cb_pop_front(cb_rowIdx, 1);
    cb_pop_front(cb_codeBook, 1);
}
}  // namespace NAMESPACE


        // get_cb_capacity(cb_out0)

// if(colIdx_tile_id / 16 != current_row_tile_id) {
//             current_row_tile_id ++;
//             cb_pop_front(cb_rowIdx, 1);
//             cb_wait_front(cb_rowIdx, 1);
//             cb_get_tile(cb_rowIdx, 0, &rowIdx_tile_ptr);
//             rowIdx_tile_ptr = &rowIdx_tile_ptr[16];
//         }




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