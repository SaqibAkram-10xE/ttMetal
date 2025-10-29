// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

inline float bfloat16_to_float(uint16_t bf16_val) {
    union {
        uint32_t u32;
        float f32;
    } u;
    u.u32 = static_cast<uint32_t>(bf16_val) << 16;
    return u.f32;
}

void kernel_main() {
    uint32_t colIdx_addr = get_arg_val<uint32_t>(0);
    uint32_t rowIdx_addr = get_arg_val<uint32_t>(1);
    uint32_t codeBook_addr = get_arg_val<uint32_t>(2);
    uint32_t n_tiles_colIdx = get_arg_val<uint32_t>(3);
    uint32_t tile_size_bytes_codebook = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_colIdx = tt::CBIndex::c_0;
    constexpr uint32_t cb_rowIdx = tt::CBIndex::c_1;
    constexpr uint32_t cb_codeBook = tt::CBIndex::c_2;

    // Get the tile size used in the circular buffers. We assume the
    // circular buffers are created with the same tile size as the DRAM
    // buffers (Whis is most of the cases).
    const uint32_t tile_size_bytes = get_tile_size(cb_colIdx);  // 1024*1 = 1024 // typically 2048
    const uint32_t tile_size_bytes_rowIdx = get_tile_size(cb_rowIdx);  // 1024*1 = 1024 //typically 2048
    // const uint32_t tile_size_bytes_codebook = get_tile_size(cb_codeBook);  // 64*2 = 128 //typically 2048
    DPRINT << "tile_size_bytes: " << tile_size_bytes << ENDL();
    DPRINT << "tile_size_bytes_rowIdx: " << tile_size_bytes_rowIdx << ENDL();
    DPRINT << "tile_size_bytes_codebook: " << tile_size_bytes_codebook << ENDL();

    const uint8_t* ptr16;
    uint32_t current_row_tile_id = 0;
    constexpr auto colIdx_args = TensorAccessorArgs<0>();
    const auto colIdx = TensorAccessor(colIdx_args, colIdx_addr, tile_size_bytes);
    constexpr auto rowIdx_args = TensorAccessorArgs<colIdx_args.next_compile_time_args_offset()>();
    const auto rowIdx = TensorAccessor(rowIdx_args, rowIdx_addr, tile_size_bytes_rowIdx);
    constexpr auto codeBook_args = TensorAccessorArgs<rowIdx_args.next_compile_time_args_offset()>();
    const auto codeBook = TensorAccessor(codeBook_args, codeBook_addr, tile_size_bytes_codebook);

    cb_reserve_back(cb_codeBook, 1);
    uint32_t cb_codeBook_addr = get_write_ptr(cb_codeBook);
    noc_async_read_tile(0, codeBook, cb_codeBook_addr); 
    noc_async_read_barrier();
    cb_push_back(cb_codeBook, 1);
    // ptr16 = reinterpret_cast<const uint16_t*>(cb_codeBook_addr);
    // DPRINT << "tile# " << 1 << ", cb_codeBook_addr from 0 to 1024 elements (bf16->f32): ";
    // for (uint32_t i = 0; i < 1024; i++) {
    //     float val = bfloat16_to_float(ptr16[i]);
    //     DPRINT << val << " ";
    // }
    // DPRINT << ENDL();
    
    cb_reserve_back(cb_rowIdx, 1);
    uint32_t cb_rowIdx_addr = get_write_ptr(cb_rowIdx);
    noc_async_read_tile(current_row_tile_id, rowIdx, cb_rowIdx_addr); 
    noc_async_read_barrier();
    cb_push_back(cb_rowIdx, 1);
    // ptr16 = reinterpret_cast<const uint16_t*>(cb_rowIdx_addr);
    // DPRINT << "tile# " << 1 << ", cb_rowIdx_addr from 1024 elements (bf16->f32): ";
    // for (uint32_t i = 0; i < 1024; i++) {
    //     float val = bfloat16_to_float(ptr16[i]);
    //     DPRINT << val << " ";
    // }
    // DPRINT << ENDL();

    for (uint32_t colIdx_tile_id = 0; colIdx_tile_id < n_tiles_colIdx; colIdx_tile_id++) {
        
        cb_reserve_back(cb_colIdx, 1);
        uint32_t cb_colIdx_addr = get_write_ptr(cb_colIdx);
        noc_async_read_tile(colIdx_tile_id, colIdx, cb_colIdx_addr);
        noc_async_read_barrier();
        cb_push_back(cb_colIdx, 1);
        // if (colIdx_tile_id ==2)    
        // {
        //     ptr16 = reinterpret_cast<const uint8_t*>(cb_colIdx_addr);
        //     DPRINT << "tile# " << colIdx_tile_id << ", cb_colIdx_addr from 0 to 1024 elements (bf16->f32): ";
        //     for (uint32_t i = 0; i < 1024; i++) {
        //         int val = (ptr16[i]);
        //         DPRINT << val << " ";
        //     }
        //     DPRINT << ENDL();
        // }

        if(colIdx_tile_id / 16 != current_row_tile_id) {
            current_row_tile_id++;
            DPRINT << "current_row_tile_id: " << current_row_tile_id << ENDL();
            cb_reserve_back(cb_rowIdx, 1);
            uint32_t cb_rowIdx_addr = get_write_ptr(cb_rowIdx);
            noc_async_read_tile(current_row_tile_id, rowIdx, cb_rowIdx_addr); 
            noc_async_read_barrier();
            cb_push_back(cb_rowIdx, 1);
            // if (current_row_tile_id < 4)    
            // {
            //     ptr16 = reinterpret_cast<const uint16_t*>(cb_colIdx_addr);
            //     DPRINT << "tile# " << colIdx_tile_id << ", cb_colIdx_addr from 0 to 1024 elements (bf16->f32): ";
            //     for (uint32_t i = 0; i < 1024; i++) {
            //         float val = bfloat16_to_float(ptr16[i]);
            //         DPRINT << val << " ";
            //     }
            //     DPRINT << ENDL();
            // }
            
        }
    }
} // namespace NAMESPACE
