// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

inline float bfloat16_to_float(uint16_t bf16_val) {
    union {
        uint32_t u32;
        float f32;a
    } u;
    u.u32 = static_cast<uint32_t>(bf16_val) << 16;
    return u.f32;
}

void kernel_main() {
    // Read parameters from the kernel arguments
    uint32_t colIdx_addr = get_arg_val<uint32_t>(0);
    uint32_t rowIdx_addr = get_arg_val<uint32_t>(1);
    uint32_t codeBook_addr = get_arg_val<uint32_t>(2);
    uint32_t n_tiles_colIdx = get_arg_val<uint32_t>(3);

    // The circular buffers to read the tiles into
    constexpr uint32_t cb_colIdx = tt::CBIndex::c_0;
    constexpr uint32_t cb_rowIdx = tt::CBIndex::c_1;
    constexpr uint32_t cb_codeBook = tt::CBIndex::c_2;

    // Get the tile size used in the circular buffers. We assume the
    // circular buffers are created with the same tile size as the DRAM
    // buffers (Whis is most of the cases).
    const uint32_t tile_size_bytes = get_tile_size(cb_colIdx);  // typically 2048

    // Create address generators for the input buffers. Consider these the
    // pointers for interleaved buffers
    // Setting the page size to be tile_size_bytes works because we set it up
    // explicitly in host code. This is usually a good idea as it makes coding
    // easy.
    constexpr auto colIdx_args = TensorAccessorArgs<0>();
    const auto colIdx = TensorAccessor(colIdx_args, colIdx_addr, tile_size_bytes);
    constexpr auto rowIdx_args = TensorAccessorArgs<colIdx_args.next_compile_time_args_offset()>();
    const auto rowIdx = TensorAccessor(rowIdx_args, rowIdx_addr, tile_size_bytes);
    constexpr auto codeBook_args = TensorAccessorArgs<rowIdx_args.next_compile_time_args_offset()>();
    const auto codeBook = TensorAccessor(codeBook_args, codeBook_addr, tile_size_bytes);


    cb_reserve_back(cb_codeBook, 1);
    uint32_t cb_codeBook_addr = get_write_ptr(cb_codeBook);
    noc_async_read_tile(0, codeBook, cb_codeBook_addr); 

    // const uint16_t* ptr16 = reinterpret_cast<const uint16_t*>(cb_codeBook_addr);
    // DPRINT << "tile# " << 1 << ", cb_codeBook_addr from 0 to 64 elements (bf16->f32): ";
    // for (uint32_t i = 0; i < 64; i++) {
    //     float val = bfloat16_to_float(ptr16[i]);
    //     DPRINT << val << " ";
    // }
    // DPRINT << ENDL();

    noc_async_read_barrier();
    cb_push_back(cb_codeBook, 1);


    cb_reserve_back(cb_rowIdx, 1);
    uint32_t cb_rowIdx_addr = get_write_ptr(cb_rowIdx);
    noc_async_read_tile(0, rowIdx, cb_rowIdx_addr); 
    
    // const uint16_t* ptr16 = reinterpret_cast<const uint16_t*>(cb_rowIdx_addr);
    // DPRINT << "tile# " << 1 << ", cb_rowIdx_addr from 0 to 64 elements (bf16->f32): ";
    // for (uint32_t i = 0; i < 1024; i++) {
    //     float val = bfloat16_to_float(ptr16[i]);
    //     DPRINT << val << " ";
    // }
    // DPRINT << ENDL();

    noc_async_read_barrier();
    cb_push_back(cb_rowIdx, 1);




    // Loop over all the tiles and read them into the circular buffers
    for (uint32_t i = 0; i < n_tiles_colIdx; i++) {
        // First make sure there is space in the circular buffers to be written to.
        cb_reserve_back(cb_colIdx, 1);
        // cb_reserve_back(cb_rowIdx, 1);  // wait until we have 1 free slot. This blocks if the
                                     // other kernels cannot consume the tiles fast enough.
                                     // Deciding how large the buffer should be is a tradeoff.
        uint32_t cb_colIdx_addr = get_write_ptr(cb_colIdx);
        // uint32_t cb_rowIdx_addr = get_write_ptr(cb_rowIdx);
        noc_async_read_tile(i, colIdx, cb_colIdx_addr);  // read the tile into the circular buffer
        // noc_async_read_tile(i, rowIdx, cb_rowIdx_addr);  // We can overlap async reads and writes
                                                   // to reduce the data movement overhead.

        noc_async_read_barrier();  // Wait until tile reads are done

        // const uint16_t* ptr16 = reinterpret_cast<const uint16_t*>(cb_colIdx_addr);
        // DPRINT << "tile# " << i << ", cb_colIdx_addr from 0 to 64 elements (bf16->f32): ";
        // for (uint32_t i = 0; i < 64; i++) {
        //     float val = bfloat16_to_float(ptr16[i]);
        //     DPRINT << val << " ";
        // }
        // DPRINT << ENDL();

        cb_push_back(cb_colIdx, 1);
        // cb_push_back(cb_rowIdx, 1);  // mark the tiles as ready. From this point forward kernels
                                  // calling `cb_wait_front` will see this tile
    }
}