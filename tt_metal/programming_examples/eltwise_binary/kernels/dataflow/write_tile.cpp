// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    uint32_t c_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles_colIdx = get_arg_val<uint32_t>(1);

    // The circular buffer that we are going to read from and write to DRAM
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
    const uint32_t tile_size_bytes = get_tile_size(cb_out0);

    // Address of the output buffer
    constexpr auto out0_args = TensorAccessorArgs<0>();
    const auto out0 = TensorAccessor(out0_args, c_addr, tile_size_bytes);
    // Loop over all the tiles and write them to the output buffer
    for (uint32_t i = 0; i < n_tiles_colIdx; i++) {
        // Make sure there is a tile in the circular buffer
        cb_wait_front(cb_out0, 1);
        uint32_t cb_out0_addr = get_read_ptr(cb_out0);
        // write the tile to DRAM
        noc_async_write_tile(i, out0, cb_out0_addr);
        
        
        noc_async_write_barrier();  // This will wait until the write is done. As an alternative,
                                    // noc_async_write_flushed() can be faster because it waits
                                    // until the write request is sent. In that case, you have to
                                    // use noc_async_write_barrier() at least once at the end of
                                    // data movement kernel to make sure all writes are done.
        // Mark the tile as consumed
        if(i == 0){
            DPRINT << "\n\n Writer KERNEL: \n\n";
            const uint8_t * ptr8t = reinterpret_cast<const uint8_t*>(cb_out0_addr);
            DPRINT << "tile# " << i << ", cb_out0_addr writer: ";
            for (uint32_t i = 0; i < 1024; i++) {
                int val = (ptr8t[i]);
                DPRINT << val << " ";
            }
            DPRINT << ENDL();
        }
        cb_pop_front(cb_out0, 1);
    }
}