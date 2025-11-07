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
    uint32_t c_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles_colIdx = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
    const uint32_t tile_size_bytes = get_tile_size(cb_out0); // 1024*2 = 2048
    // DPRINT << "WRITER: Tile size: " << tile_size_bytes << ENDL();
    // DPRINT << "WRITER: Number of tiles to write: " << n_tiles_colIdx << ENDL();
    constexpr auto out0_args = TensorAccessorArgs<0>();
    const auto out0 = TensorAccessor(out0_args, c_addr, tile_size_bytes);

    for (uint32_t i = 0; i < n_tiles_colIdx; i++) {
        cb_wait_front(cb_out0, 1);
        uint32_t cb_out0_addr = get_read_ptr(cb_out0);
        if (i <= 2) {
            DPRINT << "WRITER: read_cb_addr=" << (int)cb_out0_addr << " tile_index=" << i << ENDL();
            // //     const uint16_t* ptr16t = reinterpret_cast<const uint16_t*>(cb_out0_addr);
            // //     DPRINT << "Writer KERNEL: tile# " << i << ", cb_out0_addr 10 writer: ";
            // //     for (uint32_t j = 0; j < 5; j++) {
            // //         float val = bfloat16_to_float(ptr16t[j]);
            // //         DPRINT << val << " ";
            // //     }
            // //     DPRINT << ENDL();
        }
        noc_async_write_tile(i, out0, cb_out0_addr);
        // noc_async_writes_flushed();
        noc_async_write_barrier();  // This will wait until the write is done. As an alternative,
                                    // noc_async_write_flushed() can be faster because it waits
                                    // until the write request is sent. In that case, you have to
                                    // use noc_async_write_barrier() at least once at the end of
                                    // data movement kernel to make sure all writes are done.
        cb_pop_front(cb_out0, 1);
    }
    // noc_async_write_barrier();
}
