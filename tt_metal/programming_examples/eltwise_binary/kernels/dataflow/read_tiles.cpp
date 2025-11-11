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
inline uint16_t float_to_bfloat16(float fval) {
    union {
        uint32_t u32;
        float f32;
    } u;
    u.f32 = fval;
    return static_cast<uint16_t>(u.u32 >> 16);
}

void kernel_main() {
    uint32_t colIdx_addr = get_arg_val<uint32_t>(0);
    uint32_t rowIdx_addr = get_arg_val<uint32_t>(1);
    uint32_t codeBook_addr = get_arg_val<uint32_t>(2);
    uint32_t scales_addr = get_arg_val<uint32_t>(3);
    uint32_t out_addr = get_arg_val<uint32_t>(4);
    uint32_t n_tiles_colIdx = get_arg_val<uint32_t>(5);
    uint32_t tile_size_bytes_codebook = get_arg_val<uint32_t>(6);
    uint32_t start_tile_id = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_colIdx = tt::CBIndex::c_0;
    constexpr uint32_t cb_rowIdx = tt::CBIndex::c_1;
    constexpr uint32_t cb_codeBook = tt::CBIndex::c_2;
    constexpr uint32_t cb_scales = tt::CBIndex::c_3;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    const uint32_t tile_size_bytes = get_tile_size(cb_colIdx);  // 1024*1 = 1024 // typically 2048
    const uint32_t tile_size_bytes_rowIdx = get_tile_size(cb_rowIdx);  // 1024*1 = 1024 //typically 2048
    const uint32_t tile_size_bytes_scales = get_tile_size(cb_scales);  // 1024*1 = 1024 //typically 2048
    // const uint32_t tile_size_bytes_codebook = get_tile_size(cb_codeBook);  // 1024*1 = 1024 //typically 2048
    const uint32_t out_tile_size_bytes = get_tile_size(cb_out);  // 1024*2 = 2048
    DPRINT << "READER: tile_size_bytes_scales: " << tile_size_bytes_scales << ENDL();

    const uint16_t* ptr16_codebook;
    uint16_t* ptr16_output;
    const uint8_t* ptr8_row;
    const uint8_t* ptr8_col;
    const uint16_t* ptr16_scales;
    uint32_t current_row_tile_id = start_tile_id;
    int codebook_index = 0;
    int rowidx = 0;
    int colidx = 0;
    float output_value = 0.0f;

    constexpr auto colIdx_args = TensorAccessorArgs<0>();
    const auto colIdx = TensorAccessor(colIdx_args, colIdx_addr, tile_size_bytes);

    constexpr auto rowIdx_args = TensorAccessorArgs<colIdx_args.next_compile_time_args_offset()>();
    const auto rowIdx = TensorAccessor(rowIdx_args, rowIdx_addr, tile_size_bytes_rowIdx);

    constexpr auto codeBook_args = TensorAccessorArgs<rowIdx_args.next_compile_time_args_offset()>();
    const auto codeBook = TensorAccessor(codeBook_args, codeBook_addr, tile_size_bytes_codebook);

    constexpr auto scales_args = TensorAccessorArgs<codeBook_args.next_compile_time_args_offset()>();
    const auto scales = TensorAccessor(scales_args, scales_addr, tile_size_bytes_scales);

    constexpr auto out_args = TensorAccessorArgs<scales_args.next_compile_time_args_offset()>();
    const auto output = TensorAccessor(out_args, out_addr, out_tile_size_bytes);

    uint32_t cb_codeBook_addr = get_write_ptr(cb_codeBook);
    noc_async_read_tile(0, codeBook, cb_codeBook_addr);
    ptr16_codebook = reinterpret_cast<const uint16_t*>(cb_codeBook_addr);
    noc_async_read_barrier();

    uint32_t cb_rowIdx_addr = get_write_ptr(cb_rowIdx);
    noc_async_read_tile(current_row_tile_id, rowIdx, cb_rowIdx_addr);
    ptr8_row = reinterpret_cast<const uint8_t*>(cb_rowIdx_addr);
    noc_async_read_barrier();

    uint32_t cb_scales_addr = get_write_ptr(cb_scales);
    noc_async_read_tile(current_row_tile_id, scales, cb_scales_addr);
    ptr16_scales = reinterpret_cast<const uint16_t*>(cb_scales_addr);
    noc_async_read_barrier();

    // DPRINT << "n_tiles_colIdx: " << n_tiles_colIdx << ENDL();
    for (uint32_t colIdx_tile_id = start_tile_id; colIdx_tile_id < start_tile_id + n_tiles_colIdx; colIdx_tile_id++) {
        if(colIdx_tile_id / 16 != current_row_tile_id) {
            current_row_tile_id++;
            cb_rowIdx_addr = get_write_ptr(cb_rowIdx);
            noc_async_read_tile(current_row_tile_id, rowIdx, cb_rowIdx_addr);
            ptr8_row = reinterpret_cast<const uint8_t*>(cb_rowIdx_addr);
            noc_async_read_barrier();

            cb_scales_addr = get_write_ptr(cb_scales);
            noc_async_read_tile(current_row_tile_id, scales, cb_scales_addr);
            ptr16_scales = reinterpret_cast<const uint16_t*>(cb_scales_addr);
            noc_async_read_barrier();
        }

        uint32_t cb_colIdx_addr = get_write_ptr(cb_colIdx);
        noc_async_read_tile(colIdx_tile_id, colIdx, cb_colIdx_addr);
        ptr8_col = reinterpret_cast<const uint8_t*>(cb_colIdx_addr);
        noc_async_read_barrier();

        // Take output ptr for out tile

        // uint64_t cb_out_addr = get_noc_addr(colIdx_tile_id, cb_out);

        uint32_t cb_out_addr = get_write_ptr(cb_out);
        ptr16_output = reinterpret_cast<uint16_t*>(cb_out_addr);
        // DPRINT << "." ;
        for (uint16_t idx_b = 0; idx_b < 1024; idx_b++) {
            rowidx = (ptr8_row[idx_b >> 4]);
            colidx = (ptr8_col[idx_b]);
            codebook_index = (colidx + (rowidx << 4));

            ptr16_output[idx_b] = (ptr16_codebook[codebook_index]);
            output_value = bfloat16_to_float(ptr16_output[idx_b]) * bfloat16_to_float(ptr16_scales[idx_b >> 4]);
            ptr16_output[idx_b] = float_to_bfloat16(output_value);
            if ((idx_b == 0) && (colIdx_tile_id == start_tile_id)) {
                DPRINT << "READER/COMPUTE: out_cb_addr=" << (int)ptr16_output << " tile_index=" << colIdx_tile_id
                       << ENDL();
                DPRINT << "out[0]: " << output_value << ENDL();
                // DPRINT << "colidx: " << (int)colidx << ENDL();
                // DPRINT << "rowidx: " << (int)rowidx << ENDL();
                DPRINT << "codebook_index: " << (int)codebook_index << ENDL();
                DPRINT << "Scale value " << bfloat16_to_float(ptr16_scales[idx_b >> 4]) << ENDL();
            }
        }
        ptr8_row = &ptr8_row[64];
        ptr16_scales = &ptr16_scales[64];

        // noc_async_write(ptr16_output, cb_out_addr, sizeof(uint32_t));
        // noc_async_write_barrier();

        noc_async_write_tile(colIdx_tile_id, output, cb_out_addr);
        noc_async_write_barrier();  // from output tile
    }

    // noc_async_read_barrier();
}

// } // namespace NAMESPACE
