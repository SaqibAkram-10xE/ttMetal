# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import numpy as np

# --- Setup device ---
device_id = 0
device = ttnn.open_device(device_id=device_id)

# --- Define shapes and datatypes ---
# Example: assume 1 tile = 32 elements for simplicity; use your real shape/tile size
num_row_tiles = 1
num_col_tiles = 1
num_codebook_tiles = 1
elements_per_tile = 32

# --- Create random numpy arrays following your logic ---
rng = np.random.default_rng()

# colIdx in [0, 15]
colIdx_np = rng.integers(low=0, high=16, size=(num_col_tiles, elements_per_tile), dtype=np.uint8)

# rowIdx in [0, 3]
rowIdx_np = rng.integers(low=0, high=4, size=(num_row_tiles, elements_per_tile), dtype=np.uint8)

# codebook repeating 0..63
codebook_np = np.arange(0, 64, dtype=np.uint8)
codebook_np = np.tile(codebook_np, int(np.ceil(elements_per_tile * num_codebook_tiles / 64)))[:elements_per_tile * num_codebook_tiles]
codebook_np = codebook_np.reshape(num_codebook_tiles, elements_per_tile)

# --- Convert to torch tensors ---
rowIdx_torch = torch.tensor(rowIdx_np, dtype=torch.float32)
codeBook_torch = torch.tensor(codebook_np, dtype=torch.float32)
colIdx_torch = torch.tensor(colIdx_np, dtype=torch.float32)

# --- Move to device ---
rowIdx_tt = ttnn.from_torch(rowIdx_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
codeBook_tt = ttnn.from_torch(codeBook_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
colIdx_tt = ttnn.from_torch(colIdx_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

# --- Run your example op ---
output_tensor = ttnn.example(rowIdx_tt, codeBook_tt, colIdx_tt)

# --- Copy back to host ---
torch_output = ttnn.to_torch(output_tensor)

print("Output tensor:")
print(torch_output)

# --- Cleanup ---
ttnn.close_device(device)

