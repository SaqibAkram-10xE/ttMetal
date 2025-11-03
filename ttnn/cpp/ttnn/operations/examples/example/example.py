# import torch
# import ttnn

# device_id = 0
# device = ttnn.open_device(device_id=device_id)



# import ttnn

# # For the golden function, use the same signature as the operation
# # Keep in mind that all `ttnn.Tensor`s are converted to `torch.Tensor`s
# # And arguments not needed by torch can be ignored using `*args` and `**kwargs`
# def golden_function(input_tensor: "torch.Tensor", *args, **kwargs):
#     output_tensor:  "torch.Tensor" = ...
#     return output_tensor

# # TT-NN Tensors are converted to torch tensors before calling the golden function automatically
# # And the outputs are converted back to TT-NN Tensors
# # But in some cases you may need to preprocess the inputs and postprocess the outputs manually

# # In order to preprocess the inputs manually, use the following signature
# # Note that the arguments are not packed into *args and **kwargs as in the golden function!!!
# def preprocess_golden_function_inputs(args, kwargs):
#     # i.e.
#     ttnn_input_tensor = args[0]
#     return ttnn.to_torch(ttnn_input_tensor)

# # In order to postprocess the outputs manually, use the following signature
# # Note that the arguments are not packed into *args and **kwargs as in the golden function!!!
# def postprocess_golden_function_outputs(args, kwargs, output):
#     # i.e.
#     ttnn_input_tensor = args[0]
#     torch_output_tensor = outputs[0]
#     return ttnn.from_torch(torch_output_tensor, dtype=ttnn_input_tensor.dtype, device=ttnn_input_tensor.device)

# ttnn.attach_golden_function(
#     ttnn.prim.example,
#     golden_function=golden_function,
#     preprocess_golden_function_inputs=preprocess_golden_function_inputs, # Optional
#     postprocess_golden_function_outputs=postprocess_golden_function_outputs # Optional
# )


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
elements_per_tile = 1024
elements_per_tile_codebook = 256

# --- Create random numpy arrays following your logic ---
rng = np.random.default_rng()

# colIdx in [0, 15]
colIdx_np = rng.integers(low=0, high=16, size=(num_col_tiles, elements_per_tile), dtype=np.uint8)

# rowIdx in [0, 3]
rowIdx_np = rng.integers(low=0, high=4, size=(num_row_tiles, elements_per_tile), dtype=np.uint8)

codebook_np = np.arange(0, elements_per_tile_codebook * num_codebook_tiles, dtype=np.uint8)
codebook_np = codebook_np.reshape(num_codebook_tiles, elements_per_tile_codebook)

# --- Convert to torch tensors ---
rowIdx_torch = torch.tensor(rowIdx_np, dtype=torch.float32)
codeBook_torch = torch.tensor(codebook_np, dtype=torch.float32)
colIdx_torch = torch.tensor(colIdx_np, dtype=torch.float32)

# --- Move to device ---
rowIdx_tt = ttnn.from_torch(rowIdx_torch, dtype=ttnn.uint8, layout=ttnn.ROw, device=device)
codeBook_tt = ttnn.from_torch(codeBook_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
colIdx_tt = ttnn.from_torch(colIdx_torch, dtype=ttnn.uint8, layout=ttnn.TILE_LAYOUT, device=device)

# --- Run your example op ---
output_tensor = ttnn.prim.example(rowIdx_tt, rowIdx_tt, rowIdx_tt)

# --- Copy back to host ---
torch_output = ttnn.to_torch(output_tensor)

print("Output tensor:")
print(torch_output)

# --- Cleanup ---
ttnn.close_device(device)

