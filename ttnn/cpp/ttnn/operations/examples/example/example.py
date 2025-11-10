import torch
import ttnn
import numpy as np

# ----------------------------
# Setup
# ----------------------------
device_id = 0
device = None

try:
    device = ttnn.open_device(device_id=device_id)
    print(f"Device {device_id} opened successfully.\n")

    # ----------------------------
    # Parameters
    # ----------------------------
    n_tiles_colIdx = 512
    n_tiles_rowIdx = 32
    n_tiles_codebook = 1
    elements_per_tile_colIdx = 1024
    elements_per_tile_rowIdx = 1024
    elements_per_tile_codebook = 256

    # ----------------------------
    # Random Input Generation
    # ----------------------------
    rng = np.random.default_rng()

    # colIdx_np = rng.integers(0, 16, size=(n_tiles_colIdx * elements_per_tile_colIdx,), dtype=np.uint8)
    colIdx_np = rng.integers(0, 16, size=(n_tiles_colIdx, elements_per_tile_colIdx), dtype=np.uint8)

    # rowIdx_np = rng.integers(0, 4, size=(n_tiles_rowIdx * elements_per_tile_rowIdx,), dtype=np.uint8)
    rowIdx_np = rng.integers(0, 4, size=(n_tiles_rowIdx, elements_per_tile_rowIdx), dtype=np.uint8)

    codeBook_np = np.arange(0, n_tiles_codebook * elements_per_tile_codebook, dtype=np.float32) % 256

    print(f"colIdx_data size (ele): {colIdx_np.size}")
    print(f"rowIdx_data size (ele): {rowIdx_np.size}")
    print(f"codeBook_data size (ele): {codeBook_np.size}")

    # print(f"colIdx first 16: {[int(v) for v in colIdx_np[:16]]}")
    # print(f"colIdx 1024: 1024+16: {[int(v) for v in colIdx_np[1024:1024+16]]}")

    flat = colIdx_np.ravel()
    print(f"colIdx first 16: {flat[:16].tolist()}")
    print(f"colIdx 1024–1040: {flat[1024:1040].tolist()}")
    flatRow = rowIdx_np.ravel()
    print(f"rowIdx first 16: {[int(v) for v in flatRow[:16]]}")

    print(f"codeBook first 16: {[float(v) for v in codeBook_np[:16]]}\n")

    # ----------------------------
    # Convert to Torch tensors
    # ----------------------------
    colIdx_torch = torch.tensor(colIdx_np, dtype=torch.uint8)
    rowIdx_torch = torch.tensor(rowIdx_np, dtype=torch.uint8)
    codeBook_torch = torch.tensor(codeBook_np, dtype=torch.bfloat16)

    # ----------------------------
    # Move to TT-NN device
    # ----------------------------
    rowIdx_tt = ttnn.from_torch(rowIdx_torch, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    colIdx_tt = ttnn.from_torch(colIdx_torch, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    codeBook_tt = ttnn.from_torch(codeBook_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    print("Tensors moved to TT-NN device.\n")
    print("rowIdx_tt:", rowIdx_tt)
    print("codeBook_tt:", codeBook_tt)
    print("colIdx_tt:", colIdx_tt, "\n----------------INPUTS END-----------------\n")
    # ----------------------------
    # Run the TT-NN op
    # ----------------------------
    output_tt = ttnn.prim.example(rowIdx_tt, codeBook_tt, colIdx_tt)
    # output_tt = ttnn.to_dtype(output_tt, ttnn.float16)
    print("\nOutput tensor on TT-NN device:")
    print(output_tt)
    # ----------------------------
    # Copy back to host
    # ----------------------------
    torch_output = ttnn.to_torch(output_tt)
    output_np = torch_output.cpu().to(torch.float32).numpy()
    # print("Output tensor:")
    # print(output_np)

    # ----------------------------
    # Validation (C++ equivalent)
    # ----------------------------
    eps = 1e-2
    pass_test = True
    count = 0

    # Flatten output tensor
    output_flat = output_np.flatten()
    colIdx_flat = colIdx_np.flatten()
    rowIdx_flat = rowIdx_np.flatten()

    # output_flat = output_np
    # colIdx_flat = colIdx_np

    print("Output colIdx_flat 0: 0+16:", [float(v) for v in colIdx_flat[0:16]])
    print("Output colIdx_flat 1024: 1024+16:", [float(v) for v in colIdx_flat[1024 : 1024 + 16]])
    # print(colIdx_flat)
    # num_rows, num_cols = colIdx_np.shape  # Get shape from colIdx

    print("\n---------------Validating results...---------------\n")

    count = 0
    pass_test = True

    for idx_b in range(len(output_flat)):
        rowidx = int(rowIdx_flat[idx_b // 16])
        colidx = int(colIdx_flat[idx_b])
        codebook_index = int(colidx + rowidx * 16)

        if codebook_index >= len(codeBook_np):
            print(f"[ERROR] codebook_index out of bounds: {codebook_index}")
            continue

        expected = float(codeBook_np[codebook_index])
        result = float(output_flat[idx_b])

        if abs(expected - result) > eps:
            pass_test = False
            if count < 5:
                print(f"Mismatch at idx_b {idx_b}: expected {expected:.2f}, got {result:.2f}")
                print(f"  rowidx={rowidx}, colidx={colidx}, codebook_index={codebook_index}")
                count += 1

    # Summary
    if pass_test:
        print("\nAll results matched expected output. TEST PASSED")
    else:
        print("\nSome mismatches found (see above). TEST FAILED")


finally:
    if device:
        ttnn.close_device(device)
        print("\nDevice closed.")
