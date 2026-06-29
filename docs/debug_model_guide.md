# VectorBlox Debug Model Guide

The VectorBlox Accelerator executes INT8-based models, and the VectorBlox SDK accepts quantized INT8 TFLite models as input. The SDK provides a suite of tools for inspecting, verifying, and modifying both TFLite files and VectorBlox's proprietary VNNX file format. The "VectorBlox SDK Tools" section of this document describes the available tools, while the "Debugging with Python Inference Scripts" section outlines the recommended workflow for generating VNNX files for deployment on the VectorBlox Accelerator.

## Table of Contents

- [VectorBlox SDK Tools](#vectorblox-sdk-tools)
  - [tflite\_verify](#tflite_verify)
  - [tflite\_compare](#tflite_compare)
  - [tflite\_infer](#tflite_infer)
  - [tflite\_cut](#tflite_cut)
  - [vnnx\_infer](#vnnx_infer)
  - [vnnx\_compile with -d flag](#vnnx_compile-with--d-flag)
- [Debugging with Python Inference Scripts (Python Simulation)](#debugging-with-python-inference-scripts-python-simulation)
  - [FP32 ONNX/OpenVINO](#fp32-onnxopenvino)
  - [FP32 TFLITE](#fp32-tflite)
  - [INT8 TFLITE](#int8-tflite)
  - [INT8 TFLITE w/ preprocessing](#int8-tflite-w-preprocessing)
  - [Transformed TFLITE](#transformed-tflite)
  - [VNNX Binary (simulator)](#vnnx-binary-simulator)

## VectorBlox SDK Tools

### tflite_verify

Ensures that all layer operators in a `.tflite` model are supported by the VectorBlox SDK. See [OPS.md](OPS.md) for the full list of supported layers.

Use Case: checking compatibility of TFLite layers before vnnx_compile, especially when compiling a custom model or when `vnnx_compile` fails with an unexpected operator error.

```bash
# Basic check — verifies INT8 operator types and parameters
tflite_verify onnx_resnet18-v1.tflite

# Check after transformation passes have been applied (-t)
tflite_verify onnx_resnet18-v1.pre.tr.tflite -t
```

Upon successful execution of `tflite_verify` the terminal should return nothing. If an unsupported operator is encountered, `tflite_verify` will display an error.

### tflite_compare

Splits a `.tflite` into single-layer subgraphs, compiles each to `.vnnx`, and compares the INT8 outputs between the TFLite interpreter and the VectorBlox simulator. Reports the per-layer error rate (% of values differing by more than 1) and saves a heatmap `.npy` for any layer that exceeds the threshold.

Use Case: locating the first layer where quantization errors accumulate between TFLite and the compiled VNNX output.

```bash
# Compare tflite vs V1000 No Compression vnnx outputs
tflite_compare onnx_resnet18-v1.tflite -s V1000 -c ncomp
```

**Interpreting the output.** `tflite_compare` produces one line per layer that looks roughly like:

```text
Layer 12 (CONV_2D): %error = 4.20%  max_diff = 1  mean_diff = 0.04  -> OK
Layer 27 (ADD):     %error = 0.15%  max_diff = 7  mean_diff = 0.31  -> CHECK heatmap.27.npy
```

Two numbers need to be read *together*:

- **`%error`** — the percentage of output values that differ by more than 1 between the TFLite interpreter and the VectorBlox simulator. Because both sides are INT8, a single-step rounding difference is counted here, so this number can look alarmingly high on otherwise healthy layers.
- **`max_diff`** — the largest absolute INT8 difference observed at that layer. This is the better indicator of real divergence.

A high `%error` with a low `max_diff` (e.g. `%error = 40%`, `max_diff = 1`) means the two backends simply rounded differently on many values — **this is not a bug**. A low `%error` with a high `max_diff` (e.g. `%error = 0.2%`, `max_diff = 20`) is the pattern that indicates an actual numerical problem and is worth investigating via the saved `heatmap.N.npy`.

The command prints one row per layer in the model, so the log can be long. Redirect to a file (`tflite_compare ... | tee compare.log`) and search for layers with large `max_diff` values rather than scanning every line.

### tflite_infer

Runs inference on a `.tflite` model (or directory of subgraph `.tflite` files) and saves each layer's input and output activations as `.npy` files under `input0/`. Optionally compares subgraph outputs against the corresponding VNNX subgraph outputs using `--vnnx`.

Use Case: capturing intermediate activations for manual inspection, or doing a layer-by-layer tflite vs vnnx diff with custom input images.

```bash
# Run inference and save activations to input0/
tflite_infer onnx_resnet18-v1.tflite -i $VBX_SDK/tutorials/test_images/oreo.jpg

# Compare tflite subgraph outputs vs vnnx subgraph outputs
tflite_infer subgraphs/ -i $VBX_SDK/tutorials/test_images/oreo.jpg --vnnx
```

**Interpreting the output.** Without `--vnnx`, the command runs silently and populates an `input0/` directory with one `.npy` file per layer (for example `input0/layer_0_input.npy`, `input0/layer_0_output.npy`, ...) which can be loaded with `numpy.load()` for manual inspection. With `--vnnx`, it additionally prints one line per subgraph comparing the TFLite reference to the VectorBlox simulator output, for example:

```text
subgraph 0: max_diff = 0   mean_diff = 0.00   MATCH
subgraph 1: max_diff = 2   mean_diff = 0.01   MATCH
subgraph 2: max_diff = 18  mean_diff = 0.42   MISMATCH -> see input0/subgraph_2_diff.npy
```

As with `tflite_compare`, `max_diff` is the most reliable indicator — a `MATCH` means the two backends agree within the expected INT8 rounding tolerance, while a `MISMATCH` points to a specific subgraph whose saved `.npy` diff is worth loading to see where the divergence occurs.

### tflite_cut

Cuts a `.tflite` model at one or more operator indices, producing numbered subgraph files (`model.0.tflite`, `model.1.tflite`, ...) in the same directory. Each subgraph contains the operators up to and including the specified cut point, with shared predecessor operators merged automatically.

Use Case: This tool can be used for debugging when isolating a specific section of the graph to compile and test independently, or when narrowing down which group of layers is causing an issue (sometimes it's a series of layers).

```bash
# Cut the model at operator index 10, producing two subgraphs
tflite_cut onnx_resnet18-v1.tflite -c 10

# Cut at multiple indices
tflite_cut onnx_resnet18-v1.tflite -c 10 20
```

### vnnx_infer

Simulates a compiled `.vnnx` binary using the VectorBlox software simulator. Prints memory and performance estimates: `DATA_BYTES`, `ALLOCATE_BYTES`, `DMA_BYTES`, `INSTR_CYCLES`, and a `CHECKSUM` of the output with a test input. With `-d`, additionally dumps input/output `.npy` files, a per-channel `heatmap.N.npy` diff against the embedded test vectors, and an `io.json` summary.

The reported metrics are:

- **`DATA_BYTES`** — Size of the static model data (weights, biases, and compiled instructions) that must be stored in memory for the model. Reported in MB.
- **`ALLOCATE_BYTES`** — Total memory that must be allocated at runtime. This includes `DATA_BYTES` plus the scratch/activation buffers needed during inference, so it is always greater than or equal to `DATA_BYTES`. Reported in MB.
- **`DMA_BYTES`** — Total bytes transferred over DMA during a single inference. Reported in raw bytes (not MB). Useful for estimating memory bandwidth pressure.
- **`INSTR_CYCLES`** — Estimated number of hardware clock cycles required to execute the model on the CoreVectorBlox accelerator. Divide by the FPGA clock frequency to approximate inference latency.
- **`CHECKSUM`** — A hash of the output tensor produced from the embedded test input. The same `.vnnx` binary should yield the same checksum in both simulation and on hardware.

Use Case: confirming a compiled model runs correctly in simulation, checking estimated hardware performance, or **verifying output checksums match between simulation and hardware**. This can also be done with the C Simulation command that can be found in our tutorials.

```bash
# Simulate and print memory/cycle estimates
vnnx_infer onnx_resnet18-v1_V1000_ncomp.vnnx
# Output:
#   DATA_BYTES = 13.36 MB
#   ALLOCATE_BYTES = 15.92 MB
#   DMA_BYTES = 7698876
#   INSTR_CYCLES = 846293
#   CHECKSUM = 5935dc4a
```

### vnnx_compile with -d flag

Compiles a `.tflite` to `.vnnx` in debug mode, preserving all intermediate files in a `temp/` subdirectory. The most useful artifact is `model.tr.tflite`. Here you can see whether the transformed TFLite graph is problematic (you can do this with a Python simulation, as shown below) or with vnnx_compile. If a bug with vnnx_compile is noticed, contact the VectorBlox team at <vectorblox@microchip.com>.

Use Case: investigating unexpected compilation failures and verifying the transformed tflite is correct.

```bash
# Compile with debug mode — produces both .vnnx and .pre.tr.tflite
vnnx_compile -s V1000 -c ncomp -t onnx_resnet18-v1.pre.tflite -o onnx_resnet18-v1_V1000_ncomp.vnnx -d

# Verify the transformed graph is working properly with python simulation
python $VBX_SDK/example/python/classifier.py onnx_resnet18-v1.pre.tr.tflite $VBX_SDK/tutorials/test_images/oreo.jpg 
```

## Debugging with Python Inference Scripts (Python Simulation)

To execute a model on the VectorBlox Accelerator, it must first be converted to a quantized .tflite format and then compiled into a .vnnx format. Depending on the starting format, the trained model undergoes several format conversions. It is recommended to verify each conversion step using Python simulation and to inspect the model in Netron (<https://netron.app/>).  

The following sections illustrate the suggested workflow using the resnet18 (<https://github.com/Microchip-Vectorblox/VectorBlox-SDK/tree/release-v2.0.3/tutorials/onnx/onnx_resnet18-v1>) tutorial. The simulation command can be used after each step.  

### FP32 ONNX/OpenVINO

Before conversion, if using ONNX or OpenVINO, the model can be run to verify that image preprocessing and post-processing are working as expected.

```bash
python $VBX_SDK/example/python/classifier.py resnet18-v1-7.onnx $VBX_SDK/tutorials/test_images/oreo.jpg --scale 255.
```

### FP32 TFLITE

If using `onnx2tf` or `openvino2tensorflow`, the FP32 version of the TFLITE can be checked to ensure TFLITE conversion was successful.

```bash
python $VBX_SDK/example/python/classifier.py saved_model/resnet18-v1-7_float32.tflite $VBX_SDK/tutorials/test_images/oreo.jpg --scale 255.
```

### INT8 TFLITE

Next, the INT8 version of the TFLITE can be checked to ensure TFLITE quantization was successful.

```bash
python $VBX_SDK/example/python/classifier.py onnx_resnet18-v1.tflite $VBX_SDK/tutorials/test_images/oreo.jpg --scale 255.
```

### INT8 TFLITE w/ preprocessing

If preprocessing is included in the model, the `.pre.tflite` file can be tested on uint8 images.

```bash
python $VBX_SDK/example/python/classifier.py onnx_resnet18-v1.pre.tflite $VBX_SDK/tutorials/test_images/oreo.jpg 
```

### Transformed TFLITE

By adding the debug flag `-d` to `vnnx_compile`, the `.tr.tflite` can be checked to ensure optimization was successful.

```bash
vnnx_compile -s V1000 -c ncomp -t onnx_resnet18-v1.pre.tflite -o onnx_resnet18-v1_V1000_ncomp.vnnx -d
python $VBX_SDK/example/python/classifier.py onnx_resnet18-v1.pre.tr.tflite $VBX_SDK/tutorials/test_images/oreo.jpg 
```

### VNNX Binary (simulator)

The `.vnnx` binary can be simulated (the last step in the tutorial).

```bash
python $VBX_SDK/example/python/classifier.py onnx_resnet18-v1_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
```
