# Supported Operations (INT8 TFLite)

## Overview

This document lists all TensorFlow Lite operators supported by the VectorBlox VNNX compiler for INT8 quantized inference. These operators are optimized for deployment on VectorBlox hardware accelerators.

---

## Quick Summary

| Category                     | Count  | Details                            |
| ---------------------------- | ------ | ---------------------------------- |
| Fully Supported              | 40     | No constraints or limitations      |
| Supported (with constraints) | 28     | Check parameter restrictions below |
| **Total Operators**          | **68** | Ready for INT8/UINT8 inference     |

---

## Key Points

- **Operator parameters matter** - operators marked with a constraint only work with specific parameter values (e.g., axis values, padding modes).
- **Troubleshooting**: If an operator is unsupported, contact the VectorBlox team at <vectorblox@microchip.com>.

---

## Complete Operator List

| Operators               | Known Limitations                                                                                                                                       |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ABS                     |                                                                                                                                                         |
| ADD                     | **Fused activation function** = [0,NONE,RELU,RELU6],                                                                                                    |
| ARG_MAX                 | **Axis** = [-1],                                                                                                                                        |
| ARG_MIN                 | **Axis** = [-1],                                                                                                                                        |
| AVERAGE_POOL_2D         | **Fused activation function** = [0,NONE,RELU,RELU6],  **Padding** = [SAME,VALID],                                                                       |
| BATCH_MATMUL            |                                                                                                                                                         |
| CAST                    |                                                                                                                                                         |
| CONCATENATION           | **Axis** = [-4,-3,-2,-1],  **Fused activation function** = [0,NONE,RELU,RELU6],  **Others**: The number of dimensions of the input tensors must match., |
| CONV_2D                 | **Fused activation function** = [0,NONE,RELU,RELU6],  **Padding** = [SAME,VALID],                                                                       |
| DEPTHWISE_CONV_2D       | **Fused activation function** = [0,NONE,RELU,RELU6],  **Padding** = [SAME,VALID],                                                                       |
| DEQUANTIZE              |                                                                                                                                                         |
| DILATE                  |                                                                                                                                                         |
| DIV                     | **Fused activation function** = [0,NONE,RELU,RELU6],  **Others**: Input 2 must be a constant,                                                           |
| ELU                     |                                                                                                                                                         |
| EQUAL                   |                                                                                                                                                         |
| EXP                     |                                                                                                                                                         |
| EXPAND_DIMS             | **Axis** = [-4,-3,-2,-1],                                                                                                                               |
| FULLY_CONNECTED         | **Fused activation function** = [0,NONE,RELU,RELU6,HARD_SWISH,SWISH,LOGISTIC],                                                                          |
| GATHER                  | **Axis** = [-4,-3,-2,-1],  **Others**: Indices (second input) must be an integer tensor of any dimension,                                               |
| GELU                    |                                                                                                                                                         |
| GREATER                 | **Axis** = [-4,-3,-2,-1],                                                                                                                               |
| GREATER_EQUAL           |                                                                                                                                                         |
| HARD_SWISH              |                                                                                                                                                         |
| LEAKY_RELU              |                                                                                                                                                         |
| LESS                    |                                                                                                                                                         |
| LESS_EQUAL              |                                                                                                                                                         |
| LOG                     |                                                                                                                                                         |
| LOGISTIC                |                                                                                                                                                         |
| MAXIMUM                 |                                                                                                                                                         |
| MAX_POOL_2D             | **Fused activation function** = [0,NONE,RELU,RELU6],  **Padding** = [SAME,VALID],                                                                       |
| MEAN                    |                                                                                                                                                         |
| MINIMUM                 |                                                                                                                                                         |
| MIRROR_PAD              | **Mode** = [REFLECT,SYMMETRIC],                                                                                                                         |
| MUL                     | **Fused activation function** = [0,NONE,RELU,RELU6],                                                                                                    |
| NEG                     |                                                                                                                                                         |
| NOT_EQUAL               |                                                                                                                                                         |
| PACK                    | **Axis** = [-4,-3,-2,-1],                                                                                                                               |
| PAD                     |                                                                                                                                                         |
| PADV2                   |                                                                                                                                                         |
| POW                     |                                                                                                                                                         |
| PRELU                   |                                                                                                                                                         |
| QUANTIZE                |                                                                                                                                                         |
| REDUCE_MAX              | **Axis** = [-4,-3,-2,-1],                                                                                                                               |
| REDUCE_MIN              | **Axis** = [-4,-3,-2,-1],                                                                                                                               |
| REDUCE_PROD             | **Axis** = [-4,-3,-2,-1],                                                                                                                               |
| RELU                    |                                                                                                                                                         |
| RELU6                   |                                                                                                                                                         |
| RELU_0_TO_1             |                                                                                                                                                         |
| RELU_N1_TO_1            |                                                                                                                                                         |
| RESHAPE                 | **Others**: Reshape with Join and separate on axis -1, -2 and -3. Multiple axis reshape is not supported.,                                              |
| RESIZE_BILINEAR         |                                                                                                                                                         |
| RESIZE_NEAREST_NEIGHBOR |                                                                                                                                                         |
| RSQRT                   |                                                                                                                                                         |
| SILU                    |                                                                                                                                                         |
| SLICE                   |                                                                                                                                                         |
| SOFTMAX                 | **Dim** = [-3,-2,-1],                                                                                                                                   |
| SPLIT                   | **Axis** = [-4,-3,-2,-1],                                                                                                                               |
| SPLIT_V                 | **Axis** = [-4,-3,-2,-1],                                                                                                                               |
| SQUARED_DIFFERENCE      |                                                                                                                                                         |
| SQUEEZE                 | **Axis** = [-4,-3,-2,-1],                                                                                                                               |
| STRIDED_SLICE           |                                                                                                                                                         |
| SUB                     | **Fused activation function** = [0,NONE,RELU],                                                                                                          |
| SUM                     | **Axis** = [-4,-3,-2,-1],                                                                                                                               |
| TANH                    |                                                                                                                                                         |
| TILE                    |                                                                                                                                                         |
| TRANSPOSE               |                                                                                                                                                         |
| TRANSPOSE_CONV          | **Fused activation function** = [0,NONE,RELU,RELU6],  **Padding** = [SAME,VALID],                                                                       |
| UNPACK                  | **Axis** = [-4,-3,-2,-1],                                                                                                                               |

---

## Unstructured Compression (UCOMP) Layer Limitations

The following additional restrictions apply when compiling a model with unstructured compression (`-c ucomp`). See [VectorBlox Compression](./vectorblox_compression.md) for more information on unstructured compression.

- `CONV_2D` and `MAX_POOL_2D` kernels must be at most 9x9, symmetric and have a stride of 1 or 2.
- `STRIDED_SLICE` supports only the channel dimension with stride 1.
- `CONV_2D` layer with a 3x3 kernel and stride 2, or a 4x4 or 5x5 kernel, needs an input of at least 8x8 (HxW), and a kernel larger than 5x5 in either dimension needs an input of at least 16x16 (HxW).
- `CONV_2D` with dense weights whose input and output channels are both large may lose accuracy. When the convolution is split along its input channels, the partial results are requantized to int8 with a bounded headroom, so accumulator values beyond that range saturate.
- `CONV_2D` with asymmetric padding at the bottom and right isn't supported.
- `CONV_2D` is only supported for 4D input tensors. A 3‑D shape like HxWxC is not supported.
- `BATCH_MATMUL` is currently not supported for unstructured compression.
- If the first layer of the model is a convolution with stride 2, or stride 1 with a kernel of at least 5x5, the input height and width (HxW) must each be a multiple of 4.
