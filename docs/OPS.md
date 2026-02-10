# Supported Operations (INT8 TFLite)

## Overview

This document lists all TensorFlow Lite operators supported by the VectorBlox VNNX compiler for INT8 quantized inference. These operators are optimized for deployment on VectorBlox hardware accelerators.

---

## Quick Summary

| Category | Count | Details |
|----------|-------|---------|
| Fully Supported | 38 | No constraints or limitations |
| Supported (with constraints) | 36 | Check parameter restrictions below |
| **Total Operators** | **74** | Ready for INT8/UINT8 inference |

---

## Key Points

- **Operator parameters matter** - operators marked with only work with specific parameter values (e.g., axis values, padding modes)
- **Troubleshooting**: If an operator is unsupported, Please contact the VectorBlox team at vectorblox@microchip.com

---

## Complete Operator List 


<div class="table">

| Operators | Known Limitations |
| ----------|-------------------|
|ABS||
|ADD|**Fused activation function** = [0,NONE,RELU,RELU6],  |
|ARG_MAX|**Axis** = [-1],  |
|ARG_MIN|**Axis** = [-1],  |
|AVERAGE_POOL_2D|**Fused activation function** = [0,NONE,RELU,RELU6],  **Padding** = [SAME,VALID],  |
|CONCATENATION|**Axis** = [-4,-3,-2,-1],  **Fused activation function** = [0,NONE,RELU,RELU6],  |
|CONV_2D|**Fused activation function** = [0,NONE,RELU,RELU6],  **Padding** = [SAME,VALID],  |
|DEPTHWISE_CONV_2D|**Fused activation function** = [0,NONE,RELU,RELU6],  **Padding** = [SAME,VALID],  |
|DEQUANTIZE||
|DIV|**Fused activation function** = [0,NONE,RELU,RELU6],  **Others**: Input 2 must be a constant,  |
|ELU||
|EQUAL||
|EXP||
|EXPAND_DIMS|**Axis** = [-4,-3,-2,-1],  |
|FULLY_CONNECTED|**Fused activation function** = [0,NONE,RELU,RELU6],  |
|GATHER|**Axis** = [-4,-3,-2,-1],  |
|GELU||
|GREATER|**Axis** = [-4,-3,-2,-1],  |
|GREATER_EQUAL||
|HARD_SWISH||
|LEAKY_RELU||
|LESS||
|LESS_EQUAL||
|LOG||
|LOGISTIC||
|MAXIMUM||
|MAX_POOL_2D|**Fused activation function** = [0,NONE,RELU,RELU6],  **Padding** = [SAME,VALID],  |
|MEAN||
|MINIMUM||
|MUL|**Fused activation function** = [0,NONE,RELU,RELU6],  |
|NEG||
|NOT_EQUAL||
|PACK|**Axis** = [-4,-3,-2,-1],  |
|PAD||
|PADV2||
|POW||
|PRELU||
|QUANTIZE||
|REDUCE_MAX|**Axis** = [-4,-3,-2,-1],  |
|REDUCE_MIN|**Axis** = [-4,-3,-2,-1],  |
|REDUCE_PROD|**Axis** = [-4,-3,-2,-1],  |
|RELU||
|RELU6||
|RELU_0_TO_1||
|RELU_N1_TO_1||
|RESHAPE||
|RESIZE_BILINEAR||
|RESIZE_NEAREST_NEIGHBOR||
|RSQRT||
|SILU||
|SLICE||
|SOFTMAX|**Dim** = [-3,-2,-1],  |
|SPLIT|**Axis** = [-4,-3,-2,-1],  |
|SPLIT_V|**Axis** = [-4,-3,-2,-1],  |
|SQUEEZE|**Axis** = [-4,-3,-2,-1],  |
|STRIDED_SLICE||
|SUB|**Fused activation function** = [0,NONE,RELU],  |
|SUM|**Axis** = [-4,-3,-2,-1],  |
|TANH||
|TILE||
|TRANSPOSE||
|TRANSPOSE_CONV|**Fused activation function** = [0,NONE,RELU,RELU6],  **Padding** = [SAME,VALID],  |
|UNPACK|**Axis** = [-4,-3,-2,-1],  |
|CAST|**Others**: Cast inputs from INT8 or UINT8 to INT32,  |
  
  
</div>
  
  
<style>   
  
</style>  
  
<div></div>

