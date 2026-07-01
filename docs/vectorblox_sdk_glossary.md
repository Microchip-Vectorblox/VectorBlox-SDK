# VectorBlox SDK Glossary

A comprehensive technical reference compiling all key terms, definitions, and acronyms encountered throughout the VectorBlox SDK documentation and its associated tools. Intended to serve as a centralized resource for developers seeking precise and consistent terminology.

## General Acronyms

| Term | Meaning |
| --- | --- |
| AI/ML | Artificial Intelligence / Machine Learning |
| CLI | Command Line Interface |
| CNN | Convolutional Neural Network |
| DL | Deep Learning |
| DDR | Double Data Rate (SDRAM memory) |
| FPS | Frames Per Second |
| FPGA | Field Programmable Gate Array |
| IR | Intermediate Representation |
| IP | Intellectual Property |
| MSS | Microprocessor Sub System |
| MXP | Matrix Processor |
| NN | Neural Network |
| NMS | Non-Maximum Suppression |
| ONNX | Open Neural Network Exchange |
| PB | TensorFlow™ Protobuf™ Model Format |
| ReLU | Rectified Linear Unit |
| RISC-V® | Reduced Instruction Set Computer – Fifth generation |
| SoC | System on Chip |
| TFLite | TensorFlow Lite (now LiteRT) |
| VGG16 | Visual Geometry Group (network with 16 CNN layers) |
| XML | eXtensible Markup Language (used by OpenVINO™ model format) |
| YOLO | You Only Look Once |

## VectorBlox-Specific Acronyms

| Term | Meaning |
| --- | --- |
| BLOB | Binary Large Object |
| COMP | Compression (structured sparsity and weight pairing) |
| CoreVectorBlox | Microchip hardware CNN accelerator IP for FPGA |
| FIA | Fabric Inference Accelerator (CNN accelerator using FPGA fabric) |
| NCOMP | No Compression |
| TSNP | Tiling Sequential Neural Processor (underlying unstructured compression IP for FPGA) |
| UCOMP | Unstructured Compression (unstructured sparsity and weight pairing) |
| VNNX | VectorBlox™ SDK Neural Network compiled object (binary executable) |


## Tools & Process Acronyms

| Term | Meaning |
| --- | --- |
| MO | Model Optimizer (OpenVINO tool) |
| omz | OpenVINO Model Zoo downloader |
| PINTO | Collection of ONNX/TensorFlow conversion tools |
| venv | Python virtual environment |
| vbx_env | VectorBlox Python environment (activated via `setup_vars.sh`) |

## Definitions

### Compression & Quantization

| Term | Description |
| --- | --- |
| VNNX & BLOB | Compiled binary files produced by the VectorBlox SDK that execute on FPGA hardware. These terms are interchangeable. |
| Quantization | Converting a floating-point (FP32) neural network to fixed-point (INT8) format, reducing model size and improving inference speed with minimal accuracy loss. |
| Calibration | The process of collecting statistics from representative data to determine optimal scaling factors for quantization. |
| No Compression (NCOMP) | Configuration with no acceleration for sparsity or pairing. Recommended for models not trained with sparsity. Generally produces smaller binaries. |
| Compression (COMP) | Configuration with special acceleration for structured sparsity and weight pairing. Requires models trained with the VectorBlox Training Compression Library or 2:4 structured sparsity for maximum acceleration. |
| Unstructured Compression (UCOMP) | Configuration accelerating any structured or unstructured sparsity pattern. Enables highest network acceleration with best performance/accuracy trade-off. Requires models trained with the VectorBlox Training Compression Library for maximum acceleration. |
| Structured Sparsity | Weight pruning pattern where zeros follow a regular structure (e.g., 2:4 means 2 non-zero values per 4 weights). |
| 2:4 Structured Sparsity | Specific sparsity pattern with exactly 2 non-zero weights per group of 4, supported by all compression configurations. |

### Graph & Layer Processing

| Term | Description |
| --- | --- |
| Preprocessing | Adding normalization and quantization layers to the beginning of a TensorFlow Lite model via `tflite_preprocess`. Typically includes pixel scaling and uint8-to-int8 conversion. |
| Postprocessing | Adding model-specific output decoding layers (bounding boxes, keypoints, class predictions) after inference, handled by C functions like `post_process_ssd_torch_int8()`. |
| Graph Transformation | Internal compiler optimization passes that convert TensorFlow Lite operators to VectorBlox hardware operations. Produces a `.tr.tflite` intermediate file. |
| Layer Fusion | Compiler optimization combining multiple TensorFlow Lite operators into single fused operations (e.g., Conv2D + Activation). |
| Subgraph | A portion of a neural network graph, often a single operator or a chain of operators. Used for debugging and layer-by-layer verification. |
| Activation Function | Non-linear function applied after operations (RELU, RELU6, GELU, Sigmoid, Softmax, Tanh, SiLU, Hard Swish, etc.). |

### Inference & Simulation

| Term | Description |
| --- | --- |
| Inference | Running a neural network model on input data to produce predictions. |
| Software Simulator | Bit-accurate simulation of VectorBlox hardware inference in C or Python, producing identical results to physical hardware. |
| Python Simulation | Running inference via the `vbx.sim` package or classifier/detector example scripts. |
| C Simulation | Running compiled VNNX models using the C software simulator (no hardware required). |
| Bit-Accurate | Simulation produces mathematically identical outputs to physical hardware, down to individual bit values. |
| Test Vectors | Embedded reference input/output data compiled into VNNX binaries for verification. |
| Heatmap | Visualization (`.npy` file) showing per-layer or per-channel quantization error differences between TensorFlow Lite and VectorBlox outputs. |

### Operations & Tasks

| Term | Description |
| --- | --- |
| Image Classification | Task predicting one class label per input image (e.g., "dog", "cat"). |
| Object Detection | Task detecting multiple objects in an image, returning bounding boxes and class labels. |
| Face Detection | Task detecting human faces in images, returning bounding boxes and facial landmarks. |
| Pose Estimation | Task detecting human body keypoints (joints) and their connectivity. |

### Model Preparation

| Term | Description |
| --- | --- |
| Model Zoo | Collection of pre-trained models (e.g., OpenVINO Model Zoo, MediaPipe model collection). |
| Model Optimizer | OpenVINO tool (`mo`) converting and optimizing models to OpenVINO IR format. |
| Dropout | Training-time layer removed during inference optimization (model optimizer removes these). |
| INT8 TensorFlow Lite | Quantized model format ready for VectorBlox compilation (end goal of the model preparation pipeline). |
