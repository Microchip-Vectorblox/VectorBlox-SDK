# VectorBlox Compression

VectorBlox supports three compression configurations: no compression (ncomp), compression (comp), and unstructured compression (ucomp).  

| Configuration            | `vnnx_compile` Argument | Supported Kits                                                                       | Sparsity Pattern                                 | Retraining Required?   | Supported HW Sizes | 
| ------------------------ | ----------------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------- | --------------------- | ------------------ | 
| No Compression           | `-c ncomp`        | PolarFire SoC Video Kit,  PolarFire Non SoC Video Kit, Discovery Kit, and Icicle Kit | None (dense weights)                                        | No                    | V1000, V500, V250  |
| Compression              | `-c comp`         | PolarFire SoC Video Kit, Discovery Kit, and Icicle Kit                               | Structured (2:4, 1:4) and weight pairing                    | Yes | V1000, V500, V250  |
| Unstructured Compression | `-c ucomp`        | PolarFire SoC Video Kit                                                              | Any pattern (structured or unstructured) and weight pairing | Yes | V1000, V500        |


## Important Considerations

- The core configuration **must** match the compression and size configuration used in `vnnx_compile`. A binary file of a model compiled for one configuration will not run on a core built for another. **Reference the vnnx_compile section in the [SDK Programmer's](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/docs/VectorBloxPG.md) guide for more information on how to compile a model that has been trained with compression**.
- While using compression and unstructured compression does improve model runtime, you will have to retrain your model for compression.
- **If the model has not been trained for compression please compile the model with no compression**. Compiling with `comp` or `ucomp` will not improve performance unless the model has been retrained with compression.
- Please note that `ucomp` stands for unstructured compression and not uncompressed. To compile a model without compression, use the `ncomp` arg, which stands for no compression.

## How Compression Works

Compression reduces the number of multiplications needed in two ways:

- **Pruning** (also known as sparsity) -  sets weights to zero. The accelerator can skip those multiplications because their result is always zero. This improves model runtime.
- **Pairing** -  gives two weights the same magnitude. The accelerator adds or subtracts the two matching activations first and then does one multiplication instead of two, which makes a multiply-and-accumulate into a cheaper addition or subtraction, thus reducing runtime.

**Compiling with compression only speeds up a model whose weights are already sparse or paired**. A model trained with no compression has very few zero or matching weights, so it gets little or no benefit when compiling a model with compression.  

## No Compression

No compression is the default configuration for VectorBlox, and the SDK tutorials use it. Weights are stored and processed densely, and every multiplication is performed. If new to VectorBlox, we recommend sticking with no compression to avoid having to retrain the model with pruning and pairing.  

When to use no compression

- if you do not want to retrain your model with pruning and pairing
- if low power and FPGA resource utilization is a priority

## Compression

Structured sparsity and pairing are accelerated by compression. At compile time, the convolution weights for every block of eight (8) input channels are separated into individual blocks. The compiler then selects the maximum level of compression that will allow each block in the layer to be compressed.

| Compression level | Weights stored per block of 8 | Weight reduction | Typical pattern                            |
| ----------------- | ----------------------------- | ---------------- | ------------------------------------------ |
| 4x                | 2                             | 75%              | 1:4 sparsity, or 2:4 sparsity with pairing |
| 2x                | 4                             | 50%              | 2:4 semi-structured sparsity                 |
| 1x                | 8                             | 0%               | Dense (No Compression)         |

The level is decided per layer, which means if there is a single dense block, that layer will be set to a lower level. Input channels are padded with zeros until they reach a multiple of eight. Fully connected layers are turned into 1x1 convolutions so they can be compressed too. Overall compression is therefore between 2x and 4x when all layers are compressed. Examples of models trained with compression can be found in the compressed tutorials [here](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/tree/master/tutorials/compressed).

## Unstructured Compression

Unstructured compression accelerates any sparsity pattern and weight pairing. Zero weights don't need to be grouped into blocks (unstructured sparsity), so models can be more sparse. While unstructured compression has high performance, this comes at the cost of complexity, increased FPGA power usage, and resource utilization. Examples of models trained with unstructured compression can be found in the unstructured compressed tutorials [here](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/tree/master/tutorials/unstructure_compressed).

As with compression, the model will need to be retrained with the compression library to get performance improvements.


### Unstructured Compression Limitations

- `V1000` and `V500` are the only supported `SIZE` configurations.
- It only supports the cut version of the models using `tflite_cut` to move post-processing to the host application instead of running in the `VBX CORE`.
- Some layers have additional restrictions under unstructured compression. See [Unstructured Compression (UCOMP) Layer Limitations](./OPS.md#unstructured-compression-ucomp-layer-limitations) in the supported operations document.

## Training a Compressed Model

The VectorBlox Training Compression Library (`compress.py`) applies pruning and pairing to a PyTorch model during training. Copy `compress.py` and import it into your project. Please refer to the [VectorBlox Training Compression Library](https://github.com/Microchip-Vectorblox/VectorBlox-Compression) for more information on retraining for compression and for the following tools.

The recommended workflow is:

1. Train the model as usual with no compression
2. Apply compression to the trained model by retraining with the compression library
3. As we do in the tutorials, export the model to INT8 TF Lite and use `vnnx_compile` to compile the model with the matching compression setting

Training for Compression:

```python
import compress
compress.compress_vbx3(model, compression=compression_rate)
```

`compression_rate` should be between `0.5` and `0.75`.

Training for Unstructured Compression:

```python
import compress
compress.compress_vbx3_unstructured(model, compression=compression_rate)
```

`compression_rate` must be between `0.5` and `1.0`.