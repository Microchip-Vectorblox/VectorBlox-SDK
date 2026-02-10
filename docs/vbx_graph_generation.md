# VBX Graph Generation, vnnx_compile

This section describes the process of converting the TF Lite model generated from the commands in the sdk_toolkit.md document into a binary file. The vnnx_compile command is available from the command line within the VBX Python environment. For more details on compression and size configurations, please refer to the CoreVectorBlox v3.0 IP Handbook.

You may embed a test image in the model to quickly test inference and post-processing with the following command:

```vnnx_compile  -s  SIZE_CONFIG  -c COMPRESSION_CONFIG  -t  MODEL_NAME.tflite  -i  IMAGE_NAME.jpg```

## Compression and No Compression
The TF Lite model is processed by the VectorBlox binary file generation tool, which creates a quantized binary file that can run on both hardware and the simulator. To use the tool, provide the input TF Lite model and hardware size configuration (V250, V500, or V1000). If you use custom pre-processing or post-processing code, you can specify the starting and ending nodes for the VNNX graph. By default, tutorials use V1000 with no compression.

Display usage:

```
usage: vnnx_compile [-h] -t TFLITE -s {V250,V500,V1000} -c {ncomp,comp,ucomp} [-o OUTPUT] [--start_layer START_LAYER] [-e END_LAYER] [-i [INPUTS ...]] [-m MEAN [MEAN ...]] [-sc SCALE [SCALE ...]] [-b] [-u]

options:
  -h, --help            show this help message and exit
  -t TFLITE, --tflite TFLITE
                        tflite I8 model description (.tflite)
  -s {V250,V500,V1000}, --size_conf {V250,V500,V1000}
                        size configuration to build model for
  -c {ncomp,comp,ucomp}, --compression-vbx {ncomp,comp,ucomp}
                        compression setting for VNNX model generation
  -o OUTPUT, --output OUTPUT
                        Name of vnnx output file
  --start_layer START_LAYER
  -e END_LAYER, --end_layer END_LAYER
  -i [INPUTS ...], --inputs [INPUTS ...]
                        provide test inputs for model
  -m MEAN [MEAN ...], --mean MEAN [MEAN ...]
  -sc SCALE [SCALE ...], --scale SCALE [SCALE ...]
  -b, --bgr
  -u, --uint8 
            uint8 can only be used with the ucomp compression config
```
### No Compression

The following example shows how to generate a binary file from a TF Lite model for the V1000 with the no compression configuration:

```vnnx_compile  -s  V1000  -c ncomp  -t  MODEL_NAME.tflite```

The binary output file name for V1000 with no compression will be ```MODEL_NAME_ V1000_ncomp.vnnx```

### Compression

The following example shows how to generate a binary file from a TF Lite model for the V1000 with the compression configuration:

```vnnx_compile  -s  V1000  -c comp  -t  MODEL_NAME.tflite```

The binary output file name for V1000 with compression will be ```MODEL_NAME_V1000_comp.vnnx```

## Unstructured Compression 

For unstructured compression, V1000 is the only size configuration that is currently supported. Provide the input TF Lite model with -t; pass the --uint8 flag to convert to signed int8 if needed; and use ```-c ucomp``` specify unstructured compression.

**Note:** The --uint8 flag can only be used when generating a binary file with unstructured compression.

The following example demonstrates how to generate a binary file from a TF Lite model for the V1000 and unstructured compression configuration:

```vnnx_compile -s  V1000 -c ucomp  -t  MODEL_NAME.tflite  --uint8```

The binary output file name for V1000 with compression will be ```MODEL_NAME.ucomp```