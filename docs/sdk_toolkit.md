# SDK Toolkits
The following are the four tools used in the VectorBlox flow:
- OpenVINO's model zoo and optimizer
- PINTO's openvino2tensorflow
- PINTO's onnx2tf
- VectorBlox SDK tflite_quantize, tflite_preprocess, tflite_postprocess and tflite_cut

Depending on the source of user's model, the user may either use few of these tools, go directly to TF Lite conversion, or go directly to VNNX graph generation.

For converting to TF Lite, users need their own NumPy array files for TF Lite calibration. Tutorials that are provided in the SDK will download the necessary NumPy calibration data. Users can leverage VectorBlox SDK generate_npy tool to format their data into a NumPy array file.

Each of these VectorBlox flow tools are briefly described in the following sections along with links to their documentation. The tools are available from the command-line within the VBX Python environment.

## OpenVINO Model Zoo and Optimizer
The OpenVINO model zoo provides access to many popular neural network models, along with key information about preprocessing and performance. It also provides a tool to download models. The following is a short description of the command usage. The full documentation is available in [OpenVINO Toolkit](https://docs.openvino.ai/2025/index.html).

- Print all models available in the model zoo.
    ```
    (vbx env)~/sdk$ omz_downloader --print_all
    ```
- Download a model.
    ```
    (vbx env)~/sdk$ omz_downloader --name MODEL_NAME
    ```

The OpenVINO mo command is a tool to convert and optimize models for inference. Models from various source frameworks are converted to the OpenVINO Intermediate Representation (IR). The models are further optimized for inference by removing training-time layers like dropout and applying layer fusion. The following are the examples of the command’s usage. The full documentation is available in [OpenVINO Toolkit](https://docs.openvino.ai/2025/index.html).

- Convert and optimize a Caffe model without scale and mean values.
    ```
    (vbx env)~/sdk$ mo –framework caffe --input_model MODEL_NAME.caffe
    ```
- Convert and optimize a Caffe model with scale and mean values.
    ```
    (vbx env)~/sdk$ mo –framework caffe --input_model MODEL_NAME.caffe \
    --scale_values [127., 127., 127] --mean_values [64., 64., 64.]
    ```

**Note:** The documentation of the model must be reviewed to determine if there is pre-processing or if scale and mean values need to be included with this convert command.


## PINTO openvino2tensorflow
PINTO's openvino2tensorflow tool converts models in the OpenVINO IR into a TF Lite model. We can use it to keep inputs as batch Number, Channels, Height, Width (NCHW) dimension order or make direct changes to parameters and layers alongside the TF Lite conversion. The following is an example of the tool's usage. More information can be found on the tool's [GitHub](https://github.com/PINTO0309/openvino2tensorflow) page.

```
(vbx env)~/sdk$ openvino2tensorflow --model-path MODEL.xml --output_full_integer_quant_tflite \
--load_dest_file_path_for_the_calib_npy CALIB.npy
```


## PINTO onnx2tf

PINTO's onnx2tf tool converts models in ONNX format into a TF Lite model. Similar to PINTO's openvino2tensorflow tool, it supports different arguments and allows users flexibility in the TF Lite conversion. We can use it to maintain static input shapes or cut the ONNX graphs at designated nodes when converting to TF Lite. The following is an example of the tool's usage. For more information, see the tool's [GitHub](https://github.com/PINTO0309/onnx2tf) page.
```
(vbx env)~/sdk$ onnx2tf --input_onnx_file_path MODEL_NAME.onnx --output_integer_quantized_tflite \
--output_signaturedefs \
--custom_input_op_name_np_data_path INPUT_OP NUMPY.npy [MEAN] [STD]
```
## VectorBlox SDK generate_npy
```generate_npy``` generates a numpy calibration file using sample images, for tflite calibration. The image directory needs to be specified. If using dimensions other than 244 x 244 in height and width, shape needs to be specified. The following is an example of the tool's usage.
```
(vbx env)~/sdk$ generate_npy sample_images/ --output_name OUTPUT_NAME --shape HEIGHT WIDTH --count 20
```


## VectorBlox SDK tflite_quantize

If the model is based on TensorFlow, then users can utilize the SDK provided tflite_quantize tool to convert it to TF Lite directly. The tool can also allow for normalization based off of mean and scale parameters, which are applied to the calibration data. The following is an example of the tool's usage.
```
(vbx env)~/sdk$ tflite_quantize saved_model/ MODEL_NAME.tflite --data NUMPY.npy --mean 127.5 --scale 127.5
```
After generating a TF Lite model, the tflite_cut tool can be used to re-assign starting and ending nodes for the TF Lite graph. The following is an example of the tool's usage.
```
(vbx env)~/sdk$ tflite_cut MODEL_NAME.tflite -i 3 -o 5
```

## VectorBlox SDK tflite_preprocess
```tflite_preprocess``` adds a preprocessing layer on top of a TF Lite model. This is required for running models on hardware and in C, as the inputs obtained are uint8 rather than int8, and values would also need to be scaled.
```
(vbx env)~/sdk$ tflite_preprocess MODEL_NAME.tflite -s 255 -m 127
```
## VectorBlox SDK tflite_postprocess
```tflite_postprocess``` adds an injection layer for our demo, resizes the outputs and turns category outputs to pixel values.
```
(vbx env)~/sdk$ tflite_postprocess MODEL_NAME.tflite --dataset VOC --opacity 0.8 --height 1080 width 1920
```
## VectorBlox SDK tflite_cut
If a model needs to be cut at specific nodes, the tflite_cut function can be used to cut the graph at those specific nodes. The output will then be printed out with a `*.0.tflite, *.1.tflite, *.2.tflite` and so on. The following is an example of the tool's usage.
```
(vbx env)~/sdk$ tflite_cut MODEL_NAME.tflite -c
```