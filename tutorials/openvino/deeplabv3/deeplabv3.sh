
##########################################################
#  _    __          __             ____  __              #
# | |  / /__  _____/ /_____  _____/ __ )/ /___  _  __    #
# | | / / _ \/ ___/ __/ __ \/ ___/ __  / / __ \| |/_/    #
# | |/ /  __/ /__/ /_/ /_/ / /  / /_/ / / /_/ />  <      #
# |___/\___/\___/\__/\____/_/  /_____/_/\____/_/|_|      #
#                                                        #
# https://github.com/Microchip-Vectorblox/VectorBlox-SDK #
# v3.1                                                   #
#                                                        #
##########################################################



set -e
echo "Checking and activating VBX Python Environment..."
if [ -z $VBX_SDK ]; then
    echo "\$VBX_SDK not set. Please run 'source setup_vars.sh' from the SDK's root folder" && exit 1
fi
source $VBX_SDK/vbx_env/bin/activate


# generate_npy is an internal tool that creates a npy array
#  Purpose: Generates a npy array if an existing one does not exist, this is using custom img data
#  - Required Inputs: source dataset, output name, size
#  - Output: npy array
echo "Checking for Numpy calibration data file..."
if [ ! -f $VBX_SDK/tutorials/coco2017_rgb_20x513x513x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/coco2017_rgb_20x416x416x3.npy -o $VBX_SDK/tutorials/coco2017_rgb_20x513x513x3.npy -s 513 513 
fi

echo "Checking for deeplabv3 files..."

# model details @ https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/deeplabv3
if [ ! -f deeplabv3.tflite ]; then
# model details @ https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/deeplabv3
omz_downloader --name deeplabv3
fi


if [ -f public/deeplabv3/deeplabv3_mnv2_pascal_train_aug/frozen_inference_graph.pb ]; then
   if ! echo "b0a1d0340189d7003291010abbc2e475 public/deeplabv3/deeplabv3_mnv2_pascal_train_aug/frozen_inference_graph.pb" | md5sum -c; then
       echo -e "\n There is an issue with the deeplabv3 model file as the expected checksum does not match.\n The model source can be found at: https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/deeplabv3.\n If the model information has changed, please update this script and re-run the tutorial."
       exit 1
   fi
fi

if [ ! -f deeplabv3.tflite ]; then
   echo "Running Model Optimizer..."
   mo --input_model public/deeplabv3/deeplabv3_mnv2_pascal_train_aug/frozen_inference_graph.pb \
--input_shape=[1,513,513,3] \
--input=0:MobilenetV2/Conv/Conv2D \
--mean_values [127.5,127.5,127.5] \
--scale_values [127.5] \
--output=ArgMax \
--static_shape
fi

# openvino2tensorflow is an external model conversion tool to convert an openvino model to int8 tflite
# specific operation information can be found here: https://pypi.org/project/openvino2tensorflow
#  Purpose: Convert source model to int8 tflite format
#  - Required Inputs: openvino compliant model, calibration npy array
#  - Output: int8 tflite model
if [ ! -f deeplabv3.tflite ]; then
   echo "Running OpenVINO2Tensorflow..."
   openvino2tensorflow --load_dest_file_path_for_the_calib_npy $VBX_SDK/tutorials/coco2017_rgb_20x513x513x3.npy \
--keep_input_tensor_in_nchw \
--weight_replacement_config fix.json \
--model_path frozen_inference_graph.xml \
--output_full_integer_quant_tflite \
--string_formulas_for_normalization '(data - [0.,0.,0.]) / [1.,1.,1.]'
   cp saved_model/model_full_integer_quant.tflite deeplabv3.tflite
fi


# tflite_cut is an internal tool used to split an existing model into smaller models
#  Purpose: cuts a model into smaller subsections, can be used to decrease runtime or for debugging purposes
#  - Required Inputs: tflite source model, cut section(s)
#  - Outputs: tflite model
if [ -f deeplabv3.tflite ]; then 
   echo "Cutting graph" 
   tflite_cut deeplabv3.tflite -c 79
   mv deeplabv3.0.tflite deeplabv3.cut.tflite 
fi


# tflite_preprocess is an internal tool used to add a preprocess layer to the start of the model
#  Purpose: adds a preprocess layer to the start of the model (if none, will just preprocess by adding a uint8->int8 layer)
#  - Required Inputs: tflite source model, additional arguments 
#  - Outputs: preprocessed tflite model
if [ -f deeplabv3.cut.tflite ]; then
   tflite_preprocess deeplabv3.cut.tflite   
fi


# tflite_postprocess is an internal tool that adds a postprocess layer near the end of the model
#  Purpose: adds a postprocess layer at the end of the model
#  - Required Inputs: tflite source model, additional postprocessing arguments
#  - Outputs: tflite model with postprocessed layer at end
if [ -f deeplabv3.cut.pre.tflite ]; then
   tflite_postprocess deeplabv3.cut.pre.tflite  --post-process-layer PIXEL_VOC \
--opacity 0.8 \
--height 1080 \
--width 1920
fi


# vnnx_compile is an internal tool that converts an int8 tflite file to a binary file that can be run on the SDK and VectorBlox FPGA
#  Purpose: converts int8 tflite to binary
#  - Required Inputs: int8 tflite, size configuration, compression configuration, output file name
#  - Outputs: binary object files(.hex and binary file)
if [ -f deeplabv3.cut.pre.post.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t deeplabv3.cut.pre.post.tflite  -o deeplabv3_V1000_ncomp.vnnx
fi


# This step runs the final compiled binary in Python, it also shows how to run the same file in C simulation for SDK
#   *Currently C simulation is not supported for unstructured compression
if [ -f deeplabv3_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/segmentation.py deeplabv3_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/A0PQ76.jpg --dataset VOC --inj 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model deeplabv3_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/A0PQ76.jpg  '
fi

deactivate
