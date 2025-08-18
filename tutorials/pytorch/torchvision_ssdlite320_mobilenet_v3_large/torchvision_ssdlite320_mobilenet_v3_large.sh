
##########################################################
#  _    __          __             ____  __              #
# | |  / /__  _____/ /_____  _____/ __ )/ /___  _  __    #
# | | / / _ \/ ___/ __/ __ \/ ___/ __  / / __ \| |/_/    #
# | |/ /  __/ /__/ /_/ /_/ / /  / /_/ / / /_/ />  <      #
# |___/\___/\___/\__/\____/_/  /_____/_/\____/_/|_|      #
#                                                        #
# https://github.com/Microchip-Vectorblox/VectorBlox-SDK #
# v2.0                                                   #
#                                                        #
##########################################################

set -e
echo "Checking and activating VBX Python Environment..."
if [ -z $VBX_SDK ]; then
    echo "\$VBX_SDK not set. Please run 'source setup_vars.sh' from the SDK's root folder" && exit 1
fi
source $VBX_SDK/vbx_env/bin/activate

echo "Checking for Numpy calibration data file..."
if [ ! -f $VBX_SDK/tutorials/coco2017_rgb_norm_20x320x320x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/coco2017_rgb_20x416x416x3.npy -o $VBX_SDK/tutorials/coco2017_rgb_norm_20x320x320x3.npy -s 320 320  --norm 
fi

echo "Checking for torchvision_ssdlite320_mobilenet_v3_large files..."

# model details @ https://pytorch.org/vision/0.14/models/ssdlite.html
if [ ! -f torchvision_ssdlite320_mobilenet_v3_large.tflite ]; then
python $VBX_SDK/tutorials/torchvision_to_onnx.py ssdlite320_mobilenet_v3_large -i 320
python - <<EOF
import onnx
model_inputs = ['/transform/Unsqueeze_7_output_0']
model_outputs = ['/head/regression_head/module_list.0/module_list.0.1/Conv_output_0', '/head/classification_head/module_list.0/module_list.0.1/Conv_output_0', '/head/regression_head/module_list.1/module_list.1.1/Conv_output_0', '/head/classification_head/module_list.1/module_list.1.1/Conv_output_0', '/head/regression_head/module_list.2/module_list.2.1/Conv_output_0', '/head/classification_head/module_list.2/module_list.2.1/Conv_output_0', '/head/regression_head/module_list.3/module_list.3.1/Conv_output_0', '/head/classification_head/module_list.3/module_list.3.1/Conv_output_0', '/head/regression_head/module_list.4/module_list.4.1/Conv_output_0', '/head/classification_head/module_list.4/module_list.4.1/Conv_output_0', '/head/regression_head/module_list.5/module_list.5.1/Conv_output_0', '/head/classification_head/module_list.5/module_list.5.1/Conv_output_0']
onnx.utils.extract_model('ssdlite320_mobilenet_v3_large.onnx', 'ssdlite320_mobilenet_v3_large.onnx', model_inputs, model_outputs)
EOF
sor4onnx --input_onnx_file_path ssdlite320_mobilenet_v3_large.onnx --old_new '/transform/Unsqueeze_7_output_0' 'images' --mode full --search_mode prefix_match --output_onnx_file_path ssdlite320_mobilenet_v3_large.onnx
fi


if [ ! -f torchvision_ssdlite320_mobilenet_v3_large.tflite ]; then
   echo "Running ONNX2TF..."
   onnx2tf -cind images $VBX_SDK/tutorials/coco2017_rgb_norm_20x320x320x3.npy [[[[0.5,0.5,0.5]]]] [[[[0.5,0.5,0.5]]]] \
-ois images:1,3,320,320 \
-i ssdlite320_mobilenet_v3_large.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
   cp saved_model/ssdlite320_mobilenet_v3_large_full_integer_quant.tflite torchvision_ssdlite320_mobilenet_v3_large.tflite
fi
if [ -f torchvision_ssdlite320_mobilenet_v3_large.tflite ]; then
   tflite_preprocess torchvision_ssdlite320_mobilenet_v3_large.tflite  --mean 127.5 127.5 127.5 --scale 127.5 127.5 127.5
fi

if [ -f torchvision_ssdlite320_mobilenet_v3_large.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t torchvision_ssdlite320_mobilenet_v3_large.pre.tflite -o torchvision_ssdlite320_mobilenet_v3_large.vnnx
fi

if [ -f torchvision_ssdlite320_mobilenet_v3_large.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/ssdv2.py torchvision_ssdlite320_mobilenet_v3_large.vnnx $VBX_SDK/tutorials/test_images/dog.jpg --torch 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model torchvision_ssdlite320_mobilenet_v3_large.vnnx $VBX_SDK/tutorials/test_images/dog.jpg SSDV2'
fi

deactivate
