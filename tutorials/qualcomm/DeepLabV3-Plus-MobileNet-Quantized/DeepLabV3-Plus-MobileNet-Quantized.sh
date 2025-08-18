
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
if [ ! -f $VBX_SDK/tutorials/coco2017_rgb_norm_20x520x520x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/coco2017_rgb_20x416x416x3.npy -o $VBX_SDK/tutorials/coco2017_rgb_norm_20x520x520x3.npy -s 520 520  --norm 
fi

echo "Checking for DeepLabV3-Plus-MobileNet-Quantized files..."

# model details @ https://aihub.qualcomm.com/mobile/models/deeplabv3_plus_mobilenet
if [ ! -f DeepLabV3-Plus-MobileNet-Quantized.tflite ]; then
   [ -f DeepLabV3-Plus-MobileNet_sim.onnx ] || wget -q --no-check-certificate https://huggingface.co/qualcomm/DeepLabV3-Plus-MobileNet/resolve/e0501ac875458bf2fde9d6c910b1fc43ac701fa6/DeepLabV3-Plus-MobileNet.onnx
python - <<EOF
import onnx
import numpy as np
model = onnx.load('DeepLabV3-Plus-MobileNet.onnx')
for i,info in enumerate(model.graph.value_info):
   if info.type.tensor_type.elem_type == 7:
       model.graph.value_info[i].type.tensor_type.elem_type = 6
for init in model.graph.initializer:
    if init.name == '/model/aspp/Concat_output_0':
        t = onnx.numpy_helper.from_array(np.array([1,256,33,33],dtype='int64'))
        init.raw_data = t.raw_data
    if init.name == '/model/decoder/Concat_output_0':
        t = onnx.numpy_helper.from_array(np.array([1,256,130,130],dtype='int64'))
        init.raw_data = t.raw_data
    if init.name == '/model/Concat_output_0':
        t = onnx.numpy_helper.from_array(np.array([1,21,520,520],dtype='int64'))
        init.raw_data = t.raw_data
onnx.save(model, 'DeepLabV3-Plus-MobileNet.dynamic.onnx')
EOF
onnxsim DeepLabV3-Plus-MobileNet.dynamic.onnx DeepLabV3-Plus-MobileNet_sim.onnx
fi



if [ ! -f DeepLabV3-Plus-MobileNet-Quantized.tflite ]; then
   echo "Running ONNX2TF..."
   onnx2tf -cind image $VBX_SDK/tutorials/coco2017_rgb_norm_20x520x520x3.npy [[[0.,0.,0.]]] [[[1.,1.,1.]]] \
-i DeepLabV3-Plus-MobileNet_sim.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
   cp saved_model/DeepLabV3-Plus-MobileNet_sim_full_integer_quant.tflite DeepLabV3-Plus-MobileNet-Quantized.tflite
fi
if [ -f DeepLabV3-Plus-MobileNet-Quantized.tflite ]; then
   tflite_preprocess DeepLabV3-Plus-MobileNet-Quantized.tflite  --scale 255
fi

if [ -f DeepLabV3-Plus-MobileNet-Quantized.pre.tflite ]; then
   tflite_postprocess DeepLabV3-Plus-MobileNet-Quantized.pre.tflite  --post-process-layer PIXEL_VOC \
--opacity 0.8 \
--height 1080 \
--width 1920
fi

if [ -f DeepLabV3-Plus-MobileNet-Quantized.pre.post.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t DeepLabV3-Plus-MobileNet-Quantized.pre.post.tflite -o DeepLabV3-Plus-MobileNet-Quantized.vnnx
fi

if [ -f DeepLabV3-Plus-MobileNet-Quantized.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/segmentation.py DeepLabV3-Plus-MobileNet-Quantized.vnnx $VBX_SDK/tutorials/test_images/A0PQ76.jpg --inj 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model DeepLabV3-Plus-MobileNet-Quantized.vnnx $VBX_SDK/tutorials/test_images/A0PQ76.jpg  '
fi

deactivate
