
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
if [ ! -f $VBX_SDK/tutorials/coco2017_rgb_norm_20x288x512x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/coco2017_rgb_20x416x416x3.npy -o $VBX_SDK/tutorials/coco2017_rgb_norm_20x288x512x3.npy -s 288 512  --norm 
fi

echo "Downloading DeepLabV3-Plus-MobileNet_512x288..."
# model details @ 
[ -f DeepLabV3-Plus-MobileNet_512x288.onnx ] || wget -q --no-check-certificate https://huggingface.co/qualcomm/DeepLabV3-Plus-MobileNet/resolve/e0501ac875458bf2fde9d6c910b1fc43ac701fa6/DeepLabV3-Plus-MobileNet.onnx
python - <<EOF
import onnx
import numpy as np
model = onnx.load('DeepLabV3-Plus-MobileNet.onnx')
model.graph.input[0].type.tensor_type.shape.dim[2].Clear()
model.graph.input[0].type.tensor_type.shape.dim[3].Clear()
for info in model.graph.value_info:
    n = len(info.type.tensor_type.shape.dim)
    for ind in range(n-2,n):
        info.type.tensor_type.shape.dim[ind].Clear()
for i,info in enumerate(model.graph.value_info):
   if info.type.tensor_type.elem_type == 7:
       model.graph.value_info[i].type.tensor_type.elem_type = 6
model.graph.output[0].type.tensor_type.shape.dim[1].Clear()
model.graph.output[0].type.tensor_type.shape.dim[2].Clear()
for init in model.graph.initializer:
    if init.name == '/model/aspp/Concat_output_0':
        t = onnx.numpy_helper.from_array(np.array([1,256,18,32],dtype='int64'))
        init.raw_data = t.raw_data
    if init.name == '/model/decoder/Concat_output_0':
        t = onnx.numpy_helper.from_array(np.array([1,256,72,128],dtype='int64'))
        init.raw_data = t.raw_data
    if init.name == '/model/Concat_output_0':
        t = onnx.numpy_helper.from_array(np.array([1,21,288,512],dtype='int64'))
        init.raw_data = t.raw_data
onnx.save(model, 'DeepLabV3-Plus-MobileNet.dynamic.onnx')
EOF

onnxsim DeepLabV3-Plus-MobileNet.dynamic.onnx DeepLabV3-Plus-MobileNet_512x288.onnx --overwrite-input-shape 1,3,288,512

echo "Running ONNX2TF..."
onnx2tf -cind image $VBX_SDK/tutorials/coco2017_rgb_norm_20x288x512x3.npy [[[0.,0.,0.]]] [[[1.,1.,1.]]] \
-i DeepLabV3-Plus-MobileNet_512x288.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
cp saved_model/DeepLabV3-Plus-MobileNet_512x288_full_integer_quant.tflite DeepLabV3-Plus-MobileNet_512x288.tflite

if [ -f DeepLabV3-Plus-MobileNet_512x288.tflite ]; then
   tflite_preprocess DeepLabV3-Plus-MobileNet_512x288.tflite  --scale 255
fi

if [ -f DeepLabV3-Plus-MobileNet_512x288.pre.tflite ]; then
   tflite_postprocess DeepLabV3-Plus-MobileNet_512x288.pre.tflite  --dataset VOC \
--opacity 0.8 \
--height 1080 \
--width 1920
fi

if [ -f DeepLabV3-Plus-MobileNet_512x288.pre.post.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t DeepLabV3-Plus-MobileNet_512x288.pre.post.tflite -o DeepLabV3-Plus-MobileNet_512x288.vnnx
fi

if [ -f DeepLabV3-Plus-MobileNet_512x288.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/segmentation.py DeepLabV3-Plus-MobileNet_512x288.vnnx $VBX_SDK/tutorials/test_images/A0PQ76.jpg --dataset VOC --inj 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model DeepLabV3-Plus-MobileNet_512x288.vnnx $VBX_SDK/tutorials/test_images/A0PQ76.jpg  '
fi

deactivate
