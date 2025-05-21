
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
if [ ! -f $VBX_SDK/tutorials/coco2017_rgb_norm_20x512x1024x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/coco2017_rgb_20x416x416x3.npy -o $VBX_SDK/tutorials/coco2017_rgb_norm_20x512x1024x3.npy -s 512 1024  --norm 
fi

echo "Downloading FFNet-78S-LowRes..."
# model details @ 
[ -f FFNet-78S-LowRes.onnx ] || wget -q --no-check-certificate https://huggingface.co/qualcomm/FFNet-78S-LowRes/resolve/main/FFNet-78S-LowRes.onnx

echo "Running ONNX2TF..."
onnx2tf -cind image $VBX_SDK/tutorials/coco2017_rgb_norm_20x512x1024x3.npy [[[0.,0.,0.]]] [[[1.,1.,1.]]] \
-i FFNet-78S-LowRes.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
cp saved_model/FFNet-78S-LowRes_full_integer_quant.tflite FFNet-78S-LowRes.tflite

if [ -f FFNet-78S-LowRes.tflite ]; then
   tflite_preprocess FFNet-78S-LowRes.tflite  --scale 255
fi

if [ -f FFNet-78S-LowRes.pre.tflite ]; then
   tflite_postprocess FFNet-78S-LowRes.pre.tflite  --dataset cityscapes \
--opacity 0.8 \
--height 1080 \
--width 1920
fi

if [ -f FFNet-78S-LowRes.pre.post.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t FFNet-78S-LowRes.pre.post.tflite -o FFNet-78S-LowRes.vnnx
fi

if [ -f FFNet-78S-LowRes.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/segmentation.py FFNet-78S-LowRes.vnnx $VBX_SDK/tutorials/test_images/A0PQ76.jpg --dataset cityscapes --inj 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model FFNet-78S-LowRes.vnnx $VBX_SDK/tutorials/test_images/A0PQ76.jpg  '
fi

deactivate
