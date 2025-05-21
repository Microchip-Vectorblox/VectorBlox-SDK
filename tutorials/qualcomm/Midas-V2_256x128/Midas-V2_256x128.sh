
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
if [ ! -f $VBX_SDK/tutorials/coco2017_rgb_norm_20x128x256x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/coco2017_rgb_20x416x416x3.npy -o $VBX_SDK/tutorials/coco2017_rgb_norm_20x128x256x3.npy -s 128 256  --norm 
fi

echo "Downloading Midas-V2_256x128..."
# model details @ https://aihub.qualcomm.com/models/midas
if [ ! -f Midas-V2.onnx ]; then
    wget -q --no-check-certificate https://huggingface.co/qualcomm/Midas-V2/resolve/main/Midas-V2.onnx

fi
#wget https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.pb

python - <<EOF
import onnx
from onnx.tools import update_model_dims
from onnxsim import simplify
model = onnx.load('Midas-V2.onnx')
model.graph.input[0].type.tensor_type.shape.dim[2].Clear()
model.graph.input[0].type.tensor_type.shape.dim[3].Clear()
for info in model.graph.value_info:
    n = len(info.type.tensor_type.shape.dim)
    for ind in range(n-2,n):
        info.type.tensor_type.shape.dim[ind].Clear()
model.graph.output[0].type.tensor_type.shape.dim[2].Clear()
model.graph.output[0].type.tensor_type.shape.dim[3].Clear()
model_256x128 = update_model_dims.update_inputs_outputs_dims(model, {"image":[1,3,128,256]},{"depth_estimates":[1,1,128,256]})
model_simp, check = simplify(model_256x128)
onnx.save(model_simp,'Midas-V2_256x128.onnx')
EOF

echo "Running ONNX2TF..."
onnx2tf -cind image $VBX_SDK/tutorials/coco2017_rgb_norm_20x128x256x3.npy [[[0.,0.,0.]]] [[[1.,1.,1.]]] \
-i Midas-V2_256x128.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
cp saved_model/Midas-V2_256x128_full_integer_quant.tflite Midas-V2_256x128.tflite

if [ -f Midas-V2_256x128.tflite ]; then
   tflite_preprocess Midas-V2_256x128.tflite  --scale 255
fi

if [ -f Midas-V2_256x128.pre.tflite ]; then
   tflite_postprocess Midas-V2_256x128.pre.tflite  --dataset depth \
--opacity 0.8 \
--height 1080 \
--width 1920
fi

if [ -f Midas-V2_256x128.pre.post.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t Midas-V2_256x128.pre.post.tflite -o Midas-V2_256x128.vnnx
fi

if [ -f Midas-V2_256x128.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/segmentation.py Midas-V2_256x128.vnnx $VBX_SDK/tutorials/test_images/A0PQ76.jpg --dataset depth --inj 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model Midas-V2_256x128.vnnx $VBX_SDK/tutorials/test_images/A0PQ76.jpg  '
fi

deactivate
