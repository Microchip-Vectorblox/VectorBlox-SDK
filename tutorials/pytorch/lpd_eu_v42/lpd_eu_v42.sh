
#######################################
#                                     #
#  ____    ____  ______   ___   ___   #
#  \   \  /   / |   _  \  \  \ /  /   #
#   \   \/   /  |  |_)  |  \  V  /    #
#    \      /   |   _  <    >   <     #
#     \    /    |  |_)  |  /  ^  \    #
#      \__/     |______/  /__/ \__\   #
#                                     #
# Refer to Programmer's Guide         #
# for full details                    #
#                                     #
#                                     #
#######################################

set -e
echo "Checking and Activating VBX Python Environment..."
if [ -z $VBX_SDK ]; then
    echo "\$VBX_SDK not set. Please run 'source setup_vars.sh' from the SDK's root folder" && exit 1
fi
source $VBX_SDK/vbx_env/bin/activate

echo "Downloading lpd_eu_v42..."
[ -f lpd_eu_v42.onnx ] || wget https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/Releases/ModelZoo/lpd_eu_v42.onnx

echo "Running Model Optimizer..."
# model details @ pytorch/lpd_eu_v42/README.md
mo --input_model lpd_eu_v42.onnx \
--reverse_input_channels \
--scale_values [255.] \
--static_shape \
--output=Output_Str32_Shape,Output_Str32_Obj,Output_Str16_Shape,Output_Str16_Obj,Output_Str8_Shape,Output_Str8_Obj

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x lpd_eu_v42.xml  -c V1000 -f ../../sample_vehicles  -o lpd_eu_v42.vnnx --bias-correction

echo "Running Simulation..."
python $VBX_SDK/example/python/detect_plate.py lpd_eu_v42.vnnx ../../test_images/parked_cars.png

deactivate
