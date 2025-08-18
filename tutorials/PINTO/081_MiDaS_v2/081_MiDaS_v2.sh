
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

echo "Checking for 081_MiDaS_v2 files..."

# model details @ https://github.com/PINTO0309/PINTO_model_zoo
if ! [ -f 081_MiDaS_v2.tflite ]; then
    mkdir -p temp
    cd temp
    wget -q https://raw.githubusercontent.com/PINTO0309/PINTO_model_zoo/refs/heads/main/081_MiDaS_v2/download.sh
    bash download.sh
    cd ..
    cp temp/tflite_from_saved_model/model_full_integer_quant.tflite 081_MiDaS_v2.tflite
    rm -rf temp
fi


if [ -f 081_MiDaS_v2.tflite ]; then 
   echo "Cutting graph" 
   tflite_cut 081_MiDaS_v2.tflite -c 137
   mv 081_MiDaS_v2.0.tflite 081_MiDaS_v2.cut.tflite 
fi

if [ -f 081_MiDaS_v2.cut.tflite ]; then
   tflite_preprocess 081_MiDaS_v2.cut.tflite   
fi

if [ -f 081_MiDaS_v2.cut.pre.tflite ]; then
   tflite_postprocess 081_MiDaS_v2.cut.pre.tflite  --post-process-layer PIXEL_DEPTH \
--opacity 0.8 \
--height 1080 \
--width 1920
fi

if [ -f 081_MiDaS_v2.cut.pre.post.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t 081_MiDaS_v2.cut.pre.post.tflite -o 081_MiDaS_v2.vnnx
fi

if [ -f 081_MiDaS_v2.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/segmentation.py 081_MiDaS_v2.vnnx $VBX_SDK/tutorials/test_images/A0PQ76.jpg --dataset depth 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model 081_MiDaS_v2.vnnx $VBX_SDK/tutorials/test_images/A0PQ76.jpg  '
fi

deactivate
