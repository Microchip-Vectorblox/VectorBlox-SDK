
##########################################################
#  _    __          __             ____  __              #
# | |  / /__  _____/ /_____  _____/ __ )/ /___  _  __    #
# | | / / _ \/ ___/ __/ __ \/ ___/ __  / / __ \| |/_/    #
# | |/ /  __/ /__/ /_/ /_/ / /  / /_/ / / /_/ />  <      #
# |___/\___/\___/\__/\____/_/  /_____/_/\____/_/|_|      #
#                                                        #
# https://github.com/Microchip-Vectorblox/VectorBlox-SDK #
# v3.0                                                   #
#                                                        #
##########################################################

set -e
echo "Checking and activating VBX Python Environment..."
if [ -z $VBX_SDK ]; then
    echo "\$VBX_SDK not set. Please run 'source setup_vars.sh' from the SDK's root folder" && exit 1
fi
source $VBX_SDK/vbx_env/bin/activate

echo "Checking for Numpy calibration data file..."
if [ ! -f $VBX_SDK/tutorials/face_rgb_norm_20x288x512x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/face_rgb_20x273x481x3.npy -o $VBX_SDK/tutorials/face_rgb_norm_20x288x512x3.npy -s 288 512  --norm 
fi

echo "Checking for scrfd_500m_bnkps files..."

# model details @ https://insightface.ai/scrfd
# ONNX file was created through the following process:
# git clone https://github.com/deepinsight/insightface.git
# install https://github.com/deepinsight/insightface/tree/master/detection/scrfd#installation
# download SCRFD_500M_KPS https://github.com/deepinsight/insightface/tree/master/detection/scrfd#pretrained-models
# python tools/scrfd2onnx.py configs/scrfd/scrfd_500m_bnkps.py scrfd_500m_bnkps.pth --input-img garden_512x288.jpg --output-file scrfd_500m_bnkps.onnx

[ -f scrfd_500m_bnkps.onnx ] || wget -q --no-check-certificate https://github.com/Microchip-Vectorblox/assets/releases/download/assets/scrfd_500m_bnkps.onnx


if [ ! -f scrfd_500m_bnkps.tflite ]; then
   echo "Running ONNX2TF..."
   onnx2tf -cind input.1 $VBX_SDK/tutorials/face_rgb_norm_20x288x512x3.npy [[[[0.5,0.5,0.5]]]] [[[[0.502,0.502,0.502]]]] \
--overwrite_input_shape input.1:1,3,288,512 \
--output_names_to_interrupt_model_conversion "487" "488" "489" "462" "463" "464" "437" "438" "439" \
-i scrfd_500m_bnkps.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
   cp saved_model/scrfd_500m_bnkps_full_integer_quant.tflite scrfd_500m_bnkps.tflite
fi
if [ -f scrfd_500m_bnkps.tflite ]; then
   tflite_preprocess scrfd_500m_bnkps.tflite  --mean 127.5 127.5 127.5 --scale 128 128 128
fi

if [ -f scrfd_500m_bnkps.pre.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t scrfd_500m_bnkps.pre.tflite  -o scrfd_500m_bnkps_V1000_ncomp.vnnx
fi

if [ -f scrfd_500m_bnkps_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/scrfdInfer.py scrfd_500m_bnkps_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/garden.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model scrfd_500m_bnkps_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/garden.jpg SCRFD'
fi

deactivate
