
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
if [ ! -f $VBX_SDK/tutorials/face_rgb_norm_20x288x512x3.npy ]; then
    wget -P $VBX_SDK/tutorials/ https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/EAP/calib_npy/face_rgb_norm_20x288x512x3.npy
fi

echo "Downloading scrfd_500m_bnkps..."
# model details @ https://insightface.ai/scrfd
# ONNX file was created through the following process:
# git clone https://github.com/deepinsight/insightface.git
# install https://github.com/deepinsight/insightface/tree/master/detection/scrfd#installation
# download SCRFD_500M_KPS https://github.com/deepinsight/insightface/tree/master/detection/scrfd#pretrained-models
# python tools/scrfd2onnx.py configs/scrfd/scrfd_500m_bnkps.py scrfd_500m_bnkps.pth --input-img garden_512x288.jpg --output-file scrfd_500m_bnkps.onnx

[ -f scrfd_500m_bnkps.onnx ] || wget https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/Releases/ModelZoo/scrfd_500m_bnkps.onnx
if [ ! -f calibration_image_sample_data_20x128x128x3_float32.npy ]; then
    wget https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/EAP/calib_npy/calibration_image_sample_data_20x128x128x3_float32.npy
fi

echo "Running ONNX2TF..."
onnx2tf -cind input.1 $VBX_SDK/tutorials/face_rgb_norm_20x288x512x3.npy [[[[0.5,0.5,0.5]]]] [[[[0.502,0.502,0.502]]]] \
--overwrite_input_shape input.1:1,3,288,512 \
--output_names_to_interrupt_model_conversion "487" "488" "489" "462" "463" "464" "437" "438" "439" \
-i scrfd_500m_bnkps.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
cp saved_model/scrfd_500m_bnkps_full_integer_quant.tflite scrfd_500m_bnkps.tflite

if [ -f scrfd_500m_bnkps.tflite ]; then
   tflite_preprocess scrfd_500m_bnkps.tflite  --mean 127.5 127.5 127.5 --scale 128 128 128
fi

if [ -f scrfd_500m_bnkps.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t scrfd_500m_bnkps.pre.tflite -o scrfd_500m_bnkps.vnnx
fi

if [ -f scrfd_500m_bnkps.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/scrfdInfer.py scrfd_500m_bnkps.vnnx $VBX_SDK/tutorials/test_images/garden.jpg 
fi

deactivate
