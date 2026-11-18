
##########################################################
#  _    __          __             ____  __              #
# | |  / /__  _____/ /_____  _____/ __ )/ /___  _  __    #
# | | / / _ \/ ___/ __/ __ \/ ___/ __  / / __ \| |/_/    #
# | |/ /  __/ /__/ /_/ /_/ / /  / /_/ / / /_/ />  <      #
# |___/\___/\___/\__/\____/_/  /_____/_/\____/_/|_|      #
#                                                        #
# https://github.com/Microchip-Vectorblox/VectorBlox-SDK #
# v3.2                                                   #
#                                                        #
##########################################################



set -e
echo "Checking and activating VBX Python Environment..."
if [ -z $VBX_SDK ]; then
    echo "\$VBX_SDK not set. Please run 'source setup_vars.sh' from the SDK's root folder" && exit 1
fi
source $VBX_SDK/vbx_env/bin/activate

echo "Checking for anomaly_transformer files..."

# model details @ https://github.com/thuml/Anomaly-Transformer
if [ ! -f anomaly_transformer.tflite ]; then
    git clone https://github.com/thuml/Anomaly-Transformer
    cd Anomaly-Transformer
    cp ../anomaly_*.py .
    cp ../pyproject.toml .
    git apply ../anomaly_transformer.patch
    uv sync --group cpu
    uv run python anomaly_train.py --dataset SMD --num_epochs 5
    uv run python anomaly_export.py --dataset SMD
    uv run python anomaly_FFN_rewrite.py exports/SMD_fp32.tflite exports/SMD_ffn.tflite
    uv run python anomaly_quantize.py exports/SMD_ffn.tflite --dataset SMD --calib 64
    uv run python -c "
import numpy as np
from anomaly_train import get_synthetic_loader
x, _ = next(iter(get_synthetic_loader(batch_size=1, win_size=100, n_channels=38, n_windows=1, mode='test', seed=0)))
np.save('SMD_sample.npy', x[0].numpy())
"
    tflite_clean -i exports/SMD_ffn_norm_int8.tflite
    cp exports/SMD_ffn_norm_int8.clean.tflite ../anomaly_transformer.tflite
    cp SMD_sample.npy ../
    cd ../
fi



# tflite_transform is an internal tool used to add various custom transformations to the model
# additional information found here: https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/python/vbx/vbx/generate/transform_tflite.py
# supported arguments are found in the `passes` struct at the top#  Purpose: Adds various transformations to a tflite model
#  - Required Inputs: tflite source model, additional arguments
#  - Outputs: transformed tflite model
if [ -f anomaly_transformer.tflite ]; then
   tflite_transform anomaly_transformer.tflite  -p FRAC_BATCH FUSE_ADJ FUSE_SLICES
fi


# vnnx_compile is an internal tool that converts an int8 tflite file to a binary file that can be run on the SDK and VectorBlox FPGA
#  Purpose: converts int8 tflite to binary
#  - Required Inputs: int8 tflite, size configuration, compression configuration, output file name
#  - Outputs: binary object files(.hex and binary file)
if [ -f anomaly_transformer.tr.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t anomaly_transformer.tr.tflite  -o anomaly_transformer_V1000_ncomp.vnnx
fi


# This step runs the final compiled binary in Python, it also shows how to run the same file in C simulation for SDK
#   *Currently C simulation is not supported for unstructured compression
if [ -f anomaly_transformer_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/anomaly.py anomaly_transformer_V1000_ncomp.vnnx SMD_sample.npy 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model anomaly_transformer_V1000_ncomp.vnnx SMD_sample.npy ANOMALY'
fi

deactivate
