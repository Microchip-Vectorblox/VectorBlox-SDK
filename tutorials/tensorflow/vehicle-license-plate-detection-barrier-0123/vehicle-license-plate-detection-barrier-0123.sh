
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

echo "Downloading vehicle-license-plate-detection-barrier-0123..."
omz_downloader --name vehicle-license-plate-detection-barrier-0123

echo "Running Model Optimizer..."
# model details @ https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/vehicle-license-plate-detection-barrier-0123
mo --input_model=public/vehicle-license-plate-detection-barrier-0123/model/model.pb.frozen \
--framework tf \
--input_shape [1,256,256,3] \
--reverse_input_channels \
--scale=127.5 \
--mean_values=[127.5,127.5,127.5] \
--output=ssd_heads/head_0/layer_15/expansion_output_mbox_loc/BiasAdd,ssd_heads/head_0/layer_15/expansion_output_mbox_conf/BiasAdd,ssd_heads/head_1/layer_19_mbox_loc/BiasAdd,ssd_heads/head_1/layer_19_mbox_conf/BiasAdd,ssd_heads/head_2/feature_map_1_mbox_loc/BiasAdd,ssd_heads/head_2/feature_map_1_mbox_conf/BiasAdd,ssd_heads/head_3/feature_map_2_mbox_loc/BiasAdd,ssd_heads/head_3/feature_map_2_mbox_conf/BiasAdd,ssd_heads/head_4/feature_map_3_mbox_loc/BiasAdd,ssd_heads/head_4/feature_map_3_mbox_conf/BiasAdd,ssd_heads/head_5/feature_map_4_mbox_loc/BiasAdd,ssd_heads/head_5/feature_map_4_mbox_conf/BiasAdd \
--static_shape

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x model.pb.xml  -c V1000 -f ../../sample_images -o vehicle-license-plate-detection-barrier-0123.vnnx --bias-correction

echo "Running Simulation..."
python $VBX_SDK/example/python/vehicle.py vehicle-license-plate-detection-barrier-0123.vnnx ../../A0PQ76.jpg -p vehicle_priors.npy

deactivate
