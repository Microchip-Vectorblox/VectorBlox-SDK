
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

echo "Downloading yolov2-voc..."
rm -rf darkflow
git clone https://github.com/thtrieu/darkflow
cp loader.py darkflow/darkflow/utils/loader.py
[ -f yolov2-voc.cfg ] || wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2-voc.cfg
[ -f yolov2-voc.weights ] || wget https://pjreddie.com/media/files/yolov2-voc.weights
[ -f voc.names ] || wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/voc.names
cd darkflow


python3 -m venv dark_env
source dark_env/bin/activate
python -m pip install numpy==1.18.0
python -m pip install opencv-python==3.4.9.31
python -m pip install tensorflow==1.0
python -m pip install cython
python setup.py build_ext --inplace

python ./flow --model ../yolov2-voc.cfg --load ../yolov2-voc.weights --savepb --labels ../voc.names || true
cd ../
cp darkflow/built_graph/yolov2-voc.pb .
source $VBX_SDK/vbx_env/bin/activate

echo "Running Model Optimizer..."
# model details @ https://pjreddie.com/darknet/yolo/
converter --input_model yolov2-voc.pb \
--framework tf \
--input_shape [1,416,416,3] \
--reverse_input_channels \
--scale_values=[255.] \
--transformations_config yolo_v2_voc.json \
--static_shape 

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x yolov2-voc.xml  -c V1000 -f ../../sample_images -o yolov2-voc.vnnx

echo "Running Simulation..."
python $VBX_SDK/example/python/yolov2.py yolov2-voc.vnnx ../../dog.416.jpg

deactivate
