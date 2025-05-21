#!/bin/bash
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
function install_venv() {
    $PYTHON_EXE -m venv  $SCRIPTDIR/vbx_env
    source $SCRIPTDIR/vbx_env/bin/activate

    python -m pip install --upgrade pip
    python -m pip install --upgrade setuptools
    python -m pip install wheel 
    python -m pip install psutil==5.9.5
    python -m pip install onnx==1.16.1

    # tensorflow 2.16.1, 2.16.2, and 2.17.0 fail for the following tutorials:
    # - tensorflow/yolo-v3-tiny-tf, yolo-v3-tf, yolo-v4-tf
    # when using keras-YOLOv3-model-set
    python -m pip install tensorflow-cpu==2.15.1
    python -m pip install numpy==1.23.5

    python -m pip install tensorflow_datasets==4.9.3
    python -m pip install nvidia-pyindex
    python -m pip install onnx-graphsurgeon
    python -m pip install protobuf==3.20.3
    python -m pip install onnxsim==0.4.36
    python -m pip install sor4onnx
    python -m pip install sne4onnx
    python -m pip install sng4onnx==1.0.1
    python -m pip install onnxruntime==1.18.1
    python -m pip install ml_dtypes==0.3.1
    python -m pip install onnx2tf==1.22.3
    python -m pip install opencv-python==4.7.0.72

    # openvino 2024 fails for the openvino tutorials when using openvino2tensorflow.
    # They fail for tensorflow 2.15, 2.16 and 2.17.
    python -m pip install openvino==2023.0.1
    python -m pip install openvino-dev==2023.0.1
    python -m pip install openvino-telemetry==2023.2.1

    python -m pip install openvino2tensorflow==1.34.0
    python -m pip install torch==2.3.0
    python -m pip install torchvision==0.18.0
    python -m pip install matplotlib
    python -m pip install silence_tensorflow
    python -m pip install natsort
    python -m pip install ultralytics==8.3.72
    python -m pip install tf_keras==2.15.1
    python -m pip install tflite_support==0.4.4
    python -m pip install onnxslim==0.1.32
    python -m pip install XlsxWriter
    python -m pip install prtpy==0.8.1
    python -m pip install -e $SCRIPTDIR/python/vbx
	
	python -m pip install posix-ipc
    deactivate
}

function has_python() {
    for v in 3.11 3.10
    do
        if [ -n "$(which python$v)" ]
        then
            PYTHON_EXE=python$v
            return 0
        fi
    done
	echo "Error Unable to find compatible version of python. Allowed Versions: Python3.10 or Python3.11" >&2
	return 1
}

function has_installed() {
	if [ ! -d $SCRIPTDIR/vbx_env ]; then
		echo "Initial install of VBX Python Environment required. May take several minutes..."
		install_venv
	fi
	source $SCRIPTDIR/vbx_env/bin/activate
	python -c 'import vbx' && command -v mo &> /dev/null && command -v omz_downloader &> /dev/null
	VALID=$?
	deactivate
	if [ $VALID -gt 0 ]; then
		echo ""
		echo "VBX Python Environment failed to install correctly."
		echo "Ensure install dependencies met then reinstall by running 'source setup_vars.sh'"
		deactivate
		rm -rf vbx_env
		return 1
	fi
}

if has_python && has_installed; then
	export VBX_SDK=$SCRIPTDIR
	echo ""
	echo "export VBX_SDK="$VBX_SDK
	echo ""
	export NX_SDK=$SCRIPTDIR/../../../../../tsnp_software/
	export TF_CPP_MIN_LOG_LEVEL=3


	echo "VBX Python Environment ready. Activating..."
	source $VBX_SDK/vbx_env/bin/activate
	echo "To deactivate VBX Python Environment, run 'deactivate'"
fi
