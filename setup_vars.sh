#!/bin/bash
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
function install_venv() {
    uv venv  $SCRIPTDIR/vbx_env --$PYTHON_EXE 
    source $SCRIPTDIR/vbx_env/bin/activate
    python -m pip install "pip<23.1" --no-build-isolation
    python -m pip install uv
    uv pip install "setuptools<=68.2.2"
    uv pip install wheel
    uv pip install psutil==5.9.5
    uv pip install onnx==1.16.1

    # tensorflow 2.16.1, 2.16.2, and 2.17.0 fail for the following tutorials:
    # - tensorflow/yolo-v3-tiny-tf, yolo-v3-tf, yolo-v4-tf
    # when using keras-YOLOv3-model-set
    uv pip install tensorflow-cpu==2.15.1
    uv pip install numpy==1.23.5
    uv pip install tensorflow_datasets==4.9.3
    uv pip install tensorflow-metadata==1.13.1 --no-deps
    uv pip install onnx-graphsurgeon
    uv pip install protobuf==3.20.3
    uv pip install onnxsim==0.4.36
    uv pip install sor4onnx
    uv pip install sne4onnx
    uv pip install sng4onnx==1.0.1
    uv pip install onnxruntime==1.18.1
    uv pip install ml_dtypes==0.3.1
    uv pip install onnx2tf==1.22.3
    uv pip install opencv-python==4.7.0.72

    # openvino 2024 fails for the openvino tutorials when using openvino2tensorflow.
    # They fail for tensorflow 2.15, 2.16 and 2.17.
    uv pip install openvino==2023.0.1
    uv pip install openvino-dev==2023.0.1
    uv pip install openvino-telemetry==2023.2.1
    uv pip install openvino2tensorflow==1.34.0
    uv pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.3.0+cpu
    uv pip install --extra-index-url https://download.pytorch.org/whl/cpu torchvision==0.18.0+cpu
    uv pip install matplotlib
    uv pip install silence_tensorflow
    uv pip install natsort
    uv pip install ultralytics==8.3.72
    uv pip install tf_keras==2.15.1
    uv pip install tflite_support==0.4.4 #not supported for python 3.12, needed for onnx2tf
    uv pip install onnxslim==0.1.32
    uv pip install XlsxWriter
    uv pip install prtpy===0.8.1
    uv pip install orjson
    uv pip install -e $SCRIPTDIR/python/vbx
    uv pip install posix-ipc
    uv pip install json2onnx
    uv pip install onnx2json
    cd $SCRIPTDIR/vbx_env/lib/python*/site-packages/openvino/ &&\
    echo $'import warnings\nwarnings.filterwarnings("ignore")\nimport logging\nlogging.basicConfig(level=logging.INFO)' >> __init__.py &&\
    #sed -i 's/WARNING/ERROR/g' tools/pot/utils/logger.py
    cd -
    deactivate
}

function has_python() {
	CURRENT_VERSION=3.10
	if [ -n "$(which uv)" ]
	then
		PYTHON_EXE="python $CURRENT_VERSION"
		return 0
	fi
	echo "Error 'uv' package not found, please ensure that it is properly installed and located in your $PATH"
	return 1
}

function has_installed() {
	if [ ! -d $SCRIPTDIR/vbx_env ]; then
		echo "Initial install of VBX Python Environment required. May take several minutes..."
		install_venv
	fi
	source $SCRIPTDIR/vbx_env/bin/activate
	python -c 'import vbx' && command -v mo &> /dev/null && command -v omz_downloader &> /dev/null
	echo $?
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
	export NX_SDK=$SCRIPTDIR/snp_compiler/
	echo ""
	echo "export NX_SDK="$NX_SDK
	echo ""
	export TF_CPP_MIN_LOG_LEVEL=3

	echo "VBX Python Environment ready. Activating..."
	source $VBX_SDK/vbx_env/bin/activate
	echo "To deactivate VBX Python Environment, run 'deactivate'"
fi
