#!/bin/bash
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
function has_script() {
	if [ ! -f $SCRIPTDIR/install_venv.sh ]; then
		echo "Ensure 'install_venv.sh' exists in root folder of SDK"
		return 1
	fi
}

function has_python() {
	pymajor=$(python3 -c 'import platform; major, _ , _ = platform.python_version_tuple(); print(major)')
	pyminor=$(python3 -c 'import platform; _ , minor, _ = platform.python_version_tuple(); print(minor)')
	if [ $pymajor -ne 3 ] || [ $pyminor -ge 7 ]; then
		echo "Required: Python3.5 or Python3.6"
		echo "Current: Python$pymajor.$pyminor"
		echo "Please ensure Python3.5 or Python3.6 is the default version when calling 'python3'"
		return 1
	fi
}


function has_installed() {
	if [ ! -d $SCRIPTDIR/vbx_env ]; then
		echo "Initial install of VBX Python Environment required. May take several minutes..."
		(cd $SCRIPTDIR/; bash ./install_venv.sh)
	fi
	source $SCRIPTDIR/vbx_env/bin/activate
	python -c 'import mxnet; import vbx' && command -v converter &> /dev/null && command -v downloader &> /dev/null
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

if has_script && has_python && has_installed; then
	export VBX_SDK=$SCRIPTDIR
	export LD_LIBRARY_PATH=$VBX_SDK/vbx_env/openvino
	echo ""
	echo "export VBX_SDK="$VBX_SDK
	echo "export LD_LIBRARY_PATH="$LD_LIBRARY_PATH
	echo ""

	echo "VBX Python Environment ready. Activating..."
	source $VBX_SDK/vbx_env/bin/activate
	echo "To deactivate VBX Python Environment, run 'deactivate'"
fi
