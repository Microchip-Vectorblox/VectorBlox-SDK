#!/bin/bash
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
function install_venv() {
    $PYTHON_EXE -m venv  $SCRIPTDIR/vbx_env
    source $SCRIPTDIR/vbx_env/bin/activate

	python -m pip install --upgrade "pip==21.3.1" "setuptools==50.3.2"
    python -m pip install wheel 
    python -m pip install -r $SCRIPTDIR/requirements.txt
    python -m pip install -e $SCRIPTDIR/python/vbx
    deactivate
}

function has_python() {
    for v in 3.8 3.6 3.5
    do
        if [ -n "$(which python$v)" ]
        then
            PYTHON_EXE=python$v
            return 0
        fi
    done
	echo "Error Unable to find compatible version of python. Allowed Versions: Python3.5, Python3.6 or Python3.8" >&2
	return 1
}

function has_installed() {
	if [ ! -d $SCRIPTDIR/vbx_env ]; then
		echo "Initial install of VBX Python Environment required. May take several minutes..."
		install_venv
	fi
	source $SCRIPTDIR/vbx_env/bin/activate
	python -c 'import mxnet; import vbx' && command -v mo &> /dev/null && command -v omz_downloader &> /dev/null
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

	echo "VBX Python Environment ready. Activating..."
	source $VBX_SDK/vbx_env/bin/activate
	echo "To deactivate VBX Python Environment, run 'deactivate'"
fi
