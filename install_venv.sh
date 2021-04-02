#!/bin/bash
set -e
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
python3 -m venv vbx_env
source vbx_env/bin/activate

python -m pip install --upgrade "pip==20.3.4" "setuptools==50.3.2"
python -m pip install wheel 
python -m pip install -r $SCRIPTDIR/requirements.txt
python -m pip install -e $SCRIPTDIR/python/third_party/openvino/model-optimizer
python -m pip install -e $SCRIPTDIR/python/third_party/open_model_zoo/tools
python -m pip install -e $SCRIPTDIR/python/vbx


deactivate
 
