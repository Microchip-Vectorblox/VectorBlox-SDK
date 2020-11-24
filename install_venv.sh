#!/bin/bash
set -e
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
python3 -m venv vbx_env
source vbx_env/bin/activate

python -m pip install --upgrade pip setuptools
python -m pip install wheel 
python -m pip install -r $SCRIPTDIR/requirements.txt
python -m pip install -e $SCRIPTDIR/python/third_party/dldt/model-optimizer
python -m pip install -e $SCRIPTDIR/python/third_party/open_model_zoo/tools
python -m pip install -e $SCRIPTDIR/python/vbx
python -m pip install --no-index --find-links $SCRIPTDIR/python/third_party/wheels mxnet


deactivate
 
