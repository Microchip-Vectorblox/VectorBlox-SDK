#!/usr/bin/env python3

"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import warnings
import sys

from mo.utils.versions_checker import check_python_version  # pylint: disable=no-name-in-module
def mo_main():
    sys.stderr.write("WARNING: 'mo' command deprecated in favour of 'converter'. 'mo' will be removed in a future release'\n")
    converter_main()
def converter_main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    ret_code = check_python_version()
    if ret_code:
        sys.exit(ret_code)

    from mo.main import main
    from mo.utils.cli_parser import get_all_cli_parser  # pylint: disable=no-name-in-module

    sys.exit(main(get_all_cli_parser(), None))

if __name__ == "__main__":
    main()
