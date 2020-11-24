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

import importlib.util
import logging as log
import os
import re
import sys

# WA for abseil bug that affects logging while importing TF starting 1.14 version
# Link to original issue: https://github.com/abseil/abseil-py/issues/99
if importlib.util.find_spec('absl') is not None:
    import absl.logging

    log.root.removeHandler(absl.logging._absl_handler)

handler_num = 0


class LvlFormatter(log.Formatter):
    format_dict = {
        log.DEBUG: "[ %(asctime)s ] [ %(levelname)s ] [ %(module)s:%(lineno)d ]  %(msg)s",
        log.INFO: "[ %(levelname)s ]  %(msg)s",
        log.WARNING: "[ WARNING ]  %(msg)s",
        log.ERROR: "[ %(levelname)s ]  %(msg)s",
        log.CRITICAL: "[ %(levelname)s ]  %(msg)s",
        'framework_error': "[ FRAMEWORK ERROR ]  %(msg)s",
        'analysis_info': "[ ANALYSIS INFO ]  %(msg)s"
    }

    def __init__(self, lvl, fmt=None):
        log.Formatter.__init__(self, fmt)
        self.lvl = lvl

    def format(self, record: log.LogRecord):
        if self.lvl == 'DEBUG':
            self._style._fmt = self.format_dict[log.DEBUG]
        else:
            self._style._fmt = self.format_dict[record.levelno]
        if 'is_warning' in record.__dict__.keys():
            self._style._fmt = self.format_dict[log.WARNING]
        if 'framework_error' in record.__dict__.keys():
            self._style._fmt = self.format_dict['framework_error']
        if 'analysis_info' in record.__dict__.keys():
            self._style._fmt = self.format_dict['analysis_info']
        return log.Formatter.format(self, record)


class TagFilter(log.Filter):
    def __init__(self, regex: str):
        self.regex = regex

    def filter(self, record: log.LogRecord):
        if record.__dict__['funcName'] == 'load_grammar':  # for nx not to log into our logs
            return False
        if self.regex:
            if 'tag' in record.__dict__.keys():
                tag = record.__dict__['tag']
                return re.findall(self.regex, tag)
            else:
                return False
        return True  # if regex wasn't set print all logs


def init_logger(lvl: str, silent: bool):
    global handler_num
    log_exp = os.environ.get('MO_LOG_PATTERN')
    if silent:
        lvl = 'ERROR'
    fmt = LvlFormatter(lvl=lvl)
    handler = log.StreamHandler()
    handler.setFormatter(fmt)
    logger = log.getLogger()
    logger.setLevel(lvl)
    logger.addFilter(TagFilter(regex=log_exp))
    if handler_num == 0:
        logger.addHandler(handler)
        handler_num += 1


def progress_bar(function: callable):
    """
    Decorator for model conversion pipeline progress display
    Works in combination with function: mo.utils.class_registration.apply_transform
    """

    def wrapper(*args, **kwargs):
        for arg in ['graph', 'curr_transform_num', 'num_transforms']:
            msg = 'Progress bar decorator is enabled for Model Optimizer transformation applying cycle only. ' \
                  'Argument `{}` {}'

            assert arg in kwargs, msg.format(arg, 'is missing')
            assert kwargs[arg] is not None, msg.format(arg, 'should not be None')

        if kwargs['graph'].graph['cmd_params'].progress:
            bar_len = 20
            total_replacers_count = kwargs['num_transforms']

            def progress(i):
                return int((i + 1) / total_replacers_count * bar_len)

            def percent(i):
                return (i + 1) / total_replacers_count * 100

            end = '' if not kwargs['graph'].graph['cmd_params'].stream_output else '\n'
            curr_i = kwargs['curr_transform_num']
            print('\rProgress: [{:{}}]{:>7.2f}% done'.format('.' * progress(curr_i), bar_len, percent(curr_i)), end=end)

            sys.stdout.flush()

        function(*args, **kwargs)

    return wrapper
