# Third-Party Software Notices and Licenses

The VectorBlox SDK includes or depends on the following third-party software
components. Each component is subject to its own license terms as described
below.

---

## Bundled Third-Party Libraries

These libraries are included in the SDK source tree.

### libfixmath

- **License:** MIT License
- **Copyright:** (c) 2011-2021 Flatmush, Petteri Aimonen, & libfixmath AUTHORS

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

### libfixmatrix

- **License:** MIT License
- **Copyright:** (c) 2011-2012 Petteri Aimonen

```
Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

### TensorFlow Lite FlatBuffers Schema

- **License:** Apache License 2.0
- **Copyright:** (c) 2017 The TensorFlow Authors

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## Python Package Dependencies

The following Python packages are installed as dependencies (not bundled in
source). Their licenses are listed here for informational purposes.

### Apache License 2.0

| Package | Version | Copyright |
|---------|---------|-----------|
| onnx | 1.16.1 | The Linux Foundation |
| opencv-python | 4.7.x | OpenCV team |
| tensorflow-cpu | 2.15.1 | The TensorFlow Authors |
| tensorflow_datasets | 4.9.3 | The TensorFlow Authors |
| openvino | 2023.0.1 | Intel Corporation |
| openvino-dev | 2023.0.1 | Intel Corporation |
| openvino-telemetry | 2023.2.1 | Intel Corporation |
| onnxsim | 0.4.36 | daquexian |
| onnx-graphsurgeon | - | NVIDIA Corporation |
| ml_dtypes | 0.3.1 | The ml_dtypes Authors |
| tf_keras | 2.15.1 | The TensorFlow Authors |
| tflite_support | 0.4.4 | The TensorFlow Authors |

### MIT License

| Package | Version | Copyright |
|---------|---------|-----------|
| onnxruntime | 1.18.1 | Microsoft Corporation |
| pip | <23.1 | The pip developers |
| setuptools | <=68.2.2 | Python Packaging Authority |
| wheel | - | Python Packaging Authority |
| sor4onnx | - | PINTO0309 |
| sne4onnx | - | PINTO0309 |
| sng4onnx | 1.0.1 | PINTO0309 |
| onnx2tf | 1.22.3 | PINTO0309 |
| openvino2tensorflow | 1.34.0 | PINTO0309 |
| silence_tensorflow | - | Various |
| natsort | - | Seth M. Morton |
| onnxslim | 0.1.32 | Various |
| prtpy | 0.8.1 | Various |

### BSD-3-Clause License

| Package | Version | Copyright |
|---------|---------|-----------|
| numpy | 1.23.5 | NumPy Developers |
| protobuf | 3.20.3 | Google LLC |
| psutil | 5.9.5 | Giampaolo Rodola |
| torch | 2.3.0+cpu | Meta Platforms, Inc. |
| torchvision | 0.18.0+cpu | Meta Platforms, Inc. |

### BSD-2-Clause License

| Package | Version | Copyright |
|---------|---------|-----------|
| XlsxWriter | - | John McNamara |

### Other Licenses

| Package | Version | License | Copyright |
|---------|---------|---------|-----------|
| tqdm | - | MPL 2.0 / MIT | Various |
| matplotlib | - | PSF License (BSD-compatible) | Matplotlib Development Team |
| orjson | - | MIT / Apache 2.0 | ijl |
| posix-ipc | - | BSD | Philip Semanchuk |
| ultralytics | 8.3.72 | AGPL-3.0 (development tool only, not distributed) | Ultralytics Inc. |

---

## Notes

1. **ultralytics** is used exclusively as an offline development tool for model
   export (via the `yolo` CLI command). It is not imported by any distributed
   code, is not a runtime dependency, and no ultralytics source code is
   included in this repository.
   Ultralytics YOLO models are available under the AGPL-3.0 open-source license.
   Projects that are not open source require an Ultralytics Enterprise License.
   To obtain a commercial license for R&D and production use without open-source
   obligations, please complete the licensing form at https://www.ultralytics.com/license.

2. Python dependencies listed above are installed via `pip` at setup time and
   are governed by their respective upstream licenses. They are not bundled
   with this SDK.

---

## Microchip Technology Inc.

Copyright (c) 2020-2026 Microchip Technology Inc. All rights reserved.

For the VectorBlox SDK proprietary components, see [LICENSE.md](LICENSE.md).
