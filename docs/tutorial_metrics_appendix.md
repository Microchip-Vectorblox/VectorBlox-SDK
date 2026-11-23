# Tutorial Metrics

Below is a list of tutorials included with the VectorBlox SDK. Runtime in milliseconds (ms) measured on [PolarFire SoC Video Kit](https://github.com/Microchip-Vectorblox/VectorBlox-SoC-Video-Kit-Demo). Accuracy measured over 1000 samples.

<details>

<summary>No Compression Tutorial Metrics</summary>

## No Compression

| Source | Tutorial | Input<br>(H,W,C) | Runtime<br>(ms) | Task | Metric | TFLITE | VNNX |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| PINTO | [016_EfficientNet-lite](../tutorials/PINTO/016_EfficientNet-lite/016_EfficientNet-lite.sh) | [224, 224, 3] | 16.817 | classification | Top1 | 70.8 | 70.4 |
| PINTO | [081_MiDaS_v2](../tutorials/PINTO/081_MiDaS_v2/081_MiDaS_v2.sh) | [256, 256, 3] | 122.564 | depth estimation | depthdelta1 (nyuv2) | 60.89 | 59.93 |
| PINTO | [132_YOLOX](../tutorials/PINTO/132_YOLOX/132_YOLOX.sh) | [416, 416, 3] | 17.685 | object detection | mAP⁵⁰⁻⁹⁵ |  |  |
| PINTO | [307_YOLOv7](../tutorials/PINTO/307_YOLOv7/307_YOLOv7.sh) | [640, 640, 3] | 57.461 | object detection | mAP |  |  |
| darknet | [yolov2-tiny](../tutorials/darknet/yolov2-tiny/yolov2-tiny.sh) | [416, 416, 3] | 33.646 | object detection | mAP⁵⁰⁻⁹⁵ | 10.92 | 11.08 |
| darknet | [yolov3-tiny](../tutorials/darknet/yolov3-tiny/yolov3-tiny.sh) | [416, 416, 3] | 35.674 | object detection | mAP⁵⁰⁻⁹⁵ | 14.57 | 10.88 |
| kaggle | [efficientnet-lite0](../tutorials/kaggle/efficientnet-lite0/efficientnet-lite0.sh) | [224, 224, 3] | 16.684 | classification | Top1 | 70.8 | 70.4 |
| kaggle | [efficientnet-lite4](../tutorials/kaggle/efficientnet-lite4/efficientnet-lite4.sh) | [224, 224, 3] | 83.459 | classification | Top1 | 77.9 | 77.8 |
| mediapipe | [efficientnet_lite0](../tutorials/mediapipe/efficientnet_lite0/efficientnet_lite0.sh) | [224, 224, 3] | 16.811 | classification | Top1 | 70.8 | 70.4 |
| mediapipe | [efficientnet_lite2](../tutorials/mediapipe/efficientnet_lite2/efficientnet_lite2.sh) | [260, 260, 3] | 35.719 | classification | Top1 | 70.8 | 71.1 |
| onnx | [face_detection_yunet_2023mar](../tutorials/onnx/face_detection_yunet_2023mar/face_detection_yunet_2023mar.sh) | [288, 512, 3] | 7.791 | face detection |  |  |  |
| onnx | [face_recognition_sface_2021dec](../tutorials/onnx/face_recognition_sface_2021dec/face_recognition_sface_2021dec.sh) | [112, 112, 3] | 29.56 | face comparison |  |  |  |
| onnx | [mobilenetv2_mcp](../tutorials/onnx/mobilenetv2_mcp/mobilenetv2_mcp.sh) | [112, 112, 3] | 32.63 | face comparison |  |  |  |
| onnx | [onnx_resnet101-v1](../tutorials/onnx/onnx_resnet101-v1/onnx_resnet101-v1.sh) | [224, 224, 3] | 136.284 | classification | Top1 | 74.4 | 74.6 |
| onnx | [onnx_resnet18-v1](../tutorials/onnx/onnx_resnet18-v1/onnx_resnet18-v1.sh) | [224, 224, 3] | 25.982 | classification | Top1 | 69.1 | 68.9 |
| onnx | [onnx_resnet34-v1](../tutorials/onnx/onnx_resnet34-v1/onnx_resnet34-v1.sh) | [224, 224, 3] | 47.427 | classification | Top1 | 72.6 | 72.5 |
| onnx | [onnx_squeezenet1.1](../tutorials/onnx/onnx_squeezenet1.1/onnx_squeezenet1.1.sh) | [224, 224, 3] | 10.191 | classification | Top1 | 54.3 | 54.2 |
| onnx | [yolov7](../tutorials/onnx/yolov7/yolov7.sh) | [640, 640, 3] | 964.431 | object detection | mAP⁵⁰⁻⁹⁵ |  |  |
| onnx | [yolov9-m](../tutorials/onnx/yolov9-m/yolov9-m.sh) | [640, 640, 3] | 739.537 | object detection | mAP⁵⁰⁻⁹⁵ | 52.82 | 52.83 |
| onnx | [yolov9-s](../tutorials/onnx/yolov9-s/yolov9-s.sh) | [640, 640, 3] | 190.326 | object detection | mAP⁵⁰⁻⁹⁵ | 36.39 | 35.8 |
| onnx | [yolov9-s_512x288_argmax](../tutorials/onnx/yolov9-s_512x288_argmax/yolov9-s_512x288_argmax.sh) | [288, 512, 3] | 72.454 | object detection | mAP⁵⁰⁻⁹⁵ | 31.62 | 31.51 |
| onnx | [yolov9-t](../tutorials/onnx/yolov9-t/yolov9-t.sh) | [640, 640, 3] | 74.569 | object detection | mAP⁵⁰⁻⁹⁵ | 31.32 | 31.28 |
| onnx | [yolov9-t_512x288_argmax](../tutorials/onnx/yolov9-t_512x288_argmax/yolov9-t_512x288_argmax.sh) | [288, 512, 3] | 29.158 | object detection | mAP⁵⁰⁻⁹⁵ | 27.75 | 27.63 |
| openvino | [mobilenet-v1-0.25-128](../tutorials/openvino/mobilenet-v1-0.25-128/mobilenet-v1-0.25-128.sh) | [128, 128, 3] | 1.516 | classification | Top1 | 20.8 | 20.9 |
| openvino | [mobilenet-v1-1.0-224](../tutorials/openvino/mobilenet-v1-1.0-224/mobilenet-v1-1.0-224.sh) | [224, 224, 3] | 11.814 | classification | Top1 | 70.1 | 70.1 |
| openvino | [mobilenet-v1-1.0-224-tf](../tutorials/openvino/mobilenet-v1-1.0-224-tf/mobilenet-v1-1.0-224-tf.sh) | [224, 224, 3] | 11.654 | classification | Top1 | 69.7 | 69.9 |
| openvino | [mobilenet-v2](../tutorials/openvino/mobilenet-v2/mobilenet-v2.sh) | [224, 224, 3] | 15.688 | classification | Top1 | 72.5 | 72.9 |
| openvino | [mobilenet-v2-1.0-224](../tutorials/openvino/mobilenet-v2-1.0-224/mobilenet-v2-1.0-224.sh) | [224, 224, 3] | 13.251 | classification | Top1 | 70.5 | 70.5 |
| openvino | [mobilenet-v2-1.4-224](../tutorials/openvino/mobilenet-v2-1.4-224/mobilenet-v2-1.4-224.sh) | [224, 224, 3] | 20.575 | classification | Top1 | 75.3 | 75.2 |
| openvino | [squeezenet1.0](../tutorials/openvino/squeezenet1.0/squeezenet1.0.sh) | [227, 227, 3] | 18.371 | classification | Top1 | 56.9 | 56.8 |
| openvino | [squeezenet1.1](../tutorials/openvino/squeezenet1.1/squeezenet1.1.sh) | [227, 227, 3] | 11.34 | classification | Top1 | 57.0 | 56.7 |
| pytorch | [anomaly_transformer](../tutorials/pytorch/anomaly_transformer/anomaly_transformer.sh) | [100, 38, 1] | 61.002 |  |  |  |  |
| pytorch | [lpr_eu_v3](../tutorials/pytorch/lpr_eu_v3/lpr_eu_v3.sh) | [34, 146, 3] | 4.355 | plate recognition |  |  |  |
| pytorch | [spnv2](../tutorials/pytorch/spnv2/spnv2.sh) | [512, 768, 3] |  | 6d pose |  |  |  |
| pytorch | [torchvision_inception_v3](../tutorials/pytorch/torchvision_inception_v3/torchvision_inception_v3.sh) | [299, 299, 3] | 139.639 | classification | Top1 | 78.2 | 75.8 |
| pytorch | [torchvision_resnet18](../tutorials/pytorch/torchvision_resnet18/torchvision_resnet18.sh) | [224, 224, 3] | 25.724 | classification | Top1 | 69.0 | 68.6 |
| pytorch | [torchvision_resnet50](../tutorials/pytorch/torchvision_resnet50/torchvision_resnet50.sh) | [224, 224, 3] | 81.979 | classification | Top1 | 80.7 | 80.6 |
| pytorch | [torchvision_squeezenet1_0](../tutorials/pytorch/torchvision_squeezenet1_0/torchvision_squeezenet1_0.sh) | [227, 227, 3] | 16.971 | classification | Top1 | 58.9 | 58.6 |
| pytorch | [torchvision_ssdlite320_mobilenet_v3_large](../tutorials/pytorch/torchvision_ssdlite320_mobilenet_v3_large/torchvision_ssdlite320_mobilenet_v3_large.sh) | [320, 320, 3] | 38.946 | object detection | mAP⁵⁰⁻⁹⁵ |  |  |
| pytorch | [torchvision_vit_sym](../tutorials/pytorch/torchvision_vit_sym/torchvision_vit_sym.sh) | [224, 224, 3] | 770.176 | classification | Top1 | 71.9 | 70.5 |
| pytorch | [torchvision_wide_resnet50_2](../tutorials/pytorch/torchvision_wide_resnet50_2/torchvision_wide_resnet50_2.sh) | [224, 224, 3] | 201.961 | classification | Top1 | 80.5 | 81.0 |
| qualcomm | [DeepLabV3-Plus-MobileNet-Quantized](../tutorials/qualcomm/DeepLabV3-Plus-MobileNet-Quantized/DeepLabV3-Plus-MobileNet-Quantized.sh) | [520, 520, 3] | 727.667 | segmentation | meanIoU (VOC2012) | 59.72 | 60.28 |
| qualcomm | [DeepLabV3-Plus-MobileNet_512x288](../tutorials/qualcomm/DeepLabV3-Plus-MobileNet_512x288/DeepLabV3-Plus-MobileNet_512x288.sh) | [288, 512, 3] | 312.996 | segmentation | meanIoU (VOC2012) | 52.73 | 54.92 |
| qualcomm | [FFNet-122NS-LowRes](../tutorials/qualcomm/FFNet-122NS-LowRes/FFNet-122NS-LowRes.sh) | [512, 1024, 3] | 202.413 | segmentation | meanIoU (cityscapes) | 61.75 | 61.77 |
| qualcomm | [FFNet-122NS-LowRes_512x288](../tutorials/qualcomm/FFNet-122NS-LowRes_512x288/FFNet-122NS-LowRes_512x288.sh) | [288, 512, 3] | 76.451 | segmentation | meanIoU (cityscapes) | 48.34 | 48.7 |
| qualcomm | [FFNet-78S-LowRes](../tutorials/qualcomm/FFNet-78S-LowRes/FFNet-78S-LowRes.sh) | [512, 1024, 3] | 247.88 | segmentation | meanIoU (cityscapes) | 63.17 | 63.39 |
| qualcomm | [FFNet-78S-LowRes_512x288](../tutorials/qualcomm/FFNet-78S-LowRes_512x288/FFNet-78S-LowRes_512x288.sh) | [288, 512, 3] | 91.37 | segmentation | meanIoU (cityscapes) | 49.83 | 50.45 |
| qualcomm | [GoogLeNetQuantized](../tutorials/qualcomm/GoogLeNetQuantized/GoogLeNetQuantized.sh) | [224, 224, 3] | 30.74 | classification | Top1 | 68.7 | 68.5 |
| qualcomm | [Midas-V2-Quantized](../tutorials/qualcomm/Midas-V2-Quantized/Midas-V2-Quantized.sh) | [256, 256, 3] | 122.543 | depth estimation | depthdelta1 (nyuv2) | 85.06 | 81.06 |
| qualcomm | [Midas-V2_256x128](../tutorials/qualcomm/Midas-V2_256x128/Midas-V2_256x128.sh) | [128, 256, 3] | 75.294 | depth estimation | depthdelta1 (nyuv2) | 81.83 | 77.95 |
| qualcomm | [MobileNet-v2-Quantized](../tutorials/qualcomm/MobileNet-v2-Quantized/MobileNet-v2-Quantized.sh) | [224, 224, 3] | 12.916 | classification | Top1 | 66.7 | 66.5 |
| qualcomm | [MobileNet-v3-Large-Quantized](../tutorials/qualcomm/MobileNet-v3-Large-Quantized/MobileNet-v3-Large-Quantized.sh) | [224, 224, 3] | 23.02 | classification | Top1 | 68.9 | 68.8 |
| qualcomm | [QuickSRNetLarge-Quantized](../tutorials/qualcomm/QuickSRNetLarge-Quantized/QuickSRNetLarge-Quantized.sh) | [128, 128, 3] | 63.888 | image enhancement | PSNR (bsd300) | 27.13 | 27.1 |
| qualcomm | [QuickSRNetMedium-Quantized](../tutorials/qualcomm/QuickSRNetMedium-Quantized/QuickSRNetMedium-Quantized.sh) | [128, 128, 3] | 9.367 | image enhancement | PSNR (bsd300) | 26.79 | 26.81 |
| qualcomm | [QuickSRNetSmall-Quantized](../tutorials/qualcomm/QuickSRNetSmall-Quantized/QuickSRNetSmall-Quantized.sh) | [128, 128, 3] | 5.951 | image enhancement | PSNR (bsd300) | 26.95 | 26.95 |
| qualcomm | [ResNet101Quantized](../tutorials/qualcomm/ResNet101Quantized/ResNet101Quantized.sh) | [224, 224, 3] | 140.6 | classification | Top1 | 75.6 | 76.0 |
| qualcomm | [ResNet18Quantized](../tutorials/qualcomm/ResNet18Quantized/ResNet18Quantized.sh) | [224, 224, 3] | 27.399 | classification | Top1 | 68.8 | 68.8 |
| qualcomm | [ResNet50Quantized](../tutorials/qualcomm/ResNet50Quantized/ResNet50Quantized.sh) | [224, 224, 3] | 85.593 | classification | Top1 | 77.0 | 77.0 |
| qualcomm | [SESR-M5-Quantized](../tutorials/qualcomm/SESR-M5-Quantized/SESR-M5-Quantized.sh) | [128, 128, 3] | 60.177 | image enhancement | PSNR (bsd300) | 25.41 | 25.39 |
| qualcomm | [VITQuantized](../tutorials/qualcomm/VITQuantized/VITQuantized.sh) | [224, 224, 3] | 1180.194 | classification | Top1 | 77.4 | 75.8 |
| qualcomm | [WideResNet50-Quantized](../tutorials/qualcomm/WideResNet50-Quantized/WideResNet50-Quantized.sh) | [224, 224, 3] | 201.959 | classification | Top1 | 77.2 | 77.1 |
| qualcomm | [XLSR-Quantized](../tutorials/qualcomm/XLSR-Quantized/XLSR-Quantized.sh) | [128, 128, 3] | 7.599 | image enhancement | PSNR (bsd300) | 27.23 | 27.23 |
| tensorflow | [efficientnet-lite0-int8](../tutorials/tensorflow/efficientnet-lite0-int8/efficientnet-lite0-int8.sh) | [224, 224, 3] | 16.519 | classification | Top1 | 70.8 | 70.4 |
| tensorflow | [efficientnet-lite1-int8](../tutorials/tensorflow/efficientnet-lite1-int8/efficientnet-lite1-int8.sh) | [240, 240, 3] | 23.942 | classification | Top1 | 72.0 | 71.1 |
| tensorflow | [efficientnet-lite2-int8](../tutorials/tensorflow/efficientnet-lite2-int8/efficientnet-lite2-int8.sh) | [260, 260, 3] | 35.816 | classification | Top1 | 70.8 | 71.1 |
| tensorflow | [efficientnet-lite3-int8](../tutorials/tensorflow/efficientnet-lite3-int8/efficientnet-lite3-int8.sh) | [280, 280, 3] | 49.413 | classification | Top1 | 76.3 | 76.1 |
| tensorflow | [efficientnet-lite4-int8](../tutorials/tensorflow/efficientnet-lite4-int8/efficientnet-lite4-int8.sh) | [300, 300, 3] | 83.438 | classification | Top1 | 77.9 | 77.8 |
| tensorflow | [mobilenet_v1_050_160](../tutorials/tensorflow/mobilenet_v1_050_160/mobilenet_v1_050_160.sh) | [160, 160, 3] | 3.379 | classification | Top1 | 49.6 | 49.9 |
| tensorflow | [mobilenet_v2](../tutorials/tensorflow/mobilenet_v2/mobilenet_v2.sh) | [224, 224, 3] | 13.085 | classification | Top1 | 70.6 | 70.1 |
| tensorflow | [mobilenet_v2_140_224](../tutorials/tensorflow/mobilenet_v2_140_224/mobilenet_v2_140_224.sh) | [224, 224, 3] | 20.591 | classification | Top1 | 75.4 | 75.4 |
| tensorflow | [posenet](../tutorials/tensorflow/posenet/posenet.sh) | [273, 481, 3] | 41.357 | pose detection |  |  |  |
| tensorflow | [yolo-v3-tf](../tutorials/tensorflow/yolo-v3-tf/yolo-v3-tf.sh) | [416, 416, 3] | 511.477 | object detection | mAP⁵⁰⁻⁹⁵ | 34.48 | 34.54 |
| tensorflow | [yolo-v3-tiny-tf](../tutorials/tensorflow/yolo-v3-tiny-tf/yolo-v3-tiny-tf.sh) | [416, 416, 3] | 35.765 | object detection | mAP⁵⁰⁻⁹⁵ | 14.58 | 11.0 |
| tensorflow | [yolo-v4-tiny-tf](../tutorials/tensorflow/yolo-v4-tiny-tf/yolo-v4-tiny-tf.sh) | [416, 416, 3] | 46.499 | object detection | mAP⁵⁰⁻⁹⁵ | 16.83 | 12.29 |
| ultralytics | [yolo11n](../tutorials/ultralytics/yolo11n/yolo11n.sh) | [640, 640, 3] | 83.199 | object detection | mAP⁵⁰⁻⁹⁵ | 40.45 | 40.29 |
| ultralytics | [yolo11n_sym](../tutorials/ultralytics/yolo11n_sym/yolo11n_sym.sh) | [640, 640, 3] | 70.54 | object detection | mAP⁵⁰⁻⁹⁵ | 39.02 | 38.85 |
| ultralytics | [yolo26n](../tutorials/ultralytics/yolo26n/yolo26n.sh) | [640, 640, 3] | 110.012 | object detection | mAP⁵⁰⁻⁹⁵ | 40.32 | 39.99 |
| ultralytics | [yolo26n_sym](../tutorials/ultralytics/yolo26n_sym/yolo26n_sym.sh) | [640, 640, 3] | 80.967 | object detection | mAP⁵⁰⁻⁹⁵ | 40.72 | 41.11 |
| ultralytics | [yolov8l-cls](../tutorials/ultralytics/yolov8l-cls/yolov8l-cls.sh) | [224, 224, 3] | 99.381 | classification |  |  |  |
| ultralytics | [yolov8m](../tutorials/ultralytics/yolov8m/yolov8m.sh) | [640, 640, 3] | 578.69 | object detection | mAP⁵⁰⁻⁹⁵ | 52.13 | 52.16 |
| ultralytics | [yolov8m-cls](../tutorials/ultralytics/yolov8m-cls/yolov8m-cls.sh) | [224, 224, 3] | 39.034 | classification | Top1 | 75.9 | 75.9 |
| ultralytics | [yolov8m_FULL](../tutorials/ultralytics/yolov8m_FULL/yolov8m_FULL.sh) | [640, 640, 3] | 624.703 | object detection | mAP⁵⁰⁻⁹⁵ | 47.21 | 47.37 |
| ultralytics | [yolov8n](../tutorials/ultralytics/yolov8n/yolov8n.sh) | [640, 640, 3] | 55.727 | object detection | mAP⁵⁰⁻⁹⁵ | 37.38 | 37.42 |
| ultralytics | [yolov8n-cls](../tutorials/ultralytics/yolov8n-cls/yolov8n-cls.sh) | [224, 224, 3] | 4.24 | classification | Top1 | 67.2 | 67.3 |
| ultralytics | [yolov8n-obb](../tutorials/ultralytics/yolov8n-obb/yolov8n-obb.sh) | [1024, 1024, 3] | 146.691 | obb detection | mAP⁵⁰⁻⁹⁵ |  |  |
| ultralytics | [yolov8n-pose](../tutorials/ultralytics/yolov8n-pose/yolov8n-pose.sh) | [640, 640, 3] | 59.619 | pose detection | Pose Detection |  |  |
| ultralytics | [yolov8n-pose_512x288](../tutorials/ultralytics/yolov8n-pose_512x288/yolov8n-pose_512x288.sh) | [288, 512, 3] | 22.747 | pose detection | Pose Detection |  |  |
| ultralytics | [yolov8n-pose_512x288_split](../tutorials/ultralytics/yolov8n-pose_512x288_split/yolov8n-pose_512x288_split.sh) | [288, 512, 3] | 22.478 | pose detection | Pose Detection |  |  |
| ultralytics | [yolov8n-seg](../tutorials/ultralytics/yolov8n-seg/yolov8n-seg.sh) | [640, 640, 3] | 72.522 | instance segmentation |  |  |  |
| ultralytics | [yolov8n_512x288](../tutorials/ultralytics/yolov8n_512x288/yolov8n_512x288.sh) | [288, 512, 3] | 21.643 | object detection | mAP⁵⁰⁻⁹⁵ | 28.2 | 28.15 |
| ultralytics | [yolov8n_512x288_argmax](../tutorials/ultralytics/yolov8n_512x288_argmax/yolov8n_512x288_argmax.sh) | [288, 512, 3] | 22.028 | object detection | mAP⁵⁰⁻⁹⁵ | 28.2 | 28.15 |
| ultralytics | [yolov8n_FULL](../tutorials/ultralytics/yolov8n_FULL/yolov8n_FULL.sh) | [640, 640, 3] | 101.736 | object detection | mAP⁵⁰⁻⁹⁵ | 35.25 | 35.34 |
| ultralytics | [yolov8n_argmax](../tutorials/ultralytics/yolov8n_argmax/yolov8n_argmax.sh) | [640, 640, 3] | 56.954 | object detection | mAP⁵⁰⁻⁹⁵ | 37.31 | 37.31 |
| ultralytics | [yolov8s](../tutorials/ultralytics/yolov8s/yolov8s.sh) | [640, 640, 3] | 204.151 | object detection | mAP⁵⁰⁻⁹⁵ | 46.99 | 46.92 |
| ultralytics | [yolov8s-cls](../tutorials/ultralytics/yolov8s-cls/yolov8s-cls.sh) | [224, 224, 3] | 11.995 | classification | Top1 | 72.6 | 72.5 |
| ultralytics | [yolov8s_FULL](../tutorials/ultralytics/yolov8s_FULL/yolov8s_FULL.sh) | [640, 640, 3] | 250.175 | object detection | mAP⁵⁰⁻⁹⁵ | 43.9 | 43.72 |
| ultralytics | [yolov8x-cls](../tutorials/ultralytics/yolov8x-cls/yolov8x-cls.sh) | [224, 224, 3] | 280.525 | classification |  |  |  |
| ultralytics | [yolov9s](../tutorials/ultralytics/yolov9s/yolov9s.sh) | [640, 640, 3] | 175.855 | object detection | mAP⁵⁰⁻⁹⁵ | 47.99 | 47.93 |
| ultralytics | [yolov9s_FULL](../tutorials/ultralytics/yolov9s_FULL/yolov9s_FULL.sh) | [640, 640, 3] | 222.277 | object detection | mAP⁵⁰⁻⁹⁵ | 44.52 | 44.35 |
| ultralytics | [yolov9t](../tutorials/ultralytics/yolov9t/yolov9t.sh) | [640, 640, 3] | 67.155 | object detection | mAP⁵⁰⁻⁹⁵ | 37.88 | 38.2 |
| ultralytics | [yolov9t_FULL](../tutorials/ultralytics/yolov9t_FULL/yolov9t_FULL.sh) | [640, 640, 3] | 113.097 | object detection | mAP⁵⁰⁻⁹⁵ | 35.43 | 35.79 |
| vectorblox | [yolov9m-spn](../tutorials/vectorblox/yolov9m-spn/yolov9m-spn.sh) | [512, 768, 3] | 873.016 | 6d pose |  |  |  |
| vectorblox | [yolov9s-spn](../tutorials/vectorblox/yolov9s-spn/yolov9s-spn.sh) | [512, 768, 3] | 216.766 | 6d pose |  |  |  |
| vectorblox | [yolov9s-spn_800x480](../tutorials/vectorblox/yolov9s-spn_800x480/yolov9s-spn_800x480.sh) | [480, 800, 3] | 208.538 | 6d pose |  |  |  |
| vectorblox | [yolov9t-spn](../tutorials/vectorblox/yolov9t-spn/yolov9t-spn.sh) | [512, 768, 3] | 78.265 | 6d pose |  |  |  |
| vectorblox | [yolov9t-spn_800x480](../tutorials/vectorblox/yolov9t-spn_800x480/yolov9t-spn_800x480.sh) | [480, 800, 3] | 77.755 | 6d pose |  |  |  |

</details>

<details>

<summary>Compression Tutorial Metrics</summary>

## Compression

| Source | Tutorial | Input<br>(H,W,C) | Runtime<br>(ms) | Task | Metric | TFLITE | VNNX |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| compressed | [yolov8m_comp69](../tutorials/compressed/yolov8m_comp69/yolov8m_comp69.sh) | [640, 640, 3] | 239.842 | object detection | mAP⁵⁰⁻⁹⁵ | 51.82 | 51.5 |
| compressed | [yolov8n_comp66](../tutorials/compressed/yolov8n_comp66/yolov8n_comp66.sh) | [640, 640, 3] | 37.053 | object detection | mAP⁵⁰⁻⁹⁵ | 37.18 | 37.3 |
| compressed | [yolov8s_comp68](../tutorials/compressed/yolov8s_comp68/yolov8s_comp68.sh) | [640, 640, 3] | 95.119 | object detection | mAP⁵⁰⁻⁹⁵ | 46.11 | 46.15 |
| compressed | [yolov9m_comp67](../tutorials/compressed/yolov9m_comp67/yolov9m_comp67.sh) | [640, 640, 3] | 322.171 | object detection | mAP⁵⁰⁻⁹⁵ | 51.69 | 51.91 |
| compressed | [yolov9s_comp68](../tutorials/compressed/yolov9s_comp68/yolov9s_comp68.sh) | [640, 640, 3] | 122.47 | object detection | mAP⁵⁰⁻⁹⁵ | 46.99 | 47.01 |
| compressed | [yolov9t_comp61](../tutorials/compressed/yolov9t_comp61/yolov9t_comp61.sh) | [640, 640, 3] | 55.879 | object detection | mAP⁵⁰⁻⁹⁵ | 38.14 | 38.35 |

</details>


<details>

<summary>Unstructured Compression Tutorial Metrics</summary>

## Unstructured Compression

| Source | Tutorial | Input<br>(H,W,C) | Runtime<br>(ms) | Task | Metric | TFLITE | UCOMP|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| unstructure_compressed | [resnet18_86s_07p](../tutorials/unstructure_compressed/resnet18_86s_07p/resnet18_86s_07p.sh) | [224, 224, 3] | 13.591 | classification | Top1 | 65.6 | 65.6 |
| unstructure_compressed | [yolov5n_70s_512x512](../tutorials/unstructure_compressed/yolov5n_70s_512x512/yolov5n_70s_512x512.sh) | [512, 512, 3] | 19.744 | object detection | mAP⁵⁰⁻⁹⁵ | 20.2 | 19.97 |
| unstructure_compressed | [yolov8n_50s_25p_512x288](../tutorials/unstructure_compressed/yolov8n_50s_25p_512x288/yolov8n_50s_25p_512x288.sh) | [288, 512, 3] | 15.474 | object detection | mAP⁵⁰⁻⁹⁵ | 34.6 |  |
| unstructure_compressed | [yolov8n_50s_25p_cut](../tutorials/unstructure_compressed/yolov8n_50s_25p_cut/yolov8n_50s_25p_cut.sh) | [640, 640, 3] | 48.049 | object detection | mAP⁵⁰⁻⁹⁵ | 36.7 | 36.64 |
| unstructure_compressed | [yolov8n_pose_50s_25p_512x288_split](../tutorials/unstructure_compressed/yolov8n_pose_50s_25p_512x288_split/yolov8n_pose_50s_25p_512x288_split.sh) | [288, 512, 3] | 15.406 | pose detection |  |  |  |
| unstructure_compressed | [yolov9s_70s_15p_512x288](../tutorials/unstructure_compressed/yolov9s_70s_15p_512x288/yolov9s_70s_15p_512x288.sh) | [288, 512, 3] | 34.463 | object detection | mAP⁵⁰⁻⁹⁵ | 43.2 |  |
| unstructure_compressed | [yolov9s_70s_15p_cut](../tutorials/unstructure_compressed/yolov9s_70s_15p_cut/yolov9s_70s_15p_cut.sh) | [640, 640, 3] | 98.857 | object detection | mAP⁵⁰⁻⁹⁵ | 45.8 | 45.48 |

</details>