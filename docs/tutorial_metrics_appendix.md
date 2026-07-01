# Tutorial Metrics

Below is a list of tutorials included with the VectorBlox SDK. Runtime in milliseconds (ms) measured on [PolarFire SoC Video Kit](https://github.com/Microchip-Vectorblox/VectorBlox-SoC-Video-Kit-Demo). Accuracy measured over 1000 samples.

<details>

<summary>No Compression Tutorial Metrics</summary>

## No Compression

| Source | Tutorial | Input<br>(H,W,C) | Runtime<br>(ms) | Task | Metric | TFLITE | VNNX |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| PINTO | [016_EfficientNet-lite](PINTO/016_EfficientNet-lite/016_EfficientNet-lite.sh) | [224, 224, 3] | 16.464 | classification | Top1 | 70.8 | 70.4 |
| PINTO | [046_yolov4-tiny](PINTO/046_yolov4-tiny/046_yolov4-tiny.sh) | [416, 416, 3] | 46.062 | object detection | mAP⁵⁰⁻⁹⁵ |  |  |
| PINTO | [081_MiDaS_v2](PINTO/081_MiDaS_v2/081_MiDaS_v2.sh) | [256, 256, 3] | 121.921 | depth estimation | depthdelta1 (nyuv2) | 60.89 | 59.93 |
| PINTO | [132_YOLOX](PINTO/132_YOLOX/132_YOLOX.sh) | [416, 416, 3] | 17.226 | object detection | mAP⁵⁰⁻⁹⁵ |  |  |
| PINTO | [307_YOLOv7](PINTO/307_YOLOv7/307_YOLOv7.sh) | [640, 640, 3] | 56.941 | object detection | mAP |  |  |
| darknet | [yolov2-tiny](darknet/yolov2-tiny/yolov2-tiny.sh) | [416, 416, 3] | 33.459 | object detection | mAP⁵⁰⁻⁹⁵ | 10.92 | 11.08 |
| darknet | [yolov3-tiny](darknet/yolov3-tiny/yolov3-tiny.sh) | [416, 416, 3] | 35.527 | object detection | mAP⁵⁰⁻⁹⁵ | 10.93 | 10.88 |
| kaggle | [efficientnet-lite0](kaggle/efficientnet-lite0/efficientnet-lite0.sh) | [224, 224, 3] | 16.481 | classification | Top1 | 70.8 | 70.4 |
| kaggle | [efficientnet-lite4](kaggle/efficientnet-lite4/efficientnet-lite4.sh) | [224, 224, 3] | 82.484 | classification | Top1 | 77.9 | 77.8 |
| mediapipe | [efficientnet_lite0](mediapipe/efficientnet_lite0/efficientnet_lite0.sh) | [224, 224, 3] | 16.474 | classification | Top1 | 70.8 | 70.4 |
| mediapipe | [efficientnet_lite2](mediapipe/efficientnet_lite2/efficientnet_lite2.sh) | [260, 260, 3] | 35.412 | classification | Top1 | 70.8 | 71.1 |
| onnx | [onnx_resnet101-v1](onnx/onnx_resnet101-v1/onnx_resnet101-v1.sh) | [224, 224, 3] | 134.658 | classification | Top1 | 74.4 | 74.6 |
| onnx | [onnx_resnet18-v1](onnx/onnx_resnet18-v1/onnx_resnet18-v1.sh) | [224, 224, 3] | 25.741 | classification | Top1 | 69.1 | 68.9 |
| onnx | [onnx_resnet34-v1](onnx/onnx_resnet34-v1/onnx_resnet34-v1.sh) | [224, 224, 3] | 46.933 | classification | Top1 | 72.6 | 72.5 |
| onnx | [onnx_squeezenet1.1](onnx/onnx_squeezenet1.1/onnx_squeezenet1.1.sh) | [224, 224, 3] | 10.012 | classification | Top1 | 54.3 | 54.2 |
| onnx | [yolov7](onnx/yolov7/yolov7.sh) | [640, 640, 3] | 950.586 | object detection | mAP⁵⁰⁻⁹⁵ |  |  |
| onnx | [yolov9-m](onnx/yolov9-m/yolov9-m.sh) | [640, 640, 3] | 718.413 | object detection | mAP⁵⁰⁻⁹⁵ | 52.82 | 52.83 |
| onnx | [yolov9-s](onnx/yolov9-s/yolov9-s.sh) | [640, 640, 3] | 182.448 | object detection | mAP⁵⁰⁻⁹⁵ | 36.39 | 35.8 |
| onnx | [yolov9-s_512x288_argmax](onnx/yolov9-s_512x288_argmax/yolov9-s_512x288_argmax.sh) | [288, 512, 3] | 69.897 | object detection | mAP⁵⁰⁻⁹⁵ | 31.48 | 31.44 |
| onnx | [yolov9-t](onnx/yolov9-t/yolov9-t.sh) | [640, 640, 3] | 72.596 | object detection | mAP⁵⁰⁻⁹⁵ | 31.32 | 31.28 |
| onnx | [yolov9-t_512x288_argmax](onnx/yolov9-t_512x288_argmax/yolov9-t_512x288_argmax.sh) | [288, 512, 3] | 28.562 | object detection | mAP⁵⁰⁻⁹⁵ | 27.75 | 27.63 |
| openvino | [deeplabv3](openvino/deeplabv3/deeplabv3.sh) | [513, 513, 3] | 257.886 | segmentation | meanIoU (VOC2012) | 64.03 | 64.32 |
| openvino | [mobilenet-v1-0.25-128](openvino/mobilenet-v1-0.25-128/mobilenet-v1-0.25-128.sh) | [128, 128, 3] | 1.489 | classification | Top1 | 20.8 | 20.9 |
| openvino | [mobilenet-v1-1.0-224](openvino/mobilenet-v1-1.0-224/mobilenet-v1-1.0-224.sh) | [224, 224, 3] | 11.811 | classification | Top1 | 70.1 | 70.1 |
| openvino | [mobilenet-v1-1.0-224-tf](openvino/mobilenet-v1-1.0-224-tf/mobilenet-v1-1.0-224-tf.sh) | [224, 224, 3] | 11.517 | classification | Top1 | 69.7 | 69.9 |
| openvino | [mobilenet-v2](openvino/mobilenet-v2/mobilenet-v2.sh) | [224, 224, 3] | 15.712 | classification | Top1 | 72.5 | 72.9 |
| openvino | [mobilenet-v2-1.0-224](openvino/mobilenet-v2-1.0-224/mobilenet-v2-1.0-224.sh) | [224, 224, 3] | 12.984 | classification | Top1 | 70.5 | 70.5 |
| openvino | [mobilenet-v2-1.4-224](openvino/mobilenet-v2-1.4-224/mobilenet-v2-1.4-224.sh) | [224, 224, 3] | 20.237 | classification | Top1 | 75.3 | 75.2 |
| openvino | [squeezenet1.0](openvino/squeezenet1.0/squeezenet1.0.sh) | [227, 227, 3] | 18.406 | classification | Top1 | 56.9 | 56.8 |
| openvino | [squeezenet1.1](openvino/squeezenet1.1/squeezenet1.1.sh) | [227, 227, 3] | 11.503 | classification | Top1 | 57.0 | 56.7 |
| pytorch | [lpr_eu_v3](pytorch/lpr_eu_v3/lpr_eu_v3.sh) | [34, 146, 3] | 4.288 | plate recognition |  |  |  |
| pytorch | [spnv2](pytorch/spnv2/spnv2.sh) | [512, 768, 3] | 2949.666 | pose detection |  |  |  |
| pytorch | [torchvision_googlenet](pytorch/torchvision_googlenet/torchvision_googlenet.sh) | [224, 224, 3] | 30.523 | classification | Top1 | 62.4 | 62.3 |
| pytorch | [torchvision_inception_v3](pytorch/torchvision_inception_v3/torchvision_inception_v3.sh) | [299, 299, 3] | 138.194 | classification | Top1 | 78.2 | 75.8 |
| pytorch | [torchvision_resnet18](pytorch/torchvision_resnet18/torchvision_resnet18.sh) | [224, 224, 3] | 25.439 | classification | Top1 | 69.0 | 68.6 |
| pytorch | [torchvision_resnet50](pytorch/torchvision_resnet50/torchvision_resnet50.sh) | [224, 224, 3] | 81.05 | classification | Top1 | 80.7 | 80.6 |
| pytorch | [torchvision_squeezenet1_0](pytorch/torchvision_squeezenet1_0/torchvision_squeezenet1_0.sh) | [227, 227, 3] | 16.812 | classification | Top1 | 58.9 | 58.6 |
| pytorch | [torchvision_ssdlite320_mobilenet_v3_large](pytorch/torchvision_ssdlite320_mobilenet_v3_large/torchvision_ssdlite320_mobilenet_v3_large.sh) | [320, 320, 3] | 38.738 | object detection | mAP⁵⁰⁻⁹⁵ |  |  |
| pytorch | [torchvision_wide_resnet50_2](pytorch/torchvision_wide_resnet50_2/torchvision_wide_resnet50_2.sh) | [224, 224, 3] | 199.298 | classification | Top1 | 80.5 | 81.0 |
| qualcomm | [DeepLabV3-Plus-MobileNet-Quantized](qualcomm/DeepLabV3-Plus-MobileNet-Quantized/DeepLabV3-Plus-MobileNet-Quantized.sh) | [520, 520, 3] | 717.911 | segmentation | meanIoU (VOC2012) | 60.87 |  |
| qualcomm | [DeepLabV3-Plus-MobileNet_512x288](qualcomm/DeepLabV3-Plus-MobileNet_512x288/DeepLabV3-Plus-MobileNet_512x288.sh) | [288, 512, 3] | 308.922 | segmentation | meanIoU (VOC2012) | 53.25 |  |
| qualcomm | [FFNet-122NS-LowRes](qualcomm/FFNet-122NS-LowRes/FFNet-122NS-LowRes.sh) | [512, 1024, 3] | 200.61 | segmentation | meanIoU (cityscapes) | 61.75 | 61.77 |
| qualcomm | [FFNet-122NS-LowRes_512x288](qualcomm/FFNet-122NS-LowRes_512x288/FFNet-122NS-LowRes_512x288.sh) | [288, 512, 3] | 76.168 | segmentation | meanIoU (cityscapes) | 44.82 | 44.45 |
| qualcomm | [FFNet-78S-LowRes](qualcomm/FFNet-78S-LowRes/FFNet-78S-LowRes.sh) | [512, 1024, 3] | 246.535 | segmentation | meanIoU (cityscapes) | 63.17 | 63.39 |
| qualcomm | [FFNet-78S-LowRes_512x288](qualcomm/FFNet-78S-LowRes_512x288/FFNet-78S-LowRes_512x288.sh) | [288, 512, 3] | 91.064 | segmentation | meanIoU (cityscapes) | 47.33 | 47.81 |
| qualcomm | [GoogLeNetQuantized](qualcomm/GoogLeNetQuantized/GoogLeNetQuantized.sh) | [224, 224, 3] | 30.571 | classification | Top1 | 67.8 | 68.5 |
| qualcomm | [Midas-V2-Quantized](qualcomm/Midas-V2-Quantized/Midas-V2-Quantized.sh) | [256, 256, 3] | 121.875 | depth estimation | depthdelta1 (nyuv2) | 85.06 | 81.06 |
| qualcomm | [Midas-V2_256x128](qualcomm/Midas-V2_256x128/Midas-V2_256x128.sh) | [128, 256, 3] | 74.68 | depth estimation | depthdelta1 (nyuv2) | 81.83 | 77.95 |
| qualcomm | [MobileNet-v2-Quantized](qualcomm/MobileNet-v2-Quantized/MobileNet-v2-Quantized.sh) | [224, 224, 3] | 12.914 | classification | Top1 | 67.1 | 66.5 |
| qualcomm | [MobileNet-v3-Large-Quantized](qualcomm/MobileNet-v3-Large-Quantized/MobileNet-v3-Large-Quantized.sh) | [224, 224, 3] | 22.895 | classification | Top1 | 69.6 | 68.8 |
| qualcomm | [QuickSRNetLarge-Quantized](qualcomm/QuickSRNetLarge-Quantized/QuickSRNetLarge-Quantized.sh) | [128, 128, 3] | 62.521 | image enhancement | PSNR (bsd300) | 27.13 | 27.1 |
| qualcomm | [QuickSRNetMedium-Quantized](qualcomm/QuickSRNetMedium-Quantized/QuickSRNetMedium-Quantized.sh) | [128, 128, 3] | 9.353 | image enhancement | PSNR (bsd300) | 26.79 | 26.81 |
| qualcomm | [QuickSRNetSmall-Quantized](qualcomm/QuickSRNetSmall-Quantized/QuickSRNetSmall-Quantized.sh) | [128, 128, 3] | 5.961 | image enhancement | PSNR (bsd300) | 26.95 | 26.95 |
| qualcomm | [ResNet101Quantized](qualcomm/ResNet101Quantized/ResNet101Quantized.sh) | [224, 224, 3] | 138.809 | classification | Top1 | 73.6 | 75.8 |
| qualcomm | [ResNet18Quantized](qualcomm/ResNet18Quantized/ResNet18Quantized.sh) | [224, 224, 3] | 27.272 | classification | Top1 | 66.3 | 68.8 |
| qualcomm | [ResNet50Quantized](qualcomm/ResNet50Quantized/ResNet50Quantized.sh) | [224, 224, 3] | 84.202 | classification | Top1 | 74.8 | 76.3 |
| qualcomm | [SESR-M5-Quantized](qualcomm/SESR-M5-Quantized/SESR-M5-Quantized.sh) | [128, 128, 3] | 59.496 | image enhancement | PSNR (bsd300) | 25.41 | 25.39 |
| qualcomm | [WideResNet50-Quantized](qualcomm/WideResNet50-Quantized/WideResNet50-Quantized.sh) | [224, 224, 3] | 199.307 | classification | Top1 | 76.4 | 77.1 |
| qualcomm | [XLSR-Quantized](qualcomm/XLSR-Quantized/XLSR-Quantized.sh) | [128, 128, 3] | 7.596 | image enhancement | PSNR (bsd300) | 27.23 | 27.23 |
| tensorflow | [efficientnet-lite0-int8](tensorflow/efficientnet-lite0-int8/efficientnet-lite0-int8.sh) | [224, 224, 3] | 16.482 | classification | Top1 | 70.5 | 70.4 |
| tensorflow | [efficientnet-lite1-int8](tensorflow/efficientnet-lite1-int8/efficientnet-lite1-int8.sh) | [240, 240, 3] | 23.662 | classification | Top1 | 72.3 | 71.1 |
| tensorflow | [efficientnet-lite2-int8](tensorflow/efficientnet-lite2-int8/efficientnet-lite2-int8.sh) | [260, 260, 3] | 35.459 | classification | Top1 | 71.1 | 71.1 |
| tensorflow | [efficientnet-lite3-int8](tensorflow/efficientnet-lite3-int8/efficientnet-lite3-int8.sh) | [280, 280, 3] | 48.76 | classification | Top1 | 76.6 | 76.1 |
| tensorflow | [efficientnet-lite4-int8](tensorflow/efficientnet-lite4-int8/efficientnet-lite4-int8.sh) | [300, 300, 3] | 82.229 | classification | Top1 | 77.8 | 77.8 |
| tensorflow | [mobilenet_v1_050_160](tensorflow/mobilenet_v1_050_160/mobilenet_v1_050_160.sh) | [160, 160, 3] | 3.319 | classification | Top1 | 49.4 | 49.9 |
| tensorflow | [mobilenet_v2](tensorflow/mobilenet_v2/mobilenet_v2.sh) | [224, 224, 3] | 12.875 | classification | Top1 | 70.2 | 70.1 |
| tensorflow | [mobilenet_v2_140_224](tensorflow/mobilenet_v2_140_224/mobilenet_v2_140_224.sh) | [224, 224, 3] | 20.146 | classification | Top1 | 75.5 | 75.4 |
| tensorflow | [yolo-v3-tf](tensorflow/yolo-v3-tf/yolo-v3-tf.sh) | [416, 416, 3] | 477.906 | object detection | mAP⁵⁰⁻⁹⁵ | 34.73 | 34.54 |
| tensorflow | [yolo-v3-tiny-tf](tensorflow/yolo-v3-tiny-tf/yolo-v3-tiny-tf.sh) | [416, 416, 3] | 35.531 | object detection | mAP⁵⁰⁻⁹⁵ | 10.86 | 11.0 |
| tensorflow | [yolo-v4-tiny-tf](tensorflow/yolo-v4-tiny-tf/yolo-v4-tiny-tf.sh) | [416, 416, 3] | 46.247 | object detection | mAP⁵⁰⁻⁹⁵ | 12.29 | 12.29 |
| ultralytics | [yolov3-tinyu_FULL](ultralytics/yolov3-tinyu_FULL/yolov3-tinyu_FULL.sh) | [640, 640, 3] | 129.812 | object detection | mAP⁵⁰⁻⁹⁵ | 29.6 | 29.48 |
| ultralytics | [yolov5m](ultralytics/yolov5m/yolov5m.sh) | [640, 640, 3] | 408.463 | object detection | mAP⁵⁰⁻⁹⁵ | 41.34 | 41.43 |
| ultralytics | [yolov5m.relu](ultralytics/yolov5m.relu/yolov5m.relu.sh) | [416, 416, 3] | 166.107 | object detection | mAP⁵⁰⁻⁹⁵ | 38.23 | 38.38 |
| ultralytics | [yolov5n](ultralytics/yolov5n/yolov5n.sh) | [640, 640, 3] | 38.817 | object detection | mAP⁵⁰⁻⁹⁵ | 22.88 | 22.95 |
| ultralytics | [yolov5n.relu](ultralytics/yolov5n.relu/yolov5n.relu.sh) | [416, 416, 3] | 17.016 | object detection | mAP⁵⁰⁻⁹⁵ | 19.2 | 19.01 |
| ultralytics | [yolov5n_512x288](ultralytics/yolov5n_512x288/yolov5n_512x288.sh) | [288, 512, 3] | 15.104 | object detection | mAP⁵⁰⁻⁹⁵ | 20.9 | 20.74 |
| ultralytics | [yolov5nu_FULL](ultralytics/yolov5nu_FULL/yolov5nu_FULL.sh) | [640, 640, 3] | 115.065 | object detection | mAP⁵⁰⁻⁹⁵ | 33.16 | 32.54 |
| ultralytics | [yolov5s](ultralytics/yolov5s/yolov5s.sh) | [640, 640, 3] | 140.364 | object detection | mAP⁵⁰⁻⁹⁵ | 33.24 | 33.03 |
| ultralytics | [yolov5s.relu](ultralytics/yolov5s.relu/yolov5s.relu.sh) | [416, 416, 3] | 63.475 | object detection | mAP⁵⁰⁻⁹⁵ | 31.54 | 31.51 |
| ultralytics | [yolov8l-cls](ultralytics/yolov8l-cls/yolov8l-cls.sh) | [224, 224, 3] | 95.853 | classification |  |  |  |
| ultralytics | [yolov8m](ultralytics/yolov8m/yolov8m.sh) | [640, 640, 3] | 561.775 | object detection | mAP⁵⁰⁻⁹⁵ | 52.09 | 52.1 |
| ultralytics | [yolov8m-cls](ultralytics/yolov8m-cls/yolov8m-cls.sh) | [224, 224, 3] | 36.966 | classification | Top1 | 75.5 | 75.9 |
| ultralytics | [yolov8m_FULL](ultralytics/yolov8m_FULL/yolov8m_FULL.sh) | [640, 640, 3] | 625.049 | object detection | mAP⁵⁰⁻⁹⁵ | 47.2 | 47.35 |
| ultralytics | [yolov8n](ultralytics/yolov8n/yolov8n.sh) | [640, 640, 3] | 54.129 | object detection | mAP⁵⁰⁻⁹⁵ | 37.4 | 37.38 |
| ultralytics | [yolov8n-cls](ultralytics/yolov8n-cls/yolov8n-cls.sh) | [224, 224, 3] | 4.104 | classification | Top1 | 67.2 | 67.3 |
| ultralytics | [yolov8n-obb](ultralytics/yolov8n-obb/yolov8n-obb.sh) | [1024, 1024, 3] | 142.268 | obb detection | mAP⁵⁰⁻⁹⁵ |  |  |
| ultralytics | [yolov8n-pose](ultralytics/yolov8n-pose/yolov8n-pose.sh) | [640, 640, 3] | 57.905 | pose detection | Pose Detection |  |  |
| ultralytics | [yolov8n-pose_512x288](ultralytics/yolov8n-pose_512x288/yolov8n-pose_512x288.sh) | [288, 512, 3] | 22.136 | pose detection | Pose Detection |  |  |
| ultralytics | [yolov8n-pose_512x288_split](ultralytics/yolov8n-pose_512x288_split/yolov8n-pose_512x288_split.sh) | [288, 512, 3] | 21.998 | pose detection | Pose Detection |  |  |
| ultralytics | [yolov8n-seg](ultralytics/yolov8n-seg/yolov8n-seg.sh) | [640, 640, 3] | 70.769 | instance segmentation |  |  |  |
| ultralytics | [yolov8n_512x288](ultralytics/yolov8n_512x288/yolov8n_512x288.sh) | [288, 512, 3] | 21.131 | object detection | mAP⁵⁰⁻⁹⁵ | 28.2 | 28.15 |
| ultralytics | [yolov8n_512x288_argmax](ultralytics/yolov8n_512x288_argmax/yolov8n_512x288_argmax.sh) | [288, 512, 3] | 21.407 | object detection | mAP⁵⁰⁻⁹⁵ | 28.2 | 28.15 |
| ultralytics | [yolov8n_FULL](ultralytics/yolov8n_FULL/yolov8n_FULL.sh) | [640, 640, 3] | 117.005 | object detection | mAP⁵⁰⁻⁹⁵ | 35.4 | 35.36 |
| ultralytics | [yolov8n_argmax](ultralytics/yolov8n_argmax/yolov8n_argmax.sh) | [640, 640, 3] | 55.271 | object detection | mAP⁵⁰⁻⁹⁵ | 37.4 | 37.38 |
| ultralytics | [yolov8s](ultralytics/yolov8s/yolov8s.sh) | [640, 640, 3] | 198.938 | object detection | mAP⁵⁰⁻⁹⁵ | 47.82 | 46.91 |
| ultralytics | [yolov8s-cls](ultralytics/yolov8s-cls/yolov8s-cls.sh) | [224, 224, 3] | 11.52 | classification | Top1 | 72.6 | 72.5 |
| ultralytics | [yolov8s_FULL](ultralytics/yolov8s_FULL/yolov8s_FULL.sh) | [640, 640, 3] | 261.843 | object detection | mAP⁵⁰⁻⁹⁵ | 43.69 | 43.52 |
| ultralytics | [yolov8x-cls](ultralytics/yolov8x-cls/yolov8x-cls.sh) | [224, 224, 3] | 272.231 | classification |  |  |  |
| ultralytics | [yolov9s](ultralytics/yolov9s/yolov9s.sh) | [640, 640, 3] | 167.99 | object detection | mAP⁵⁰⁻⁹⁵ | 47.82 | 47.91 |
| ultralytics | [yolov9s_FULL](ultralytics/yolov9s_FULL/yolov9s_FULL.sh) | [640, 640, 3] | 231.006 | object detection | mAP⁵⁰⁻⁹⁵ | 44.33 | 44.26 |
| ultralytics | [yolov9t](ultralytics/yolov9t/yolov9t.sh) | [640, 640, 3] | 65.068 | object detection | mAP⁵⁰⁻⁹⁵ | 37.92 | 38.18 |
| ultralytics | [yolov9t_FULL](ultralytics/yolov9t_FULL/yolov9t_FULL.sh) | [640, 640, 3] | 127.71 | object detection | mAP⁵⁰⁻⁹⁵ | 35.51 | 35.63 |
| vectorblox | [yolov8n-relu](vectorblox/yolov8n-relu/yolov8n-relu.sh) | [640, 640, 3] | 116.913 | object detection | mAP⁵⁰⁻⁹⁵ | 33.28 | 33.15 |
| vectorblox | [yolov9s-spn](vectorblox/yolov9s-spn/yolov9s-spn.sh) | [512, 768, 3] | 207.024 | pose detection |  |  |  |

</details>

<details>

<summary>Compression Tutorial Metrics</summary>

## Compression

| Source | Tutorial | Input<br>(H,W,C) | Runtime<br>(ms) | Task | Metric | TFLITE | VNNX |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| compressed | [yolov8n_comp66](compressed/yolov8n_comp66/yolov8n_comp66.sh) | [640, 640, 3] | 35.85 | object detection | mAP⁵⁰⁻⁹⁵ | 37.18 | 37.3 |
| compressed | [yolov8s_comp68](compressed/yolov8s_comp68/yolov8s_comp68.sh) | [640, 640, 3] | 90.812 | object detection | mAP⁵⁰⁻⁹⁵ | 46.11 | 46.15 |

</details>


<details>

<summary>Unstructured Compression Tutorial Metrics</summary>

## Unstructured Compression

| Source | Tutorial | Input<br>(H,W,C) | Runtime<br>(ms) | Task | Metric | TFLITE |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| unstructure_compressed | [resnet18_86s_07p](unstructure_compressed/resnet18_86s_07p/resnet18_86s_07p.sh) | [224, 224, 3] | 14.63 | classification | Top1 | 65.6 |
| unstructure_compressed | [yolov5n_70s_512x512](unstructure_compressed/yolov5n_70s_512x512/yolov5n_70s_512x512.sh) | [512, 512, 3] | 19.779 | object detection | mAP⁵⁰⁻⁹⁵ | 20.2 |
| unstructure_compressed | [yolov8n_50s_25p_512x288](unstructure_compressed/yolov8n_50s_25p_512x288/yolov8n_50s_25p_512x288.sh) | [288, 512, 3] | 15.428 | object detection | mAP⁵⁰⁻⁹⁵ | 34.6 |
| unstructure_compressed | [yolov8n_50s_25p_512x512](unstructure_compressed/yolov8n_50s_25p_512x512/yolov8n_50s_25p_512x512.sh) | [512, 512, 3] | 25.265 | object detection | mAP⁵⁰⁻⁹⁵ |  |
| unstructure_compressed | [yolov8n_pose_50s_25p_512x288_split](unstructure_compressed/yolov8n_pose_50s_25p_512x288_split/yolov8n_pose_50s_25p_512x288_split.sh) | [288, 512, 3] | 15.507 | pose detection |  |  |
| unstructure_compressed | [yolov9s_70s_15p_512x288](unstructure_compressed/yolov9s_70s_15p_512x288/yolov9s_70s_15p_512x288.sh) | [288, 512, 3] | 32.95 | object detection |  | 43.2 |
| unstructure_compressed | [yolov9s_70s_15p_512x512](unstructure_compressed/yolov9s_70s_15p_512x512/yolov9s_70s_15p_512x512.sh) | [512, 512, 3] | 55.576 | object detection |  |  |

</details>
