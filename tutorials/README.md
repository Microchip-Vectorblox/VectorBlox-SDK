
# Tutorials

Below is a list of included tutorials. Frames per Second (FPS) assumes the cores are running at 125 MHz.
Accuracy is reported for 1000 samples.


| Source | Tutorial | Input (H,W,C) | V1000 FPS | Task  | Metric | TFLITE | VNNX |
| ----------------- | ------------- |-----------------|-------|---------------|--------------|------------|-----------|
|darknet|[yolov2-tiny-voc](darknet/yolov2-tiny-voc/yolov2-tiny-voc.sh)|[416, 416, 3]|20.8|object detection|mAP⁵⁰⁻⁹⁵|||
|darknet|[yolov2-tiny](darknet/yolov2-tiny/yolov2-tiny.sh)|[416, 416, 3]|27.2|object detection|mAP⁵⁰⁻⁹⁵|||
|darknet|[yolov3-tiny](darknet/yolov3-tiny/yolov3-tiny.sh)|[416, 416, 3]|25.9|object detection|mAP⁵⁰⁻⁹⁵|10.9|10.8|
|kaggle|[efficientnet-lite0](kaggle/efficientnet-lite0/efficientnet-lite0.sh)|[224, 224, 3]|60.4|classification|Top1|70.5|70.2|
|kaggle|[efficientnet-lite4](kaggle/efficientnet-lite4/efficientnet-lite4.sh)|[224, 224, 3]|13.5|classification|Top1|77.8||
|mediapipe|[efficientnet_lite0](mediapipe/efficientnet_lite0/efficientnet_lite0.sh)|[224, 224, 3]|61.7|classification|Top1|70.5|70.2|
|mediapipe|[efficientnet_lite2](mediapipe/efficientnet_lite2/efficientnet_lite2.sh)|[260, 260, 3]|30.9|classification|Top1|70.7|71.3|
|onnx|[onnx_resnet18-v1](onnx/onnx_resnet18-v1/onnx_resnet18-v1.sh)|[224, 224, 3]|32.7|classification|Top1|69.3|68.8|
|onnx|[onnx_resnet34-v1](onnx/onnx_resnet34-v1/onnx_resnet34-v1.sh)|[224, 224, 3]|17.8|classification|Top1|72.6|72.2|
|onnx|[onnx_squeezenet1.1](onnx/onnx_squeezenet1.1/onnx_squeezenet1.1.sh)|[224, 224, 3]|132.6|classification|Top1|54.0|54.8|
|onnx|[scrfd_500m_bnkps](onnx/scrfd_500m_bnkps/scrfd_500m_bnkps.sh)|[288, 512, 3]|85.8|face detection||||
|onnx|[yolov7](onnx/yolov7/yolov7.sh)|[640, 640, 3]|1.0|object detection|mAP⁵⁰⁻⁹⁵|||
|onnx|[yolov9-s](onnx/yolov9-s/yolov9-s.sh)|[640, 640, 3]|3.9|object detection|mAP⁵⁰⁻⁹⁵|||
|onnx|[yolov9-m](onnx/yolov9-m/yolov9-m.sh)|[640, 640, 3]|1.2|object detection|mAP⁵⁰⁻⁹⁵|||
|openvino|[mobilenet-v1-1.0-224](openvino/mobilenet-v1-1.0-224/mobilenet-v1-1.0-224.sh)|[224, 224, 3]|80.2|classification|Top1|70.1|70.0|
|openvino|[mobilenet-v2](openvino/mobilenet-v2/mobilenet-v2.sh)|[224, 224, 3]|58.8|classification|Top1|72.5|72.5|
|openvino|[mobilenet-v1-1.0-224-tf](openvino/mobilenet-v1-1.0-224-tf/mobilenet-v1-1.0-224-tf.sh)|[224, 224, 3]|77.4|classification|Top1|69.7|69.6|
|openvino|[mobilenet-v2-1.0-224](openvino/mobilenet-v2-1.0-224/mobilenet-v2-1.0-224.sh)|[224, 224, 3]|69.7|classification|Top1|70.5|70.1|
|openvino|[mobilenet-v2-1.4-224](openvino/mobilenet-v2-1.4-224/mobilenet-v2-1.4-224.sh)|[224, 224, 3]|4.5|classification|Top1|75.3|75.6|
|openvino|[mobilefacenet-arcface](openvino/mobilefacenet-arcface/mobilefacenet-arcface.sh)|[224, 224, 3]|65.0|face comparison||||
|openvino|[squeezenet1.0](openvino/squeezenet1.0/squeezenet1.0.sh)|[227, 227, 3]|65.2|classification|Top1|56.9|56.8|
|openvino|[squeezenet1.1](openvino/squeezenet1.1/squeezenet1.1.sh)|[227, 227, 3]|115.5|classification|Top1|57.0|56.9|
|openvino|[mobilenet-v1-0.25-128](openvino/mobilenet-v1-0.25-128/mobilenet-v1-0.25-128.sh)|[128, 128, 3]|414.2|classification|Top1|20.8|21.2|
|openvino|[deeplabv3](openvino/deeplabv3/deeplabv3.sh)|[513, 513, 3]|3.8|segmentation||||
|PINTO|[016_EfficientNet-lite](PINTO/016_EfficientNet-lite/016_EfficientNet-lite.sh)|[224, 224, 3]|60.1|classification|Top1|||
|PINTO|[046_yolov4-tiny](PINTO/046_yolov4-tiny/046_yolov4-tiny.sh)|[416, 416, 3]|18.2|object detection|mAP⁵⁰⁻⁹⁵|||
|PINTO|[081_MiDaS_v2](PINTO/081_MiDaS_v2/081_MiDaS_v2.sh)|[256, 256, 3]|7.7|depth estimation||||
|PINTO|[132_YOLOX](PINTO/132_YOLOX/132_YOLOX.sh)|[416, 416, 3]|43.4|classification|mAP⁵⁰⁻⁹⁵|||
|PINTO|[307_YOLOv7](PINTO/307_YOLOv7/307_YOLOv7.sh)|[640, 640, 3]|14.6|classification|mAP|||
|pytorch|[torchvision_resnet18](pytorch/torchvision_resnet18/torchvision_resnet18.sh)|[224, 224, 3]|32.2|classification|Top1|69.0|68.7|
|pytorch|[torchvision_resnet50](pytorch/torchvision_resnet50/torchvision_resnet50.sh)|[224, 224, 3]|11.4|classification|Top1|80.8|80.7|
|pytorch|[torchvision_wide_resnet50_2](pytorch/torchvision_wide_resnet50_2/torchvision_wide_resnet50_2.sh)|[224, 224, 3]|5.0|classification|Top1|81.3|81.0|
|pytorch|[torchvision_inception_v3](pytorch/torchvision_inception_v3/torchvision_inception_v3.sh)|[299, 299, 3]|7.0|classification|Top1|78.1|76.1|
|pytorch|[torchvision_ssdlite320_mobilenet_v3_large](pytorch/torchvision_ssdlite320_mobilenet_v3_large/torchvision_ssdlite320_mobilenet_v3_large.sh)|[320, 320, 3]|23.3|object detection|mAP⁵⁰⁻⁹⁵|||
|pytorch|[torchvision_googlenet](pytorch/torchvision_googlenet/torchvision_googlenet.sh)|[224, 224, 3]|31.1|classification|Top1|62.5|61.9|
|pytorch|[lpr_eu_v3](pytorch/lpr_eu_v3/lpr_eu_v3.sh)|[34, 146, 3]|198.8|plate recognition||||
|pytorch|[torchvision_squeezenet1_0](pytorch/torchvision_squeezenet1_0/torchvision_squeezenet1_0.sh)|[227, 227, 3]|66.8|classification|Top1|59.5|59.1|
|qualcomm|[DeepLabV3-Plus-MobileNet-Quantized](qualcomm/DeepLabV3-Plus-MobileNet-Quantized/DeepLabV3-Plus-MobileNet-Quantized.sh)|[520, 520, 3]|1.4|segmentation||||
|qualcomm|[DeepLabV3-Plus-MobileNet_512x288](qualcomm/DeepLabV3-Plus-MobileNet_512x288/DeepLabV3-Plus-MobileNet_512x288.sh)|[288, 512, 3]|2.6|segmentation||||
|qualcomm|[Midas-V2-Quantized](qualcomm/Midas-V2-Quantized/Midas-V2-Quantized.sh)|[256, 256, 3]|7.7|depth estimation||||
|qualcomm|[Midas-V2_256x128](qualcomm/Midas-V2_256x128/Midas-V2_256x128.sh)|[128, 256, 3]|13.4|depth estimation||||
|qualcomm|[FFNet-122NS-LowRes](qualcomm/FFNet-122NS-LowRes/FFNet-122NS-LowRes.sh)|[512, 1024, 3]|4.4|segmentation||||
|qualcomm|[FFNet-122NS-LowRes_512x288](qualcomm/FFNet-122NS-LowRes_512x288/FFNet-122NS-LowRes_512x288.sh)|[288, 512, 3]|9.6|segmentation||||
|qualcomm|[FFNet-78S-LowRes](qualcomm/FFNet-78S-LowRes/FFNet-78S-LowRes.sh)|[512, 1024, 3]|3.4|segmentation||||
|qualcomm|[FFNet-78S-LowRes_512x288](qualcomm/FFNet-78S-LowRes_512x288/FFNet-78S-LowRes_512x288.sh)|[288, 512, 3]|8.5|segmentation||||
|qualcomm|[MobileNet-v2-Quantized](qualcomm/MobileNet-v2-Quantized/MobileNet-v2-Quantized.sh)|[224, 224, 3]|68.9|classification|Top1|69.0|70.6|
|qualcomm|[GoogLeNetQuantized](qualcomm/GoogLeNetQuantized/GoogLeNetQuantized.sh)|[224, 224, 3]|30.0|classification|Top1|67.9|68.7|
|qualcomm|[ResNet18Quantized](qualcomm/ResNet18Quantized/ResNet18Quantized.sh)|[224, 224, 3]|31.7|classification|Top1|66.3|69.5|
|qualcomm|[ResNet50Quantized](qualcomm/ResNet50Quantized/ResNet50Quantized.sh)|[224, 224, 3]|11.3|classification|Top1|74.9|76.1|
|qualcomm|[ResNet101Quantized](qualcomm/ResNet101Quantized/ResNet101Quantized.sh)|[224, 224, 3]|7.0|classification|Top1|73.8|75.0|
|qualcomm|[WideResNet50-Quantized](qualcomm/WideResNet50-Quantized/WideResNet50-Quantized.sh)|[224, 224, 3]|5.0|classification|Top1|76.4|76.9|
|qualcomm|[MobileNet-v3-Large-Quantized](qualcomm/MobileNet-v3-Large-Quantized/MobileNet-v3-Large-Quantized.sh)|[224, 224, 3]|38.4|classification|Top1|70.3|70.7|
|qualcomm|[Yolo-X-Quantized](qualcomm/Yolo-X-Quantized/Yolo-X-Quantized.sh)|[640, 640, 3]|5.0|object detection|mAP⁵⁰⁻⁹⁵|||
|qualcomm|[XLSR-Quantized](qualcomm/XLSR-Quantized/XLSR-Quantized.sh)|[128, 128, 3]|86.9|image enhancement||||
|qualcomm|[QuickSRNetSmall-Quantized](qualcomm/QuickSRNetSmall-Quantized/QuickSRNetSmall-Quantized.sh)|[128, 128, 3]|0.1|image enhancement||||
|qualcomm|[QuickSRNetMedium-Quantized](qualcomm/QuickSRNetMedium-Quantized/QuickSRNetMedium-Quantized.sh)|[128, 128, 3]|83.5|image enhancement||||
|qualcomm|[QuickSRNetLarge-Quantized](qualcomm/QuickSRNetLarge-Quantized/QuickSRNetLarge-Quantized.sh)|[128, 128, 3]|14.0|image enhancement||||
|qualcomm|[SESR-M5-Quantized](qualcomm/SESR-M5-Quantized/SESR-M5-Quantized.sh)|[128, 128, 3]|13.7|image enhancement||||
|tensorflow|[yolo-v4-tiny-tf](tensorflow/yolo-v4-tiny-tf/yolo-v4-tiny-tf.sh)|[416, 416, 3]|18.9|object detection|mAP⁵⁰⁻⁹⁵|12.3|12.5|
|tensorflow|[mobilenet_v2](tensorflow/mobilenet_v2/mobilenet_v2.sh)|[224, 224, 3]|69.2|classification|Top1|70.2|70.1|
|tensorflow|[yolo-v3-tiny-tf](tensorflow/yolo-v3-tiny-tf/yolo-v3-tiny-tf.sh)|[416, 416, 3]|25.9|object detection|mAP⁵⁰⁻⁹⁵|10.9|11.0|
|tensorflow|[yolo-v3-tf](tensorflow/yolo-v3-tf/yolo-v3-tf.sh)|[416, 416, 3]|2.0|object detection|mAP⁵⁰⁻⁹⁵|34.7|34.5|
|tensorflow|[efficientnet-lite0-int8](tensorflow/efficientnet-lite0-int8/efficientnet-lite0-int8.sh)|[224, 224, 3]|61.7|classification|Top1|70.5|70.2|
|tensorflow|[efficientnet-lite1-int8](tensorflow/efficientnet-lite1-int8/efficientnet-lite1-int8.sh)|[240, 240, 3]|41.9|classification|Top1|72.3|71.9|
|tensorflow|[efficientnet-lite2-int8](tensorflow/efficientnet-lite2-int8/efficientnet-lite2-int8.sh)|[260, 260, 3]|31.2|classification|Top1|71.1|71.3|
|tensorflow|[efficientnet-lite3-int8](tensorflow/efficientnet-lite3-int8/efficientnet-lite3-int8.sh)|[280, 280, 3]|21.9|classification|Top1|76.6|76.0|
|tensorflow|[efficientnet-lite4-int8](tensorflow/efficientnet-lite4-int8/efficientnet-lite4-int8.sh)|[300, 300, 3]|13.6|classification|Top1|77.8|77.8|
|tensorflow|[mobilenet_v1_050_160](tensorflow/mobilenet_v1_050_160/mobilenet_v1_050_160.sh)|[160, 160, 3]|228.1|classification|Top1|49.4|49.2|
|tensorflow|[mobilenet_v2_140_224](tensorflow/mobilenet_v2_140_224/mobilenet_v2_140_224.sh)|[224, 224, 3]|45.1|classification|Top1|75.5|75.7|
|tensorflow|[posenet](tensorflow/posenet/posenet.sh)|[273, 481, 3]|25.6|pose detection||||
|ultralytics|[yolov5n.relu](ultralytics/yolov5n.relu/yolov5n.relu.sh)|[416, 416, 3]|47.5|object detection|mAP⁵⁰⁻⁹⁵|19.2|19.1|
|ultralytics|[yolov5s.relu](ultralytics/yolov5s.relu/yolov5s.relu.sh)|[416, 416, 3]|16.7|object detection|mAP⁵⁰⁻⁹⁵|31.5|31.6|
|ultralytics|[yolov5m.relu](ultralytics/yolov5m.relu/yolov5m.relu.sh)|[416, 416, 3]|6.4|object detection|mAP⁵⁰⁻⁹⁵|38.2|38.3|
|ultralytics|[yolov5n](ultralytics/yolov5n/yolov5n.sh)|[640, 640, 3]|19.3|object detection|mAP⁵⁰⁻⁹⁵|22.9|22.9|
|ultralytics|[yolov5n_512x288](ultralytics/yolov5n_512x288/yolov5n_512x288.sh)|[288, 512, 3]|52.1|object detection|mAP⁵⁰⁻⁹⁵|20.9|20.9|
|ultralytics|[yolov5s](ultralytics/yolov5s/yolov5s.sh)|[640, 640, 3]|6.7|object detection|mAP⁵⁰⁻⁹⁵|33.2|33.0|
|ultralytics|[yolov5m](ultralytics/yolov5m/yolov5m.sh)|[640, 640, 3]|2.5|object detection|mAP⁵⁰⁻⁹⁵|41.1|41.6|
|ultralytics|[yolov8n_FULL](ultralytics/yolov8n_FULL/yolov8n_FULL.sh)|[640, 640, 3]|7.5|object detection|mAP⁵⁰⁻⁹⁵|35.4|35.5|
|ultralytics|[yolov8s_FULL](ultralytics/yolov8s_FULL/yolov8s_FULL.sh)|[640, 640, 3]|3.5|object detection|mAP⁵⁰⁻⁹⁵|43.7|43.6|
|ultralytics|[yolov8m_FULL](ultralytics/yolov8m_FULL/yolov8m_FULL.sh)|[640, 640, 3]|1.5|object detection|mAP⁵⁰⁻⁹⁵|47.2|47.2|
|ultralytics|[yolov9t_FULL](ultralytics/yolov9t_FULL/yolov9t_FULL.sh)|[640, 640, 3]|6.6|object detection|mAP⁵⁰⁻⁹⁵|35.5|35.5|
|ultralytics|[yolov9s_FULL](ultralytics/yolov9s_FULL/yolov9s_FULL.sh)|[640, 640, 3]|3.5|object detection|mAP⁵⁰⁻⁹⁵|44.3|44.3|
|ultralytics|[yolov8n](ultralytics/yolov8n/yolov8n.sh)|[640, 640, 3]|13.3|object detection|mAP⁵⁰⁻⁹⁵|37.4|37.4|
|ultralytics|[yolov8n_argmax](ultralytics/yolov8n_argmax/yolov8n_argmax.sh)|[640, 640, 3]|13.0|object detection|mAP⁵⁰⁻⁹⁵|37.4|37.4|
|ultralytics|[yolov8n_512x288](ultralytics/yolov8n_512x288/yolov8n_512x288.sh)|[288, 512, 3]|35.6|object detection|mAP⁵⁰⁻⁹⁵|1.6|1.5|
|ultralytics|[yolov8n_512x288_argmax](ultralytics/yolov8n_512x288_argmax/yolov8n_512x288_argmax.sh)|[288, 512, 3]|35.6|object detection|mAP⁵⁰⁻⁹⁵|1.6|1.5|
|ultralytics|[yolov8n-pose_512x288](ultralytics/yolov8n-pose_512x288/yolov8n-pose_512x288.sh)|[288, 512, 3]|33.5|pose detection|Pose Detection|||
|ultralytics|[yolov8n-pose_512x288_split](ultralytics/yolov8n-pose_512x288_split/yolov8n-pose_512x288_split.sh)|[288, 512, 3]|33.4|pose detection|Pose Detection|||
|ultralytics|[yolov8s](ultralytics/yolov8s/yolov8s.sh)|[640, 640, 3]|4.5|object detection|mAP⁵⁰⁻⁹⁵|47.8|46.9|
|ultralytics|[yolov8m](ultralytics/yolov8m/yolov8m.sh)|[640, 640, 3]|1.6|object detection|mAP⁵⁰⁻⁹⁵|52.1|52.1|
|ultralytics|[yolov9s](ultralytics/yolov9s/yolov9s.sh)|[640, 640, 3]|4.4|object detection|mAP⁵⁰⁻⁹⁵|47.8|47.9|
|ultralytics|[yolov9t](ultralytics/yolov9t/yolov9t.sh)|[640, 640, 3]|10.7|object detection|mAP⁵⁰⁻⁹⁵|37.9|38.0|
|ultralytics|[yolov8n-cls](ultralytics/yolov8n-cls/yolov8n-cls.sh)|[224, 224, 3]|152.0|classification|Top1|67.2|67.9|
|ultralytics|[yolov8s-cls](ultralytics/yolov8s-cls/yolov8s-cls.sh)|[224, 224, 3]|66.2|classification|Top1|72.6|72.1|
|ultralytics|[yolov8m-cls](ultralytics/yolov8m-cls/yolov8m-cls.sh)|[224, 224, 3]|23.6|classification|Top1|75.5|75.9|
|ultralytics|[yolov8n-pose](ultralytics/yolov8n-pose/yolov8n-pose.sh)|[640, 640, 3]|12.6|pose detection|Pose Detection|||
|ultralytics|[yolov8n-seg](ultralytics/yolov8n-seg/yolov8n-seg.sh)|[640, 640, 3]|10.3|instance segmentation||||
|ultralytics|[yolov8n-obb](ultralytics/yolov8n-obb/yolov8n-obb.sh)|[1024, 1024, 3]|5.1|obb detection||||
|ultralytics|[yolov3-tinyu_FULL](ultralytics/yolov3-tinyu_FULL/yolov3-tinyu_FULL.sh)|[640, 640, 3]|6.5|object detection|mAP⁵⁰⁻⁹⁵|29.6|29.5|
|ultralytics|[yolov5nu_FULL](ultralytics/yolov5nu_FULL/yolov5nu_FULL.sh)|[640, 640, 3]|7.9|object detection|mAP⁵⁰⁻⁹⁵|33.2|32.7|
|vectorblox|[yolov8n-relu](vectorblox/yolov8n-relu/yolov8n-relu.sh)|[640, 640, 3]|7.8|object detection|mAP⁵⁰⁻⁹⁵|33.3|33.2|
