
# Tutorials



The SDK contains a set of end-to-end tutorials in the tutorials directory that describe how to generate a binary file for a specific model.

Tutorials are organized by their source.

For every given tutorial, a shell script is provided for users to run that performs the following steps:
1.  Download the corresponding model using the respective source.
2.  If not already in TF Lite format, transform the downloaded model into a TF Lite format using openvino2tensorflow, onnx2tf or tflite_quantize.
3.  Once generated, run tflite_preprocess to add a preprocessing layer prior to generating the binary file.
4.  Create the binary file by calling VectorBlox's graph generation tool, vnnx_compile.
5.  Simulate the binary file using a Python inference script from example/python directory with ourprovided test image.

These scripts are examples of the complete pipeline for generating a binary file for a model. Users may use or modify these scripts to generate a binary file for their own custom model to fit their use cases.

To run a tutorial shell script, navigate to the appropriate model tutorial directory and run the following command:

```
(vbx_env) ~/SDK/VectorBlox-SDK/tutorials/SOURCE_NAME/MODEL_NAME$ bash MODEL_NAME.sh
```

# Tutorial Metrics

Below is a list of included tutorials. Runtime in milliseconds (ms) measured on [SoC Video Kit](https://github.com/Microchip-Vectorblox/VectorBlox-SoC-Video-Kit-Demo).
Accuracy measured over 1000 samples.

## No Compression



| Source | Tutorial | Input (H,W,C) | Runtime (ms) | Task  | Metric | TFLITE | VNNX |
| ----------------- | ------------- |-----------------|-------|---------------|--------------|------------|-----------|
|darknet|[yolov2-tiny-voc](darknet/yolov2-tiny-voc/yolov2-tiny-voc.sh)|[416, 416, 3]|44.0|object detection|mAP⁵⁰⁻⁹⁵|||
|darknet|[yolov2-tiny](darknet/yolov2-tiny/yolov2-tiny.sh)|[416, 416, 3]|33.3|object detection|mAP⁵⁰⁻⁹⁵|11.1|11.1|
|darknet|[yolov3-tiny](darknet/yolov3-tiny/yolov3-tiny.sh)|[416, 416, 3]|35.1|object detection|mAP⁵⁰⁻⁹⁵|10.9|10.8|
|kaggle|[efficientnet-lite0](kaggle/efficientnet-lite0/efficientnet-lite0.sh)|[224, 224, 3]|19.1|classification|Top1|70.5|70.4|
|kaggle|[efficientnet-lite4](kaggle/efficientnet-lite4/efficientnet-lite4.sh)|[224, 224, 3]|89.3|classification|Top1|77.8||
|mediapipe|[efficientnet_lite0](mediapipe/efficientnet_lite0/efficientnet_lite0.sh)|[224, 224, 3]|19.1|classification|Top1|70.5|70.4|
|mediapipe|[efficientnet_lite2](mediapipe/efficientnet_lite2/efficientnet_lite2.sh)|[260, 260, 3]|38.7|classification|Top1|70.7|71.3|
|onnx|[onnx_resnet18-v1](onnx/onnx_resnet18-v1/onnx_resnet18-v1.sh)|[224, 224, 3]|26.9|classification|Top1|69.3|69.2|
|onnx|[onnx_resnet34-v1](onnx/onnx_resnet34-v1/onnx_resnet34-v1.sh)|[224, 224, 3]|49.3|classification|Top1|72.6|72.4|
|onnx|[onnx_resnet101-v1](onnx/onnx_resnet101-v1/onnx_resnet101-v1.sh)|[224, 224, 3]|163.0|classification|Top1|72.0|70.0|
|onnx|[onnx_squeezenet1.1](onnx/onnx_squeezenet1.1/onnx_squeezenet1.1.sh)|[224, 224, 3]|6.7|classification|Top1|54.0|54.5|
|onnx|[scrfd_500m_bnkps](onnx/scrfd_500m_bnkps/scrfd_500m_bnkps.sh)|[288, 512, 3]|11.5|face detection||||
|onnx|[yolov7](onnx/yolov7/yolov7.sh)|[640, 640, 3]|1073.4|object detection|mAP⁵⁰⁻⁹⁵|||
|onnx|[yolov9-s](onnx/yolov9-s/yolov9-s.sh)|[640, 640, 3]|211.3|object detection|mAP⁵⁰⁻⁹⁵|36.0|46.8|
|onnx|[yolov9-m](onnx/yolov9-m/yolov9-m.sh)|[640, 640, 3]|729.5|object detection|mAP⁵⁰⁻⁹⁵|59.2|59.4|
|openvino|[mobilenet-v1-1.0-224](openvino/mobilenet-v1-1.0-224/mobilenet-v1-1.0-224.sh)|[224, 224, 3]|13.4|classification|Top1|70.1|70.1|
|openvino|[mobilenet-v2](openvino/mobilenet-v2/mobilenet-v2.sh)|[224, 224, 3]|18.1|classification|Top1|72.5|72.5|
|openvino|[mobilenet-v1-1.0-224-tf](openvino/mobilenet-v1-1.0-224-tf/mobilenet-v1-1.0-224-tf.sh)|[224, 224, 3]|13.2|classification|Top1|69.7|69.6|
|openvino|[mobilenet-v2-1.0-224](openvino/mobilenet-v2-1.0-224/mobilenet-v2-1.0-224.sh)|[224, 224, 3]|15.1|classification|Top1|70.5|70.5|
|openvino|[mobilenet-v2-1.4-224](openvino/mobilenet-v2-1.4-224/mobilenet-v2-1.4-224.sh)|[224, 224, 3]|23.0|classification|Top1|75.3|75.2|
|openvino|[mobilefacenet-arcface](openvino/mobilefacenet-arcface/mobilefacenet-arcface.sh)|[224, 224, 3]|18.8|face comparison||||
|openvino|[squeezenet1.0](openvino/squeezenet1.0/squeezenet1.0.sh)|[227, 227, 3]|15.6|classification|Top1|56.9|56.8|
|openvino|[squeezenet1.1](openvino/squeezenet1.1/squeezenet1.1.sh)|[227, 227, 3]|7.3|classification|Top1|57.0|56.7|
|openvino|[mobilenet-v1-0.25-128](openvino/mobilenet-v1-0.25-128/mobilenet-v1-0.25-128.sh)|[128, 128, 3]|1.6|classification|Top1|20.8|20.9|
|openvino|[deeplabv3](openvino/deeplabv3/deeplabv3.sh)|[513, 513, 3]|270.8|segmentation||||
|PINTO|[016_EfficientNet-lite](PINTO/016_EfficientNet-lite/016_EfficientNet-lite.sh)|[224, 224, 3]|19.1|classification|Top1|70.5|70.4|
|PINTO|[046_yolov4-tiny](PINTO/046_yolov4-tiny/046_yolov4-tiny.sh)|[416, 416, 3]|47.4|object detection|mAP⁵⁰⁻⁹⁵|||
|PINTO|[081_MiDaS_v2](PINTO/081_MiDaS_v2/081_MiDaS_v2.sh)|[256, 256, 3]|127.4|depth estimation||||
|PINTO|[132_YOLOX](PINTO/132_YOLOX/132_YOLOX.sh)|[416, 416, 3]|19.9|classification|mAP⁵⁰⁻⁹⁵|||
|PINTO|[307_YOLOv7](PINTO/307_YOLOv7/307_YOLOv7.sh)|[640, 640, 3]|61.9|classification|mAP|||
|pytorch|[torchvision_resnet18](pytorch/torchvision_resnet18/torchvision_resnet18.sh)|[224, 224, 3]|27.0|classification|Top1|69.0|68.9|
|pytorch|[torchvision_resnet50](pytorch/torchvision_resnet50/torchvision_resnet50.sh)|[224, 224, 3]|97.0|classification|Top1|80.8|80.7|
|pytorch|[torchvision_wide_resnet50_2](pytorch/torchvision_wide_resnet50_2/torchvision_wide_resnet50_2.sh)|[224, 224, 3]|226.5|classification|Top1|81.3|80.8|
|pytorch|[torchvision_inception_v3](pytorch/torchvision_inception_v3/torchvision_inception_v3.sh)|[299, 299, 3]|145.3|classification|Top1|78.1|76.1|
|pytorch|[torchvision_ssdlite320_mobilenet_v3_large](pytorch/torchvision_ssdlite320_mobilenet_v3_large/torchvision_ssdlite320_mobilenet_v3_large.sh)|[320, 320, 3]|41.7|object detection|mAP⁵⁰⁻⁹⁵|||
|pytorch|[torchvision_googlenet](pytorch/torchvision_googlenet/torchvision_googlenet.sh)|[224, 224, 3]|31.0|classification|Top1|62.5|62.3|
|pytorch|[lpr_eu_v3](pytorch/lpr_eu_v3/lpr_eu_v3.sh)|[34, 146, 3]|4.5|plate recognition||||
|pytorch|[torchvision_squeezenet1_0](pytorch/torchvision_squeezenet1_0/torchvision_squeezenet1_0.sh)|[227, 227, 3]|15.4|classification|Top1|59.5|59.3|
|qualcomm|[DeepLabV3-Plus-MobileNet-Quantized](qualcomm/DeepLabV3-Plus-MobileNet-Quantized/DeepLabV3-Plus-MobileNet-Quantized.sh)|[520, 520, 3]|651.1|segmentation||||
|qualcomm|[DeepLabV3-Plus-MobileNet_512x288](qualcomm/DeepLabV3-Plus-MobileNet_512x288/DeepLabV3-Plus-MobileNet_512x288.sh)|[288, 512, 3]|311.1|segmentation||||
|qualcomm|[Midas-V2-Quantized](qualcomm/Midas-V2-Quantized/Midas-V2-Quantized.sh)|[256, 256, 3]|127.1|depth estimation||||
|qualcomm|[Midas-V2_256x128](qualcomm/Midas-V2_256x128/Midas-V2_256x128.sh)|[128, 256, 3]|74.9|depth estimation||||
|qualcomm|[FFNet-122NS-LowRes](qualcomm/FFNet-122NS-LowRes/FFNet-122NS-LowRes.sh)|[512, 1024, 3]|201.2|segmentation||||
|qualcomm|[FFNet-122NS-LowRes_512x288](qualcomm/FFNet-122NS-LowRes_512x288/FFNet-122NS-LowRes_512x288.sh)|[288, 512, 3]|81.4|segmentation||||
|qualcomm|[FFNet-78S-LowRes](qualcomm/FFNet-78S-LowRes/FFNet-78S-LowRes.sh)|[512, 1024, 3]|249.2|segmentation||||
|qualcomm|[FFNet-78S-LowRes_512x288](qualcomm/FFNet-78S-LowRes_512x288/FFNet-78S-LowRes_512x288.sh)|[288, 512, 3]|95.0|segmentation||||
|qualcomm|[MobileNet-v2-Quantized](qualcomm/MobileNet-v2-Quantized/MobileNet-v2-Quantized.sh)|[224, 224, 3]|15.0|classification|Top1|67.1|66.5|
|qualcomm|[GoogLeNetQuantized](qualcomm/GoogLeNetQuantized/GoogLeNetQuantized.sh)|[224, 224, 3]|30.9|classification|Top1|67.8|68.5|
|qualcomm|[ResNet18Quantized](qualcomm/ResNet18Quantized/ResNet18Quantized.sh)|[224, 224, 3]|28.5|classification|Top1|66.3|68.8|
|qualcomm|[ResNet50Quantized](qualcomm/ResNet50Quantized/ResNet50Quantized.sh)|[224, 224, 3]|98.3|classification|Top1|74.8|76.3|
|qualcomm|[ResNet101Quantized](qualcomm/ResNet101Quantized/ResNet101Quantized.sh)|[224, 224, 3]|166.8|classification|Top1|73.6|75.8|
|qualcomm|[WideResNet50-Quantized](qualcomm/WideResNet50-Quantized/WideResNet50-Quantized.sh)|[224, 224, 3]|226.5|classification|Top1|76.4|77.1|
|qualcomm|[MobileNet-v3-Large-Quantized](qualcomm/MobileNet-v3-Large-Quantized/MobileNet-v3-Large-Quantized.sh)|[224, 224, 3]|22.4|classification|Top1|69.6|68.8|
|qualcomm|[XLSR-Quantized](qualcomm/XLSR-Quantized/XLSR-Quantized.sh)|[128, 128, 3]|8.7|image enhancement||||
|qualcomm|[QuickSRNetSmall-Quantized](qualcomm/QuickSRNetSmall-Quantized/QuickSRNetSmall-Quantized.sh)|[128, 128, 3]|6.2|image enhancement||||
|qualcomm|[QuickSRNetMedium-Quantized](qualcomm/QuickSRNetMedium-Quantized/QuickSRNetMedium-Quantized.sh)|[128, 128, 3]|9.9|image enhancement||||
|qualcomm|[QuickSRNetLarge-Quantized](qualcomm/QuickSRNetLarge-Quantized/QuickSRNetLarge-Quantized.sh)|[128, 128, 3]|66.5|image enhancement||||
|qualcomm|[SESR-M5-Quantized](qualcomm/SESR-M5-Quantized/SESR-M5-Quantized.sh)|[128, 128, 3]|69.4|image enhancement||||
|tensorflow|[yolo-v4-tiny-tf](tensorflow/yolo-v4-tiny-tf/yolo-v4-tiny-tf.sh)|[416, 416, 3]|47.6|object detection|mAP⁵⁰⁻⁹⁵|12.3|12.5|
|tensorflow|[mobilenet_v2](tensorflow/mobilenet_v2/mobilenet_v2.sh)|[224, 224, 3]|15.0|classification|Top1|70.2|70.4|
|tensorflow|[yolo-v3-tiny-tf](tensorflow/yolo-v3-tiny-tf/yolo-v3-tiny-tf.sh)|[416, 416, 3]|35.1|object detection|mAP⁵⁰⁻⁹⁵|10.9|11.0|
|tensorflow|[yolo-v3-tf](tensorflow/yolo-v3-tf/yolo-v3-tf.sh)|[416, 416, 3]|592.7|object detection|mAP⁵⁰⁻⁹⁵|34.7|34.5|
|tensorflow|[efficientnet-lite0-int8](tensorflow/efficientnet-lite0-int8/efficientnet-lite0-int8.sh)|[224, 224, 3]|19.1|classification|Top1|70.5|70.4|
|tensorflow|[efficientnet-lite1-int8](tensorflow/efficientnet-lite1-int8/efficientnet-lite1-int8.sh)|[240, 240, 3]|26.6|classification|Top1|72.3|71.1|
|tensorflow|[efficientnet-lite2-int8](tensorflow/efficientnet-lite2-int8/efficientnet-lite2-int8.sh)|[260, 260, 3]|38.7|classification|Top1|71.1|71.3|
|tensorflow|[efficientnet-lite3-int8](tensorflow/efficientnet-lite3-int8/efficientnet-lite3-int8.sh)|[280, 280, 3]|54.0|classification|Top1|76.6|76.1|
|tensorflow|[efficientnet-lite4-int8](tensorflow/efficientnet-lite4-int8/efficientnet-lite4-int8.sh)|[300, 300, 3]|89.0|classification|Top1|77.8|77.8|
|tensorflow|[mobilenet_v1_050_160](tensorflow/mobilenet_v1_050_160/mobilenet_v1_050_160.sh)|[160, 160, 3]|3.8|classification|Top1|49.4|49.9|
|tensorflow|[mobilenet_v2_140_224](tensorflow/mobilenet_v2_140_224/mobilenet_v2_140_224.sh)|[224, 224, 3]|22.9|classification|Top1|75.5|75.4|
|tensorflow|[posenet](tensorflow/posenet/posenet.sh)|[273, 481, 3]|44.8|pose detection||||
|ultralytics|[yolov5n.relu](ultralytics/yolov5n.relu/yolov5n.relu.sh)|[416, 416, 3]|20.3|object detection|mAP⁵⁰⁻⁹⁵|19.2|19.1|
|ultralytics|[yolov5s.relu](ultralytics/yolov5s.relu/yolov5s.relu.sh)|[416, 416, 3]|69.7|object detection|mAP⁵⁰⁻⁹⁵|31.5|31.6|
|ultralytics|[yolov5n](ultralytics/yolov5n/yolov5n.sh)|[640, 640, 3]|49.4|object detection|mAP⁵⁰⁻⁹⁵|22.9|22.9|
|ultralytics|[yolov5n_512x288](ultralytics/yolov5n_512x288/yolov5n_512x288.sh)|[288, 512, 3]|18.5|object detection|mAP⁵⁰⁻⁹⁵|20.9|20.9|
|ultralytics|[yolov5s](ultralytics/yolov5s/yolov5s.sh)|[640, 640, 3]|172.8|object detection|mAP⁵⁰⁻⁹⁵|33.2|33.0|
|ultralytics|[yolov5m](ultralytics/yolov5m/yolov5m.sh)|[640, 640, 3]|540.3|object detection|mAP⁵⁰⁻⁹⁵|41.3|41.5|
|ultralytics|[yolov8n_FULL](ultralytics/yolov8n_FULL/yolov8n_FULL.sh)|[640, 640, 3]|120.0|object detection|mAP⁵⁰⁻⁹⁵|35.4|35.4|
|ultralytics|[yolov8s_FULL](ultralytics/yolov8s_FULL/yolov8s_FULL.sh)|[640, 640, 3]|284.1|object detection|mAP⁵⁰⁻⁹⁵|43.7|43.6|
|ultralytics|[yolov8m_FULL](ultralytics/yolov8m_FULL/yolov8m_FULL.sh)|[640, 640, 3]|752.1|object detection|mAP⁵⁰⁻⁹⁵|47.2|47.2|
|ultralytics|[yolov9t_FULL](ultralytics/yolov9t_FULL/yolov9t_FULL.sh)|[640, 640, 3]|134.1|object detection|mAP⁵⁰⁻⁹⁵|35.5|35.6|
|ultralytics|[yolov9s_FULL](ultralytics/yolov9s_FULL/yolov9s_FULL.sh)|[640, 640, 3]|247.8|object detection|mAP⁵⁰⁻⁹⁵|44.3|44.3|
|ultralytics|[yolov8n](ultralytics/yolov8n/yolov8n.sh)|[640, 640, 3]|63.2|object detection|mAP⁵⁰⁻⁹⁵|37.4|37.4|
|ultralytics|[yolov8n_argmax](ultralytics/yolov8n_argmax/yolov8n_argmax.sh)|[640, 640, 3]|64.1|object detection|mAP⁵⁰⁻⁹⁵|37.4|37.4|
|ultralytics|[yolov8n_512x288](ultralytics/yolov8n_512x288/yolov8n_512x288.sh)|[288, 512, 3]|23.6|object detection|mAP⁵⁰⁻⁹⁵|1.6|1.5|
|ultralytics|[yolov8n_512x288_argmax](ultralytics/yolov8n_512x288_argmax/yolov8n_512x288_argmax.sh)|[288, 512, 3]|24.1|object detection|mAP⁵⁰⁻⁹⁵|1.6|1.5|
|ultralytics|[yolov8n-pose_512x288](ultralytics/yolov8n-pose_512x288/yolov8n-pose_512x288.sh)|[288, 512, 3]|25.0|pose detection|Pose Detection|||
|ultralytics|[yolov8n-pose_512x288_split](ultralytics/yolov8n-pose_512x288_split/yolov8n-pose_512x288_split.sh)|[288, 512, 3]|24.8|pose detection|Pose Detection|||
|ultralytics|[yolov8s](ultralytics/yolov8s/yolov8s.sh)|[640, 640, 3]|227.6|object detection|mAP⁵⁰⁻⁹⁵|47.8|46.9|
|ultralytics|[yolov8m](ultralytics/yolov8m/yolov8m.sh)|[640, 640, 3]|695.8|object detection|mAP⁵⁰⁻⁹⁵|52.1|52.1|
|ultralytics|[yolov9s](ultralytics/yolov9s/yolov9s.sh)|[640, 640, 3]|190.9|object detection|mAP⁵⁰⁻⁹⁵|47.8|47.9|
|ultralytics|[yolov9t](ultralytics/yolov9t/yolov9t.sh)|[640, 640, 3]|77.0|object detection|mAP⁵⁰⁻⁹⁵|37.9|38.1|
|ultralytics|[yolov8n-cls](ultralytics/yolov8n-cls/yolov8n-cls.sh)|[224, 224, 3]|5.2|classification|Top1|67.2|67.3|
|ultralytics|[yolov8s-cls](ultralytics/yolov8s-cls/yolov8s-cls.sh)|[224, 224, 3]|12.9|classification|Top1|72.6|72.5|
|ultralytics|[yolov8m-cls](ultralytics/yolov8m-cls/yolov8m-cls.sh)|[224, 224, 3]|36.0|classification|Top1|75.5|75.9|
|ultralytics|[yolov8n-pose](ultralytics/yolov8n-pose/yolov8n-pose.sh)|[640, 640, 3]|66.7|pose detection|Pose Detection|||
|ultralytics|[yolov8n-seg](ultralytics/yolov8n-seg/yolov8n-seg.sh)|[640, 640, 3]|80.7|instance segmentation||||
|ultralytics|[yolov8n-obb](ultralytics/yolov8n-obb/yolov8n-obb.sh)|[1024, 1024, 3]|164.7|obb detection|mAP50-95|||
|ultralytics|[yolov3-tinyu_FULL](ultralytics/yolov3-tinyu_FULL/yolov3-tinyu_FULL.sh)|[640, 640, 3]|136.9|object detection|mAP⁵⁰⁻⁹⁵|29.6|29.5|
|ultralytics|[yolov5nu_FULL](ultralytics/yolov5nu_FULL/yolov5nu_FULL.sh)|[640, 640, 3]|119.6|object detection|mAP⁵⁰⁻⁹⁵|33.2|32.7|
|vectorblox|[yolov8n-relu](vectorblox/yolov8n-relu/yolov8n-relu.sh)|[640, 640, 3]|119.5|object detection|mAP⁵⁰⁻⁹⁵|33.3|33.2|
|ultralytics|[yolov5m.relu](ultralytics/yolov5m.relu/yolov5m.relu.sh)|[416, 416, 3]|175.9|object detection|mAP⁵⁰⁻⁹⁵|38.2|38.3|
|ultralytics|[yolov8x-cls](ultralytics/yolov8x-cls/yolov8x-cls.sh)|[224, 224, 3]|276.2|classification||||
|ultralytics|[yolov8l-cls](ultralytics/yolov8l-cls/yolov8l-cls.sh)|[224, 224, 3]|130.4|classification||||


## Compression


| Source | Tutorial | Input (H,W,C) | Runtime (ms) | Task  | Metric | TFLITE | VNNX |
| ----------------- | ------------- |-----------------|-------|---------------|--------------|------------|-----------|
|compressed|[yolov8n_comp66](compressed/yolov8n_comp66/yolov8n_comp66.sh)|[640, 640, 3]|45.9|object detection|mAP⁵⁰⁻⁹⁵|37.5|37.6|
|compressed|[yolov8s_comp68](compressed/yolov8s_comp68/yolov8s_comp68.sh)|[640, 640, 3]|108.0|object detection|mAP⁵⁰⁻⁹⁵|46.2|46.1|






## Unstructured Compression


| Source | Tutorial | Input (H,W,C) | Runtime (ms) | Task  | Metric | TFLITE |
| ----------------- | ------------- |-----------------|-------|---------------|--------------|------------|
|unstructure_compressed|[resnet18_86s_07p](unstructure_compressed/resnet18_86s_07p/resnet18_86s_07p.sh)|[224, 224, 3]|14.3|classification|Top1|65.6||
|unstructure_compressed|[yolov5n_70s_512x512](unstructure_compressed/yolov5n_70s_512x512/yolov5n_70s_512x512.sh)|[512, 512, 3]|19.8|object detection|mAP⁵⁰⁻⁹⁵|20.2||
|unstructure_compressed|[yolov8n_50s_25p_512x288](unstructure_compressed/yolov8n_50s_25p_512x288/yolov8n_50s_25p_512x288.sh)|[288, 512, 3]|15.4|object detection|mAP⁵⁰⁻⁹⁵|34.6||
|unstructure_compressed|[yolov8n_50s_25p_512x512](unstructure_compressed/yolov8n_50s_25p_512x512/yolov8n_50s_25p_512x512.sh)|[512, 512, 3]|25.3|object detection|mAP⁵⁰⁻⁹⁵|||
|unstructure_compressed|[yolov8n_pose_50s_25p_512x288_split](unstructure_compressed/yolov8n_pose_50s_25p_512x288_split/yolov8n_pose_50s_25p_512x288_split.sh)|[288, 512, 3]|15.5|pose detection||||
|unstructure_compressed|[yolov9s_70s_15p_512x288](unstructure_compressed/yolov9s_70s_15p_512x288/yolov9s_70s_15p_512x288.sh)|[288, 512, 3]|33.0|object detection||43.2||
|unstructure_compressed|[yolov9s_70s_15p_512x512](unstructure_compressed/yolov9s_70s_15p_512x512/yolov9s_70s_15p_512x512.sh)|[512, 512, 3]|55.6|object detection||||


