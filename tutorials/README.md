
# Tutorials

Below is a list of included tutorials. Frames per Second (FPS) assumes the cores are running at 125 MHz.
Accuracy is reported for 1000 samples.

<div class="acc_vnnx">

| Tutorial | Input (H,W,C) | V1000 FPS | Task  | Metric | TFLITE | VNNX |
| ------------- |-----------------|-------|---------------|--------------|------------|-----------|
|[yolov2-tiny-voc](darknet/yolov2-tiny-voc/yolov2-tiny-voc.sh)|[416, 416, 3]|20.9|object detection||||
|[yolov2-tiny](darknet/yolov2-tiny/yolov2-tiny.sh)|[416, 416, 3]|27.3|object detection||||
|[yolov3-tiny](darknet/yolov3-tiny/yolov3-tiny.sh)|[416, 416, 3]|25.9|object detection|mAP⁵⁰⁻⁹⁵|10.9|10.9|
|[efficientnet_lite0](mediapipe/efficientnet_lite0/efficientnet_lite0.sh)|[224, 224, 3]|61.7|classification|Top1|70.5|70.2|
|[efficientnet_lite2](mediapipe/efficientnet_lite2/efficientnet_lite2.sh)|[260, 260, 3]|31.6|classification|Top1|70.7|71.3|
|[onnx_resnet18-v1](onnx/onnx_resnet18-v1/onnx_resnet18-v1.sh)|[224, 224, 3]|32.7|classification|Top1|69.3|68.8|
|[onnx_resnet34-v1](onnx/onnx_resnet34-v1/onnx_resnet34-v1.sh)|[224, 224, 3]|17.9|classification|Top1|72.6|72.1|
|[onnx_squeezenet1.1](onnx/onnx_squeezenet1.1/onnx_squeezenet1.1.sh)|[224, 224, 3]|144.1|classification|Top1|54.0|54.0|
|[scrfd_500m_bnkps](onnx/scrfd_500m_bnkps/scrfd_500m_bnkps.sh)|[288, 512, 3]|86.7|face detection||||
|[mobilenet-v1-1.0-224](openvino/mobilenet-v1-1.0-224/mobilenet-v1-1.0-224.sh)|[224, 224, 3]|80.8|classification|Top1|69.6|70.3|
|[mobilenet-v2](openvino/mobilenet-v2/mobilenet-v2.sh)|[224, 224, 3]|59.1|classification|Top1|69.9|68.9|
|[mobilenet-v1-1.0-224-tf](openvino/mobilenet-v1-1.0-224-tf/mobilenet-v1-1.0-224-tf.sh)|[224, 224, 3]|80.5|classification|Top1|69.8|69.7|
|[mobilenet-v2-1.0-224](openvino/mobilenet-v2-1.0-224/mobilenet-v2-1.0-224.sh)|[224, 224, 3]|72.2|classification|Top1|69.9|70.6|
|[mobilenet-v2-1.4-224](openvino/mobilenet-v2-1.4-224/mobilenet-v2-1.4-224.sh)|[224, 224, 3]|46.5|classification|Top1|75.4|74.3|
|[mobilefacenet-arcface](openvino/mobilefacenet-arcface/mobilefacenet-arcface.sh)|[224, 224, 3]|66.2|face comparison||||
|[squeezenet1.0](openvino/squeezenet1.0/squeezenet1.0.sh)|[227, 227, 3]|63.1|classification|Top1|56.4|55.4|
|[squeezenet1.1](openvino/squeezenet1.1/squeezenet1.1.sh)|[227, 227, 3]|126.3|classification|Top1|56.5|56.6|
|[mobilenet-v1-0.25-128](openvino/mobilenet-v1-0.25-128/mobilenet-v1-0.25-128.sh)|[128, 128, 3]|427.5|classification|Top1|20.8|20.1|
|[deeplabv3](openvino/deeplabv3/deeplabv3.sh)|[513, 513, 3]|5.0|segmentation||||
|[torchvision_resnet18](pytorch/torchvision_resnet18/torchvision_resnet18.sh)|[224, 224, 3]|32.8|classification|Top1|69.0|68.7|
|[torchvision_resnet50](pytorch/torchvision_resnet50/torchvision_resnet50.sh)|[224, 224, 3]|11.5|classification|Top1|80.8|80.7|
|[torchvision_wide_resnet50_2](pytorch/torchvision_wide_resnet50_2/torchvision_wide_resnet50_2.sh)|[224, 224, 3]|5.0|classification|Top1|81.3|81.0|
|[torchvision_inception_v3](pytorch/torchvision_inception_v3/torchvision_inception_v3.sh)|[299, 299, 3]|7.1|classification|Top1|78.0|74.7|
|[torchvision_ssdlite320_mobilenet_v3_large](pytorch/torchvision_ssdlite320_mobilenet_v3_large/torchvision_ssdlite320_mobilenet_v3_large.sh)|[320, 320, 3]|23.6|object detection||||
|[torchvision_googlenet](pytorch/torchvision_googlenet/torchvision_googlenet.sh)|[224, 224, 3]|31.1|classification|Top1|62.5|61.9|
|[lpr_eu_v3](pytorch/lpr_eu_v3/lpr_eu_v3.sh)|[34, 146, 3]|198.8|plate recognition||||
|[torchvision_squeezenet1_0](pytorch/torchvision_squeezenet1_0/torchvision_squeezenet1_0.sh)|[227, 227, 3]|65.3|classification|Top1|58.6|59.3|
|[DeepLabV3-Plus-MobileNet-Quantized](qualcomm/DeepLabV3-Plus-MobileNet-Quantized/DeepLabV3-Plus-MobileNet-Quantized.sh)|[520, 520, 3]|1.1|segmentation||||
|[Midas-V2-Quantized](qualcomm/Midas-V2-Quantized/Midas-V2-Quantized.sh)|[256, 256, 3]|7.1|depth estimation||||
|[MobileNet-v2-Quantized](qualcomm/MobileNet-v2-Quantized/MobileNet-v2-Quantized.sh)|[224, 224, 3]|48.7|classification|Top1|69.0|70.6|
|[yolo-v4-tiny-tf](tensorflow/yolo-v4-tiny-tf/yolo-v4-tiny-tf.sh)|[416, 416, 3]|19.4|object detection|mAP⁵⁰⁻⁹⁵|12.3|12.4|
|[mobilenet_v2](tensorflow/mobilenet_v2/mobilenet_v2.sh)|[224, 224, 3]|70.9|classification|Top1|70.2|70.1|
|[yolo-v3-tiny-tf](tensorflow/yolo-v3-tiny-tf/yolo-v3-tiny-tf.sh)|[416, 416, 3]|25.9|object detection|mAP⁵⁰⁻⁹⁵|10.9|11.1|
|[yolo-v3-tf](tensorflow/yolo-v3-tf/yolo-v3-tf.sh)|[416, 416, 3]|2.0|object detection|mAP⁵⁰⁻⁹⁵|34.4|34.2|
|[efficientnet-lite0-int8](tensorflow/efficientnet-lite0-int8/efficientnet-lite0-int8.sh)|[224, 224, 3]|61.7|classification|Top1|70.5|70.2|
|[efficientnet-lite1-int8](tensorflow/efficientnet-lite1-int8/efficientnet-lite1-int8.sh)|[240, 240, 3]|41.9|classification|Top1|72.3|71.9|
|[efficientnet-lite2-int8](tensorflow/efficientnet-lite2-int8/efficientnet-lite2-int8.sh)|[260, 260, 3]|31.6|classification|Top1|71.1|71.3|
|[efficientnet-lite3-int8](tensorflow/efficientnet-lite3-int8/efficientnet-lite3-int8.sh)|[280, 280, 3]|22.2|classification|Top1|76.6|76.0|
|[efficientnet-lite4-int8](tensorflow/efficientnet-lite4-int8/efficientnet-lite4-int8.sh)|[300, 300, 3]|13.7|classification|Top1|77.8|77.8|
|[mobilenet_v1_050_160](tensorflow/mobilenet_v1_050_160/mobilenet_v1_050_160.sh)|[160, 160, 3]|239.0|classification|Top1|49.6|49.6|
|[mobilenet_v2_140_224](tensorflow/mobilenet_v2_140_224/mobilenet_v2_140_224.sh)|[224, 224, 3]|46.8|classification|Top1|75.5|75.7|
|[posenet](tensorflow/posenet/posenet.sh)|[273, 481, 3]|26.3|pose detection||||
|[yolov5n.relu](ultralytics/yolov5n.relu/yolov5n.relu.sh)|[416, 416, 3]|46.8|object detection|mAP⁵⁰⁻⁹⁵|19.2|19.0|
|[yolov5s.relu](ultralytics/yolov5s.relu/yolov5s.relu.sh)|[416, 416, 3]|16.7|object detection|mAP⁵⁰⁻⁹⁵|31.6|31.5|
|[yolov5m.relu](ultralytics/yolov5m.relu/yolov5m.relu.sh)|[416, 416, 3]|6.4|object detection||||
|[yolov5n](ultralytics/yolov5n/yolov5n.sh)|[640, 640, 3]|19.0|object detection|mAP|22.1|21.7|
|[yolov5n_512x288](ultralytics/yolov5n_512x288/yolov5n_512x288.sh)|[288, 512, 3]|52.1|object detection||||
|[yolov5s](ultralytics/yolov5s/yolov5s.sh)|[640, 640, 3]|6.5|object detection|mAP⁵⁰⁻⁹⁵|31.9|31.6|
|[yolov5m](ultralytics/yolov5m/yolov5m.sh)|[640, 640, 3]|2.4|object detection|mAP⁵⁰⁻⁹⁵|41.1|41.2|
|[yolov8n_FULL](ultralytics/yolov8n_FULL/yolov8n_FULL.sh)|[640, 640, 3]|7.8|object detection|mAP⁵⁰⁻⁹⁵|35.4|35.5|
|[yolov8s_FULL](ultralytics/yolov8s_FULL/yolov8s_FULL.sh)|[640, 640, 3]|3.6|object detection|mAP⁵⁰⁻⁹⁵|43.7|43.6|
|[yolov8m_FULL](ultralytics/yolov8m_FULL/yolov8m_FULL.sh)|[640, 640, 3]|1.5|object detection||||
|[yolov9t_FULL](ultralytics/yolov9t_FULL/yolov9t_FULL.sh)|[640, 640, 3]|6.8|object detection|mAP⁵⁰⁻⁹⁵|35.5|35.1|
|[yolov9s_FULL](ultralytics/yolov9s_FULL/yolov9s_FULL.sh)|[640, 640, 3]|3.6|object detection|mAP⁵⁰⁻⁹⁵|44.3|43.6|
|[yolov8n](ultralytics/yolov8n/yolov8n.sh)|[640, 640, 3]|13.9|object detection|mAP⁵⁰⁻⁹⁵|37.4|37.4|
|[yolov8n_512x288](ultralytics/yolov8n_512x288/yolov8n_512x288.sh)|[288, 512, 3]|36.9|object detection||||
|[yolov8n-pose_512x288](ultralytics/yolov8n-pose_512x288/yolov8n-pose_512x288.sh)|[288, 512, 3]|34.6|pose detection||||
|[yolov8s](ultralytics/yolov8s/yolov8s.sh)|[640, 640, 3]|4.5|object detection|mAP⁵⁰⁻⁹⁵|46.8|46.9|
|[yolov8m](ultralytics/yolov8m/yolov8m.sh)|[640, 640, 3]|1.6|object detection||||
|[yolov9s](ultralytics/yolov9s/yolov9s.sh)|[640, 640, 3]|4.5|object detection|mAP⁵⁰⁻⁹⁵|47.8|47.8|
|[yolov9t](ultralytics/yolov9t/yolov9t.sh)|[640, 640, 3]|11.2|object detection|mAP⁵⁰⁻⁹⁵|37.9|37.7|
|[yolov8n-cls](ultralytics/yolov8n-cls/yolov8n-cls.sh)|[224, 224, 3]|164.5|classification|Top1|61.6|61.3|
|[yolov8s-cls](ultralytics/yolov8s-cls/yolov8s-cls.sh)|[224, 224, 3]|70.9|classification|Top1|72.6|72.1|
|[yolov8m-cls](ultralytics/yolov8m-cls/yolov8m-cls.sh)|[224, 224, 3]|24.2|classification|Top1|73.0|73.1|
|[yolov8n-pose](ultralytics/yolov8n-pose/yolov8n-pose.sh)|[640, 640, 3]|13.1|pose detection||||
|[yolov8n-seg](ultralytics/yolov8n-seg/yolov8n-seg.sh)|[640, 640, 3]|10.6|instance segmentation||||
|[yolov8n-obb](ultralytics/yolov8n-obb/yolov8n-obb.sh)|[1024, 1024, 3]|5.2|obb detection||||
|[yolov3-tinyu_FULL](ultralytics/yolov3-tinyu_FULL/yolov3-tinyu_FULL.sh)|[640, 640, 3]|6.5|object detection|mAP⁵⁰⁻⁹⁵|29.6|29.5|
|[yolov5nu_FULL](ultralytics/yolov5nu_FULL/yolov5nu_FULL.sh)|[640, 640, 3]|7.8|object detection|mAP⁵⁰⁻⁹⁵|33.2|32.7|
|[yolov8n-relu](vectorblox/yolov8n-relu/yolov8n-relu.sh)|[640, 640, 3]|7.8|object detection|mAP⁵⁰⁻⁹⁵|46.0|44.8|


</div>
