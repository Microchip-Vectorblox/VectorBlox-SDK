# Tutorials

This directory contains scripts that will generate vectoblox compatible
Binary Large OBjects (BLOBs) for networks from various sources. The scripts
are intended as examples to be read and understood by users. Users can then
modify the scripts to generate their own networks.

Below is a list of included tutorials. Frames per Second (FPS) assumes the cores are running at 137 MHz (V1000), 143 MHz (V500) and 154 MHz (V250).



| Tutorial Name | Source Framework| Task  |Accuracy Metric|Accuracy Score FP32 / 8-bit |V1000 fps   |V500 fps   |V250 fps   | More information |
| ------------- |-----------------|-------|---------------|--------------|------------|-----------|-----------|----|
|[retinaface.mobilenet](pytorch/retinaface.mobilenet/retinaface.mobilenet.sh)|pytorch|Object Detection|||77.19|64.90|33.87|[More Info](https://github.com/biubug6/Pytorch_Retinaface)|
|[mobilefacenet-arcface](mxnet/mobilefacenet-arcface/mobilefacenet-arcface.sh)|mxnet|face_compare|||122.15|99.38|54.42|[More Info](https://github.com/deepinsight/insightface)|
|[mobilenet-v1-1.0-224](caffe/mobilenet-v1-1.0-224/mobilenet-v1-1.0-224.sh)|caffe|Classification|Top1|70.04/69.42|85.49|58.06|28.38|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/mobilenet-v1-1.0-224)|
|[mobilenet-v2](caffe/mobilenet-v2/mobilenet-v2.sh)|caffe|Classification|Top1|71.9/71.44|67.58|52.34|26.87|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/mobilenet-v2)|
|[mobilenet-v1-1.0-224-tf](tensorflow/mobilenet-v1-1.0-224-tf/mobilenet-v1-1.0-224-tf.sh)|tensorflow|Classification|Top1|72.18/71.62|85.53|59.00|28.72|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/mobilenet-v1-1.0-224-tf/)|
|[mobilenet-v2-1.0-224](tensorflow/mobilenet-v2-1.0-224/mobilenet-v2-1.0-224.sh)|tensorflow|Classification|Top1|72.44/72.16|80.47|65.86|35.03|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/mobilenet-v2-1.0-224)|
|[mobilenet-v2-1.4-224](tensorflow/mobilenet-v2-1.4-224/mobilenet-v2-1.4-224.sh)|tensorflow|Classification|Top1|75.12/74.64|52.25|40.98|21.42|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/mobilenet-v2-1.4-224/)|
|[resnet-50-tf](tensorflow/resnet-50-tf/resnet-50-tf.sh)|tensorflow|Classification|Top1|76.9/76.68|15.53|8.42|4.33|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/resnet-50-tf/)|
|[squeezenet1.0](caffe/squeezenet1.0/squeezenet1.0.sh)|caffe|Classification|Top1|57.08/57.1|66.95|44.20|22.05|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/squeezenet1.0/)|
|[squeezenet1.1](caffe/squeezenet1.1/squeezenet1.1.sh)|caffe|Classification|Top1|58.98/58.74|120.37|85.07|43.34|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/squeezenet1.1/)|
|[onnx_resnet18-v1](onnx/onnx_resnet18-v1/onnx_resnet18-v1.sh)|onnx|Classification|Top1|70.12/69.94|42.91|25.48|13.11|[More Info](https://github.com/onnx/models/tree/main/vision/classification/resnet)|
|[mnist](onnx/mnist/mnist.sh)|onnx|Classification|||5054.23|5280.65|4484.57|[More Info](https://github.com/onnx/models/tree/main/vision/classification/mnist)|
|[torchvision_resnet50](pytorch/torchvision_resnet50/torchvision_resnet50.sh)|pytorch|Classification|Top1|75.62/75.32|15.56|8.44|4.33|[More Info](https://pytorch.org/vision/0.9/models.html#torchvision.models.resnet50)|
|[torchvision_wide_resnet50_2](pytorch/torchvision_wide_resnet50_2/torchvision_wide_resnet50_2.sh)|pytorch|Classification|Top1|77.32/77.42||||[More Info](https://pytorch.org/vision/0.9/models.html#torchvision.models.wide_resnet50_2)|
|[yolov2-tiny-voc](darknet/yolov2-tiny-voc/yolov2-tiny-voc.sh)|darknet|Object Detection|mAP(VOC)|54.05/53.06|23.41|13.88|7.81|[More Info](https://pjreddie.com/darknet/yolo/)|
|[yolov2-tiny](darknet/yolov2-tiny/yolov2-tiny.sh)|darknet|Object Detection|mAP(COCO)|22.28/22.06|28.27|17.23|9.58|[More Info](https://pjreddie.com/darknet/yolo/)|
|[yolov3-tiny](darknet/yolov3-tiny/yolov3-tiny.sh)|darknet|Object Detection|mAP(COCO)|35.42/34.49|28.00|17.09|9.19|[More Info](https://pjreddie.com/darknet/yolo/)|
|[yolov2-voc](darknet/yolov2-voc/yolov2-voc.sh)|darknet|Object Detection|mAP(VOC)|74.79/74.13|6.21|3.44|1.67|[More Info](https://pjreddie.com/darknet/yolo/)|
|[BlazeFace](pytorch/BlazeFace/BlazeFace.sh)|pytorch|Classification|||274.80|255.21|152.99|[More Info](https://github.com/hollance/BlazeFace-PyTorch)|
|[license-plate-recognition-barrier-0007](tensorflow/license-plate-recognition-barrier-0007/license-plate-recognition-barrier-0007.sh)|tensorflow|Object Detection|||167.91|133.99|83.91|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/license-plate-recognition-barrier-0007)|
|[vehicle-license-plate-detection-barrier-0123](tensorflow/vehicle-license-plate-detection-barrier-0123/vehicle-license-plate-detection-barrier-0123.sh)|tensorflow|Object Detection|||94.84|90.64|20.72|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/vehicle-license-plate-detection-barrier-0123)|
|[mobilenet-v1-0.25-128](tensorflow/mobilenet-v1-0.25-128/mobilenet-v1-0.25-128.sh)|tensorflow|Classification|Top1|37.74/37.66|781.33|726.72|471.34|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/mobilenet-v1-0.25-128/)|
|[mobilenet-v1-0.50-160](tensorflow/mobilenet-v1-0.50-160/mobilenet-v1-0.50-160.sh)|tensorflow|Classification|Top1|57.5/55.14|308.72|253.22|161.37|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/mobilenet-v1-0.50-160/)|
|[mobilenet-v1-0.50-224](tensorflow/mobilenet-v1-0.50-224/mobilenet-v1-0.50-224.sh)|tensorflow|Classification|Top1|63.4/62.86|200.23|160.00|83.91|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/mobilenet-v1-0.50-224/)|
|[torchvision_resnet18](pytorch/torchvision_resnet18/torchvision_resnet18.sh)|pytorch|Classification|Top1|68.54/68.3|42.96|25.49|13.12|[More Info](https://pytorch.org/vision/0.9/models.html#torchvision.models.resnet18)|
|[torchvision_resnext101_32x8d](pytorch/torchvision_resnext101_32x8d/torchvision_resnext101_32x8d.sh)|pytorch|Classification|Top1|78.94/77.6||||[More Info](https://pytorch.org/vision/0.9/models.html#torchvision.models.resnext101_32x8d)|
|[torchvision_googlenet](pytorch/torchvision_googlenet/torchvision_googlenet.sh)|pytorch|Classification|Top1|62.16/61.92|34.77|24.23|13.21|[More Info](https://pytorch.org/vision/0.9/models.html#torchvision.models.googlenet)|
|[torchvision_squeezenet1_0](pytorch/torchvision_squeezenet1_0/torchvision_squeezenet1_0.sh)|pytorch|Classification|Top1|55.7/51.72|58.00|36.09|17.71|[More Info](https://pytorch.org/vision/0.9/models.html#torchvision.models.squeezenet1_0)|
|[onnx_resnet34-v1](onnx/onnx_resnet34-v1/onnx_resnet34-v1.sh)|onnx|Classification|Top1|74.14/73.92|23.36|13.52|7.10|[More Info](https://github.com/onnx/models/tree/main/vision/classification/resnet)|
|[onnx_resnet101-v1](onnx/onnx_resnet101-v1/onnx_resnet101-v1.sh)|onnx|Classification|Top1|76.84/76.38|10.75|6.24|3.21|[More Info](https://github.com/onnx/models/tree/main/vision/classification/resnet)|
|[onnx_squeezenet1.0](onnx/onnx_squeezenet1.0/onnx_squeezenet1.0.sh)|onnx|Classification|Top1|55.38/55.0|138.15|96.00|49.81|[More Info](https://github.com/onnx/models/tree/main/vision/classification/squeezenet)|
|[onnx_squeezenet1.1](onnx/onnx_squeezenet1.1/onnx_squeezenet1.1.sh)|onnx|Classification|Top1|56.52/54.48|99.54|61.46|30.47|[More Info](https://github.com/onnx/models/tree/main/vision/classification/squeezenet)|
|[ssd_mobilenet_v1_coco](tensorflow/ssd_mobilenet_v1_coco/ssd_mobilenet_v1_coco.sh)|tensorflow|Object Detection|mAP(COCO)|14.12/14.03|39.61|26.01|9.75|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/ssd_mobilenet_v1_coco/)|
|[ssdlite_mobilenet_v2](tensorflow/ssdlite_mobilenet_v2/ssdlite_mobilenet_v2.sh)|tensorflow|Object Detection|mAP(COCO)|14.52/14.01|37.26|29.01|16.08|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/ssdlite_mobilenet_v2/)|
|[yolo-v4-tiny-tf](tensorflow/yolo-v4-tiny-tf/yolo-v4-tiny-tf.sh)|tensorflow|Object Detection|mAP(COCO)|39.61/38.78|22.33|13.28|6.29|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/yolo-v4-tiny-tf/)|
|[mobilenet_v1_050_160](tensorflow2/mobilenet_v1_050_160/mobilenet_v1_050_160.sh)|tensorflow2|Classification|Top1|57.52/55.1|321.93|262.11|165.16|[More Info](https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/classification/5)|
|[mobilenet_v2_140_224](tensorflow2/mobilenet_v2_140_224/mobilenet_v2_140_224.sh)|tensorflow2|Classification|Top1|75.12/74.62|52.75|41.21|21.42|[More Info](https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/5)|