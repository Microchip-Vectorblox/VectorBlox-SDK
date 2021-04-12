# Tutorials

This directory contains scripts that will generate vectoblox compatible
Binary Large OBjects (BLOBs) for networks from various sources. The scripts
are intended as examples to be read and understood by users. Users can then
modify the scripts to generate their own networks.

Below is a list of included tutorials. Frames per Second (FPS) assumes core is running at 130MHz.



| Tutorial Name | Source Framework| Task  |Accuracy Metric|Accuracy Score FP32 / 8-bit |V1000 fps   |V500 fps   |V250 fps   | More information |
| ------------- |-----------------|-------|---------------|--------------|------------|-----------|-----------|----|
|[mobilenet-v1-1.0-224](caffe/mobilenet-v1-1.0-224/mobilenet-v1-1.0-224.sh)|caffe|Classification|topK|69.3/68.85|80.06|51.20|21.63|[More Info](https://github.com/opencv/open_model_zoo/blob/2019_R3.1/models/public/mobilenet-v1-1.0-224/mobilenet-v1-1.0-224.md)|
|[mobilenet-v2](caffe/mobilenet-v2/mobilenet-v2.sh)|caffe|Classification|topK|72.05/71.95|63.80|47.14|21.16|[More Info](https://github.com/opencv/open_model_zoo/blob/2019_R3.1/models/public/mobilenet-v2/mobilenet-v2.md)|
|[mobilenet-v1-1.0-224-tf](tensorflow/mobilenet-v1-1.0-224-tf/mobilenet-v1-1.0-224-tf.sh)|tensorflow|Classification|topK|72.95/72.2|80.56|51.89|23.05|[More Info](https://github.com/opencv/open_model_zoo/blob/2019_R3.1/models/public/mobilenet-v1-1.0-224-tf/mobilenet-v1-1.0-224-tf.md)|
|[mobilenet-v2-1.0-224](tensorflow/mobilenet-v2-1.0-224/mobilenet-v2-1.0-224.sh)|tensorflow|Classification|topK|72.05/71.8|76.01|59.33|28.64|[More Info](https://github.com/opencv/open_model_zoo/blob/2019_R3.1/models/public/mobilenet-v2-1.0-224/mobilenet-v2-1.0-224.md)|
|[mobilenet-v2-1.4-224](tensorflow/mobilenet-v2-1.4-224/mobilenet-v2-1.4-224.sh)|tensorflow|Classification|topK|73.9/73.9|49.51|36.36|16.96|[More Info](https://github.com/opencv/open_model_zoo/blob/2019_R3.1/models/public/mobilenet-v2-1.4-224/mobilenet-v2-1.4-224.md)|
|[resnet-50-tf](tensorflow/resnet-50-tf/resnet-50-tf.sh)|tensorflow|Classification|topK|76.7/76.5|14.80|7.65|3.66|[More Info](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/resnet-50-tf/resnet-50-tf.md)|
|[squeezenet1.0](caffe/squeezenet1.0/squeezenet1.0.sh)|caffe|Classification|topK|56.35/55.25|55.89|36.91|17.24|[More Info](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/squeezenet1.0/squeezenet1.0.md)|
|[squeezenet1.1](caffe/squeezenet1.1/squeezenet1.1.sh)|caffe|Classification|topK|57.45/57.2|99.90|70.28|33.51|[More Info](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/squeezenet1.1/squeezenet1.1.md)|
|[Sphereface](caffe/Sphereface/Sphereface.sh)|caffe|face_compare|||34.35|20.93|10.81|[More Info](https://github.com/openvinotoolkit/open_model_zoo/blob/2020.4/models/public/Sphereface/Sphereface.md)|
|[onnx_resnet18-v1](onnx/onnx_resnet18-v1/onnx_resnet18-v1.sh)|onnx|Classification|topK|70.5/70.5|39.34|22.65|10.92|[More Info](https://github.com/onnx/models/tree/master/vision/classification/resnet)|
|[mnist](onnx/mnist/mnist.sh)|onnx|Classification|||6045.11|6403.31|4659.16|[More Info](https://github.com/onnx/models/tree/master/vision/classification/mnist)|
|[torchvision_resnet50](pytorch/torchvision_resnet50/torchvision_resnet50.sh)|pytorch|Classification|topK|75.3/75.2|14.59|7.59|3.63|[More Info](https://pytorch.org/docs/stable/torchvision/models.html)|
|[torchvision_wide_resnet50_2](pytorch/torchvision_wide_resnet50_2/torchvision_wide_resnet50_2.sh)|pytorch|Classification|topK|||||[More Info](https://pytorch.org/docs/stable/torchvision/models.html)|
|[yolov2-tiny-voc](darknet/yolov2-tiny-voc/yolov2-tiny-voc.sh)|darknet|Object Detection|mAP(VOC)|53.46/52.68|22.21|12.62|6.60|[More Info](https://pjreddie.com/darknet/yolo/)|
|[yolov2-tiny](darknet/yolov2-tiny/yolov2-tiny.sh)|darknet|Object Detection|mAP(COCO)|24.08/24.03|26.84|15.66|8.10|[More Info](https://pjreddie.com/darknet/yolo/)|
|[yolov3-tiny](darknet/yolov3-tiny/yolov3-tiny.sh)|darknet|Object Detection|mAP(COCO)|37.43/36.81|26.55|15.54|7.74|[More Info](https://pjreddie.com/darknet/yolo/)|
|[BlazeFace](pytorch/BlazeFace/BlazeFace.sh)|pytorch|Classification|||276.12|243.96|132.93|[More Info](https://github.com/hollance/BlazeFace-PyTorch)|