# Tutorials

This directory contains scripts that will generate vectoblox compatible
Binary Large OBjects (BLOBs) for networks from various sources. The scripts
are intended as examples to be read and understood by users. Users can then
modify the scripts to generate their own networks.

Below is a list of included tutorials



| Tutorial Name | Source Framework| Task  |Accuracy Metric|Accuracy Score|V1000 kcycles|V500 kcycles|V250 kcycles| More information |
| ------------- |-----------------|-------|---------------|--------------|------------|-----------|-----------|----|
|[mobilenet-v1-1.0-224](caffe/mobilenet-v1-1.0-224/mobilenet-v1-1.0-224.sh)|caffe|Classification|topK|68.85|1623|2538|6010|[More Info](https://github.com/opencv/open_model_zoo/blob/2019_R3.1/models/public/mobilenet-v1-1.0-224/mobilenet-v1-1.0-224.md)|
|[mobilenet-v2](caffe/mobilenet-v2/mobilenet-v2.sh)|caffe|Classification|topK|71.95|2037|2757|6143|[More Info](https://github.com/opencv/open_model_zoo/blob/2019_R3.1/models/public/mobilenet-v2/mobilenet-v2.md)|
|[mobilenet-v1-1.0-224-tf](tensorflow/mobilenet-v1-1.0-224-tf/mobilenet-v1-1.0-224-tf.sh)|tensorflow|Classification|topK|72.2|1613|2505|5641|[More Info](https://github.com/opencv/open_model_zoo/blob/2019_R3.1/models/public/mobilenet-v1-1.0-224-tf/mobilenet-v1-1.0-224-tf.md)|
|[mobilenet-v2-1.0-224](tensorflow/mobilenet-v2-1.0-224/mobilenet-v2-1.0-224.sh)|tensorflow|Classification|topK|71.8|1710|2191|4538|[More Info](https://github.com/opencv/open_model_zoo/blob/2019_R3.1/models/public/mobilenet-v2-1.0-224/mobilenet-v2-1.0-224.md)|
|[mobilenet-v2-1.4-224](tensorflow/mobilenet-v2-1.4-224/mobilenet-v2-1.4-224.sh)|tensorflow|Classification|topK|73.9|2625|3575|7665|[More Info](https://github.com/opencv/open_model_zoo/blob/2019_R3.1/models/public/mobilenet-v2-1.4-224/mobilenet-v2-1.4-224.md)|
|[resnet-50-tf](tensorflow/resnet-50-tf/resnet-50-tf.sh)|tensorflow|Classification|topK|76.5|8786|16999|35505|[More Info](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/resnet-50-tf/resnet-50-tf.md)|
|[squeezenet1.0](caffe/squeezenet1.0/squeezenet1.0.sh)|caffe|Classification|topK|55.25|2326|3522|7542|[More Info](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/squeezenet1.0/squeezenet1.0.md)|
|[squeezenet1.1](caffe/squeezenet1.1/squeezenet1.1.sh)|caffe|Classification|topK|57.2|1301|1849|3879|[More Info](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/squeezenet1.1/squeezenet1.1.md)|
|[Sphereface](caffe/Sphereface/Sphereface.sh)|caffe|face_compare|||3784|6211|12023|[More Info](https://github.com/openvinotoolkit/open_model_zoo/blob/2020.4/models/public/Sphereface/Sphereface.md)|
|[onnx_resnet18-v1](onnx/onnx_resnet18-v1/onnx_resnet18-v1.sh)|onnx|Classification|topK|70.5|3304|5740|11902|[More Info](https://github.com/onnx/models/tree/master/vision/classification/resnet)|
|[mnist](onnx/mnist/mnist.sh)|onnx|Classification|||21|20|27|[More Info](https://github.com/onnx/models/tree/master/vision/classification/mnist)|
|[torchvision_resnet50](pytorch/torchvision_resnet50/torchvision_resnet50.sh)|pytorch|Classification|topK|75.2|8909|17116|35766|[More Info](https://pytorch.org/docs/stable/torchvision/models.html)|
|[torchvision_wide_resnet50_2](pytorch/torchvision_wide_resnet50_2/torchvision_wide_resnet50_2.sh)|pytorch|Classification|topK|76.2||||[More Info](https://pytorch.org/docs/stable/torchvision/models.html)|
|[yolov2-tiny-voc](darknet/yolov2-tiny-voc/yolov2-tiny-voc.sh)|darknet|Object Detection|mAP(VOC)|52.68|5852|10303|19703|[More Info](https://pjreddie.com/darknet/yolo/)|
|[yolov2-tiny](darknet/yolov2-tiny/yolov2-tiny.sh)|darknet|Object Detection|mAP(COCO)|24.03|4844|8300|16058|[More Info](https://pjreddie.com/darknet/yolo/)|
|[yolov3-tiny](darknet/yolov3-tiny/yolov3-tiny.sh)|darknet|Object Detection|mAP(COCO)|36.81|4895|8365|16805|[More Info](https://pjreddie.com/darknet/yolo/)|
|[BlazeFace](pytorch/BlazeFace/BlazeFace.sh)|pytorch|Classification|||470|532|977|[More Info](https://github.com/hollance/BlazeFace-PyTorch)|