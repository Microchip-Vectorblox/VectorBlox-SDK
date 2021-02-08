# Tutorials

This directory contains scripts that will generate vectoblox compatible
Binary Large OBjects (BLOBs) for networks from various sources. The scripts
are intended as examples to be read and understood by users. Users can then
modify the scripts to generate their own networks.

Below is a list of included tutorials



| Tutorial Name | Source Framework| Task  |Accuracy Metric|Accuracy Score|V1000 kcycles|V500 kcycles|V250 kcycles| More information |
| ------------- |-----------------|-------|---------------|--------------|------------|-----------|-----------|----|
|[mobilenet-v1-1.0-224](caffe/mobilenet-v1-1.0-224/mobilenet-v1-1.0-224.sh)|caffe|Classification|topk|65.4|5637|||[More Info](https://github.com/opencv/open_model_zoo/blob/2019_R3.1/models/public/mobilenet-v1-1.0-224/mobilenet-v1-1.0-224.md)|
|[mobilenet-v2](caffe/mobilenet-v2/mobilenet-v2.sh)|caffe|Classification|topk|70|10358|||[More Info](https://github.com/opencv/open_model_zoo/blob/2019_R3.1/models/public/mobilenet-v2/mobilenet-v2.md)|
|[mobilenet-v1-1.0-224-tf](tensorflow/mobilenet-v1-1.0-224-tf/mobilenet-v1-1.0-224-tf.sh)|tensorflow|Classification|topk|68.8||||[More Info](https://github.com/opencv/open_model_zoo/blob/2019_R3.1/models/public/mobilenet-v1-1.0-224-tf/mobilenet-v1-1.0-224-tf.md)|
|[mobilenet-v2-1.0-224](tensorflow/mobilenet-v2-1.0-224/mobilenet-v2-1.0-224.sh)|tensorflow|Classification|topk|69||||[More Info](https://github.com/opencv/open_model_zoo/blob/2019_R3.1/models/public/mobilenet-v2-1.0-224/mobilenet-v2-1.0-224.md)|
|[mobilenet-v2-1.4-224](tensorflow/mobilenet-v2-1.4-224/mobilenet-v2-1.4-224.sh)|tensorflow|Classification|topk|74.8||||[More Info](https://github.com/opencv/open_model_zoo/blob/2019_R3.1/models/public/mobilenet-v2-1.4-224/mobilenet-v2-1.4-224.md)|
|[resnet-50-tf](tensorflow/resnet-50-tf/resnet-50-tf.sh)|tensorflow|Classification|topk|75.2|3955|||[More Info](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/resnet-50-tf/resnet-50-tf.md)|
|[squeezenet1.0](caffe/squeezenet1.0/squeezenet1.0.sh)|caffe|Classification|topk|58.2|25298|||[More Info](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/squeezenet1.0/squeezenet1.0.md)|
|[squeezenet1.1](caffe/squeezenet1.1/squeezenet1.1.sh)|caffe|Classification|topk|58.4|6202|||[More Info](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/squeezenet1.1/squeezenet1.1.md)|
|[Sphereface](caffe/Sphereface/Sphereface.sh)|caffe|face_compare|||6827|||[More Info](https://github.com/openvinotoolkit/open_model_zoo/blob/2020.4/models/public/Sphereface/Sphereface.md)|
|[onnx_resnet18-v1](onnx/onnx_resnet18-v1/onnx_resnet18-v1.sh)|onnx|Classification|topk|71.4|18745|||[More Info](https://github.com/onnx/models/tree/master/vision/classification/resnet)|
|[mnist](onnx/mnist/mnist.sh)|onnx|Classification|||87|||[More Info](https://github.com/onnx/models/tree/master/vision/classification/mnist)|
|[torchvision_resnet50](pytorch/torchvision_resnet50/torchvision_resnet50.sh)|pytorch|Classification|topk|75.2|3953|||[More Info](https://pytorch.org/docs/stable/torchvision/models.html)|
|[torchvision_wide_resnet50_2](pytorch/torchvision_wide_resnet50_2/torchvision_wide_resnet50_2.sh)|pytorch|Classification|topk|76.2||||[More Info](https://pytorch.org/docs/stable/torchvision/models.html)|
|[yolov2-tiny-voc](darknet/yolov2-tiny-voc/yolov2-tiny-voc.sh)|darknet|Object Detection|mAP|0.224|8907|||[More Info](https://pjreddie.com/darknet/yolo/)|
|[yolov2-tiny](darknet/yolov2-tiny/yolov2-tiny.sh)|darknet|Object Detection|mAP|0.138|9056|||[More Info](https://pjreddie.com/darknet/yolo/)|
|[yolov3-tiny](darknet/yolov3-tiny/yolov3-tiny.sh)|darknet|Object Detection|mAP|0.136|10147|||[More Info](https://pjreddie.com/darknet/yolo/)|
|[yolov2-voc](darknet/yolov2-voc/yolov2-voc.sh)|darknet|Object Detection|mAP|0.351||||[More Info](https://pjreddie.com/darknet/yolo/)|
|[yolov3](darknet/yolov3/yolov3.sh)|darknet|Object Detection||||||[More Info](https://pjreddie.com/darknet/yolo/)|
|[yolov2](darknet/yolov2/yolov2.sh)|darknet|Object Detection||||||[More Info](https://pjreddie.com/darknet/yolo/)|
|[BlazeFace](pytorch/BlazeFace/BlazeFace.sh)|pytorch|Classification|||2337|||[More Info](https://github.com/hollance/BlazeFace-PyTorch)|