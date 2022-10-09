# Tutorials

This directory contains scripts that will generate vectoblox compatible
Binary Large OBjects (BLOBs) for networks from various sources. The scripts
are intended as examples to be read and understood by users. Users can then
modify the scripts to generate their own networks.

Below is a list of included tutorials. Frames per Second (FPS) assumes the cores are running at 137 MHz (V1000), 143 MHz (V500) and 154 MHz (V250).



| Tutorial Name | Source Framework| Task  |Accuracy Metric|Accuracy Score FP32 / 8-bit |V1000 fps   |V500 fps   |V250 fps   | More information |
| ------------- |-----------------|-------|---------------|--------------|------------|-----------|-----------|----|
|[ultralytics.yolov5n.relu](pytorch/ultralytics.yolov5n.relu/ultralytics.yolov5n.relu.sh)|pytorch|Object Detection|mAP(COCO)|32.43/32.1|46.42|32.42|16.48|[More Info](https://github.com/ultralytics/yolov5)|
|[ultralytics.yolov5s.relu](pytorch/ultralytics.yolov5s.relu/ultralytics.yolov5s.relu.sh)|pytorch|Object Detection|mAP(COCO)|48.9/48.61|18.36|10.22|4.86|[More Info](https://github.com/ultralytics/yolov5)|
|[ultralytics.yolov5m.relu](pytorch/ultralytics.yolov5m.relu/ultralytics.yolov5m.relu.sh)|pytorch|Object Detection|mAP(COCO)|54.6/54.38|6.68|3.56|1.85|[More Info](https://github.com/ultralytics/yolov5)|
|[yolo-v3-tf](tensorflow/yolo-v3-tf/yolo-v3-tf.sh)|tensorflow|Object Detection|||2.35|1.22||[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/yolo-v3-tf/)|
|[yolov2-voc](darknet/yolov2-voc/yolov2-voc.sh)|darknet|Object Detection|mAP(VOC)|74.79/74.13|6.13|3.38|1.61|[More Info](https://pjreddie.com/darknet/yolo/)|
|[scrfd_500m_bnkps](onnx/scrfd_500m_bnkps/scrfd_500m_bnkps.sh)|onnx|Face Detection|||79.37|63.25|33.25|[More Info](https://insightface.ai/scrfd)|
|[genderage](onnx/genderage/genderage.sh)|onnx|Gender Age Estimation|||862.19|850.06|584.32|[More Info](https://github.com/deepinsight/insightface/tree/master/model_zoo#41-genderage)|
|[retinaface.mobilenet](pytorch/retinaface.mobilenet/retinaface.mobilenet.sh)|pytorch|Object Detection|||74.10|61.67|32.77|[More Info](https://github.com/biubug6/Pytorch_Retinaface)|
|[mobilefacenet-arcface](mxnet/mobilefacenet-arcface/mobilefacenet-arcface.sh)|mxnet|face_compare|||121.06|97.43|53.20|[More Info](https://github.com/deepinsight/insightface)|
|[mobilenet-v1-1.0-224](caffe/mobilenet-v1-1.0-224/mobilenet-v1-1.0-224.sh)|caffe|Classification|Top1|70.04/69.44|83.46|55.86|27.22|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/mobilenet-v1-1.0-224)|
|[mobilenet-v2](caffe/mobilenet-v2/mobilenet-v2.sh)|caffe|Classification|Top1|71.9/71.56|65.15|49.72|25.99|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/mobilenet-v2)|
|[mobilenet-v1-1.0-224-tf](tensorflow/mobilenet-v1-1.0-224-tf/mobilenet-v1-1.0-224-tf.sh)|tensorflow|Classification|Top1|72.18/71.62|83.27|56.84|27.60|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/mobilenet-v1-1.0-224-tf/)|
|[mobilenet-v2-1.0-224](tensorflow/mobilenet-v2-1.0-224/mobilenet-v2-1.0-224.sh)|tensorflow|Classification|Top1|72.44/72.32|77.45|62.98|34.21|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/mobilenet-v2-1.0-224)|
|[mobilenet-v2-1.4-224](tensorflow/mobilenet-v2-1.4-224/mobilenet-v2-1.4-224.sh)|tensorflow|Classification|Top1|75.12/74.68|50.35|39.09|20.79|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/mobilenet-v2-1.4-224/)|
|[resnet-50-tf](tensorflow/resnet-50-tf/resnet-50-tf.sh)|tensorflow|Classification|Top1|76.9/76.74|14.75|7.75|4.06|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/resnet-50-tf/)|
|[onnx_resnet18-v1](onnx/onnx_resnet18-v1/onnx_resnet18-v1.sh)|onnx|Classification|Top1|70.12/69.94|42.14|24.84|12.96|[More Info](https://github.com/onnx/models/tree/main/vision/classification/resnet)|
|[torchvision_resnet50](pytorch/torchvision_resnet50/torchvision_resnet50.sh)|pytorch|Classification|Top1|75.62/75.1|14.66|7.66|4.02|[More Info](https://pytorch.org/vision/0.9/models.html#torchvision.models.resnet50)|
|[torchvision_wide_resnet50_2](pytorch/torchvision_wide_resnet50_2/torchvision_wide_resnet50_2.sh)|pytorch|Classification|Top1|77.32/77.42|6.90|3.32||[More Info](https://pytorch.org/vision/0.9/models.html#torchvision.models.wide_resnet50_2)|
|[BlazeFace](pytorch/BlazeFace/BlazeFace.sh)|pytorch|Classification|||271.12|254.99|158.48|[More Info](https://github.com/hollance/BlazeFace-PyTorch)|
|[posenet](tfjs/posenet/posenet.sh)|tfjs|Key Point|keypoint(COCO)|0.136/0.13||||[More Info](https://github.com/tensorflow/tfjs-models/blob/master/posenet)|
|[license-plate-recognition-barrier-0007](tensorflow/license-plate-recognition-barrier-0007/license-plate-recognition-barrier-0007.sh)|tensorflow|Object Detection|||163.45|128.95|81.18|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/license-plate-recognition-barrier-0007)|
|[vehicle-license-plate-detection-barrier-0123](tensorflow/vehicle-license-plate-detection-barrier-0123/vehicle-license-plate-detection-barrier-0123.sh)|tensorflow|Object Detection|||92.91|89.11|52.15|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/vehicle-license-plate-detection-barrier-0123)|
|[mobilenet-v1-0.25-128](tensorflow/mobilenet-v1-0.25-128/mobilenet-v1-0.25-128.sh)|tensorflow|Classification|Top1|37.74/37.68|768.55|754.98|489.77|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/mobilenet-v1-0.25-128/)|
|[torchvision_resnet18](pytorch/torchvision_resnet18/torchvision_resnet18.sh)|pytorch|Classification|Top1|68.54/68.3|42.07|24.81|12.95|[More Info](https://pytorch.org/vision/0.9/models.html#torchvision.models.resnet18)|
|[torchvision_resnext101_32x8d](pytorch/torchvision_resnext101_32x8d/torchvision_resnext101_32x8d.sh)|pytorch|Classification|Top1|78.94/77.84|0.97|0.82||[More Info](https://pytorch.org/vision/0.9/models.html#torchvision.models.resnext101_32x8d)|
|[torchvision_googlenet](pytorch/torchvision_googlenet/torchvision_googlenet.sh)|pytorch|Classification|Top1|62.16/61.92|33.87|23.44|12.85|[More Info](https://pytorch.org/vision/0.9/models.html#torchvision.models.googlenet)|
|[torchvision_squeezenet1_0](pytorch/torchvision_squeezenet1_0/torchvision_squeezenet1_0.sh)|pytorch|Classification|Top1|55.7/51.24|54.12|33.26|16.70|[More Info](https://pytorch.org/vision/0.9/models.html#torchvision.models.squeezenet1_0)|
|[onnx_resnet34-v1](onnx/onnx_resnet34-v1/onnx_resnet34-v1.sh)|onnx|Classification|Top1|74.14/73.78|23.13|13.30|7.05|[More Info](https://github.com/onnx/models/tree/main/vision/classification/resnet)|
|[onnx_resnet101-v1](onnx/onnx_resnet101-v1/onnx_resnet101-v1.sh)|onnx|Classification|Top1|76.84/76.48|9.07|5.11|2.63|[More Info](https://github.com/onnx/models/tree/main/vision/classification/resnet)|
|[onnx_squeezenet1.0](onnx/onnx_squeezenet1.0/onnx_squeezenet1.0.sh)|onnx|Classification|Top1|55.38/55.0|132.77|92.04|48.10|[More Info](https://github.com/onnx/models/tree/main/vision/classification/squeezenet)|
|[onnx_squeezenet1.1](onnx/onnx_squeezenet1.1/onnx_squeezenet1.1.sh)|onnx|Classification|Top1|56.52/53.74|91.25|55.73|28.13|[More Info](https://github.com/onnx/models/tree/main/vision/classification/squeezenet)|
|[ssd_mobilenet_v1_coco](tensorflow/ssd_mobilenet_v1_coco/ssd_mobilenet_v1_coco.sh)|tensorflow|Object Detection|mAP(COCO)|14.23/14.11|37.18|24.41|9.04|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/ssd_mobilenet_v1_coco/)|
|[ssdlite_mobilenet_v2](tensorflow/ssdlite_mobilenet_v2/ssdlite_mobilenet_v2.sh)|tensorflow|Object Detection|mAP(COCO)|14.51/14.16|35.31|27.27|15.58|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/ssdlite_mobilenet_v2/)|
|[yolo-v4-tiny-tf](tensorflow/yolo-v4-tiny-tf/yolo-v4-tiny-tf.sh)|tensorflow|Object Detection|mAP(COCO)|39.61/38.77|21.95|12.94|6.07|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/yolo-v4-tiny-tf/)|
|[mobilenet_v1_050_160](tensorflow2/mobilenet_v1_050_160/mobilenet_v1_050_160.sh)|tensorflow2|Classification|Top1|57.52/55.14|318.78|258.74|163.27|[More Info](https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/classification/5)|
|[mobilenet_v2_140_224](tensorflow2/mobilenet_v2_140_224/mobilenet_v2_140_224.sh)|tensorflow2|Classification|Top1|75.12/74.78|50.81|39.31|20.85|[More Info](https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/5)|
|[license-plate-recognition-latin](tensorflow/license-plate-recognition-latin/license-plate-recognition-latin.sh)|tensorflow|Object Detection|||145.71|113.65|70.13|[More Info](https://github.com/openvinotoolkit/training_extensions/tree/misc/misc/tensorflow_toolkit/lpr)|
|[deeplabv3](tensorflow/deeplabv3/deeplabv3.sh)|tensorflow|segmentation|||2.19|||[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/deeplabv3)|
|[Sphereface](caffe/Sphereface/Sphereface.sh)|caffe|face_compare|||36.32|22.99|12.81|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/Sphereface/)|
|[ultralytics.yolov3-tiny.plates](pytorch/ultralytics.yolov3-tiny.plates/ultralytics.yolov3-tiny.plates.sh)|pytorch|Object Detection|||28.33|17.34|9.59|[More Info](https://github.com/ultralytics/yolov3)|
|[retinaface.mobilenet.320](pytorch/retinaface.mobilenet.320/retinaface.mobilenet.320.sh)|pytorch|Object Detection|||94.91|79.14|47.21|[More Info](https://github.com/biubug6/Pytorch_Retinaface)|
|[retinaface.mobilenet.640](pytorch/retinaface.mobilenet.640/retinaface.mobilenet.640.sh)|pytorch|Object Detection|||27.27|22.32|11.52|[More Info](https://github.com/biubug6/Pytorch_Retinaface)|
|[retinaface.resnet](pytorch/retinaface.resnet/retinaface.resnet.sh)|pytorch|Object Detection|||5.78|3.03|1.24|[More Info](https://github.com/biubug6/Pytorch_Retinaface)|
|[retinaface.resnet.640](pytorch/retinaface.resnet.640/retinaface.resnet.640.sh)|pytorch|Object Detection|||1.34|||[More Info](https://github.com/biubug6/Pytorch_Retinaface)|
|[yolov2-tiny-voc](darknet/yolov2-tiny-voc/yolov2-tiny-voc.sh)|darknet|Object Detection|mAP(VOC)|54.05/52.97|23.23|13.77|7.76|[More Info](https://pjreddie.com/darknet/yolo/)|
|[yolov2-tiny](darknet/yolov2-tiny/yolov2-tiny.sh)|darknet|Object Detection|mAP(COCO)|22.28/22.09|27.90|17.02|9.49|[More Info](https://pjreddie.com/darknet/yolo/)|
|[yolov3-tiny](darknet/yolov3-tiny/yolov3-tiny.sh)|darknet|Object Detection|mAP(COCO)|35.42/34.45|27.60|16.82|8.94|[More Info](https://pjreddie.com/darknet/yolo/)|
|[myfirstcnn](tensorflow2/myfirstcnn/myfirstcnn.sh)|tensorflow2|Classification||||||[More Info]()|
|[myfirstcnn3](tensorflow2/myfirstcnn3/myfirstcnn3.sh)|tensorflow2|Classification||||||[More Info]()|
|[squeezenet1.0](caffe/squeezenet1.0/squeezenet1.0.sh)|caffe|Classification|Top1|57.08/57.1|64.37|42.01|21.43|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/squeezenet1.0/)|
|[squeezenet1.1](caffe/squeezenet1.1/squeezenet1.1.sh)|caffe|Classification|Top1|58.98/58.76|116.28|81.75|42.14|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/squeezenet1.1/)|
|[mnist](onnx/mnist/mnist.sh)|onnx|Classification|||4533.87|4746.26|4251.33|[More Info](https://github.com/onnx/models/tree/main/vision/classification/mnist)|