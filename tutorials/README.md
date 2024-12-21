
# Tutorials

Below is a list of included tutorials. Frames per Second (FPS) assumes the cores are running at 125 MHz

 <div class="acc_vnnx"> 

| Tutorial Name | Source | Task  |Accuracy Metric|Accuracy Tflite |Accuracy VNNX |V1000 FPS   | More information |
| ------------- |-----------------|-------|---------------|--------------|------------|-----------|----|
|[onnx_resnet18-v1](onnx/onnx_resnet18-v1/onnx_resnet18-v1.sh)|onnx|ImageNet|Top1|68.1|68.2|32.59|[More Info](https://github.com/onnx/models/tree/main/validated/vision/classification/resnet)|
|[onnx_resnet34-v1](onnx/onnx_resnet34-v1/onnx_resnet34-v1.sh)|onnx|ImageNet|Top1|72.34|72.06|17.79|[More Info](https://github.com/onnx/models/tree/main/validated/vision/classification/resnet)|
|[onnx_squeezenet1.1](onnx/onnx_squeezenet1.1/onnx_squeezenet1.1.sh)|onnx|ImageNet|Top1|54.1|54.04|85.76|[More Info](https://github.com/onnx/models/tree/main/validated/vision/classification/squeezenet)|
|[scrfd_500m_bnkps](onnx/scrfd_500m_bnkps/scrfd_500m_bnkps.sh)|onnx|Face Detection||||86.66|[More Info](https://insightface.ai/scrfd)|
|[mobilenet-v1-0.25-128](openvino/mobilenet-v1-0.25-128/mobilenet-v1-0.25-128.sh)|openvino|ImageNet|Top1 1001|36.52|36.44|429.18|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/mobilenet-v1-0.25-128/)|
|[mobilenet-v1-1.0-224](openvino/mobilenet-v1-1.0-224/mobilenet-v1-1.0-224.sh)|openvino|ImageNet|Top1|70.42|70.34|79.94|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/mobilenet-v1-1.0-224)|
|[mobilenet-v1-1.0-224-tf](openvino/mobilenet-v1-1.0-224-tf/mobilenet-v1-1.0-224-tf.sh)|openvino|ImageNet|Top1 1001|69.34|69.68|80.06|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/mobilenet-v1-1.0-224-tf/)|
|[mobilenet-v2](openvino/mobilenet-v2/mobilenet-v2.sh)|openvino|ImageNet|Top1|68.9|68.86|71.43|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/mobilenet-v2)|
|[mobilenet-v2-1.0-224](openvino/mobilenet-v2-1.0-224/mobilenet-v2-1.0-224.sh)|openvino|ImageNet|Top1 1001|70.38|70.62|72.05|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/mobilenet-v2-1.0-224)|
|[mobilenet-v2-1.4-224](openvino/mobilenet-v2-1.4-224/mobilenet-v2-1.4-224.sh)|openvino|ImageNet|Top1 1001|74.26|74.28|46.3|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/mobilenet-v2-1.4-224/)|
|[squeezenet1.0](openvino/squeezenet1.0/squeezenet1.0.sh)|openvino|ImageNet|Top1|55.36|55.44|63.01|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/squeezenet1.0/)|
|[mobilefacenet-arcface](openvino/mobilefacenet-arcface/mobilefacenet-arcface.sh)|openvino|face_compare||||66.18|[More Info](https://github.com/deepinsight/insightface)|
|[squeezenet1.1](openvino/squeezenet1.1/squeezenet1.1.sh)|openvino|ImageNet|Top1|56.66|56.62|125.94|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/squeezenet1.1/)|
|[mobilenet_v2_140_224](tensorflow/mobilenet_v2_140_224/mobilenet_v2_140_224.sh)|tensorflow|ImageNet|Top1 1001|74.38|74.36|25.77|[More Info](https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/5)|
|[torchvision_googlenet](pytorch/torchvision_googlenet/torchvision_googlenet.sh)|pytorch|ImageNet|Top1|62.42|62.16|30.98|[More Info](https://pytorch.org/vision/0.14/models/googlenet.html)|
|[torchvision_inception_v3](pytorch/torchvision_inception_v3/torchvision_inception_v3.sh)|pytorch|ImageNet|Top1|77.8|76.97|7.02|[More Info](https://pytorch.org/vision/0.14/models/inception.html)|
|[torchvision_resnet18](pytorch/torchvision_resnet18/torchvision_resnet18.sh)|pytorch|ImageNet|Top1|43.66|43.58|32.39|[More Info](https://pytorch.org/vision/0.14/models/generated/torchvision.models.resnet18.html#torchvision.models.resnet18)|
|[torchvision_resnet50](pytorch/torchvision_resnet50/torchvision_resnet50.sh)|pytorch|ImageNet|Top1|80.32|80.42|11.41|[More Info](https://pytorch.org/vision/0.14/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50)|
|[torchvision_wide_resnet50_2](pytorch/torchvision_wide_resnet50_2/torchvision_wide_resnet50_2.sh)|pytorch|ImageNet|Top1|81.2|81.24|5.01|[More Info](https://pytorch.org/vision/0.9/models.html#torchvision.models.wide_resnet50_2)|
|[torchvision_ssdlite320_mobilenet_v3_large](pytorch/torchvision_ssdlite320_mobilenet_v3_large/torchvision_ssdlite320_mobilenet_v3_large.sh)|pytorch|ImageNet|mAP(COCO)|||23.51|[More Info](https://pytorch.org/vision/0.14/models/ssdlite.html)|
|[mobilenet_v1_050_160](tensorflow/mobilenet_v1_050_160/mobilenet_v1_050_160.sh)|tensorflow|ImageNet|Top1 1001|49.2|49.42|218.34|[More Info](https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/classification/5)|
|[mobilenet_v2](tensorflow/mobilenet_v2/mobilenet_v2.sh)|tensorflow|ImageNet|Top1|70.38|70.4|58.89|[More Info](https://keras.io/api/applications/mobilenet/)|
|[mobilenet_v2_140_224](tensorflow/mobilenet_v2_140_224/mobilenet_v2_140_224.sh)|tensorflow|ImageNet|Top1 1001|74.38|74.36|25.77|[More Info](https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/5)|
|[yolo-v3-tf](tensorflow/yolo-v3-tf/yolo-v3-tf.sh)|tensorflow|Object Detection|mAP(COCO)|58.05|57.91|2.02|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/yolo-v3-tf/)|
|[yolo-v3-tiny-tf](tensorflow/yolo-v3-tiny-tf/yolo-v3-tiny-tf.sh)|tensorflow|Object Detection|mAP(COCO)|34.6|34.63|25.77|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/yolo-v3-tiny-tf/)|
|[yolo-v4-tiny-tf](tensorflow/yolo-v4-tiny-tf/yolo-v4-tiny-tf.sh)|tensorflow|Object Detection|mAP(COCO)|38.46|38.59|19.28|[More Info](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/yolo-v4-tiny-tf/)|
|[efficientnet-lite0-int8](tensorflow/efficientnet-lite0-int8/efficientnet-lite0-int8.sh)|tensorflow|ImageNet|Top1|70.84|70.7|61.77|[More Info](https://www.kaggle.com/models/google/efficientnet/tensorFlow1/b0-classification/1)|
|[efficientnet-lite1-int8](tensorflow/efficientnet-lite1-int8/efficientnet-lite1-int8.sh)|tensorflow|ImageNet|Top1|70.72|70.61|41.58|[More Info](https://www.kaggle.com/models/google/efficientnet/tensorFlow1/b1-classification/1)|
|[efficientnet-lite2-int8](tensorflow/efficientnet-lite2-int8/efficientnet-lite2-int8.sh)|tensorflow|ImageNet|Top1|73.48|73.5|31.49|[More Info](https://www.kaggle.com/models/google/efficientnet/tensorFlow1/b2-classification/1)|
|[efficientnet-lite3-int8](tensorflow/efficientnet-lite3-int8/efficientnet-lite3-int8.sh)|tensorflow|ImageNet|Top1|77.04|76.86|22.09|[More Info](https://www.kaggle.com/models/google/efficientnet/tensorFlow1/b3-classification/1)|
|[efficientnet-lite4-int8](tensorflow/efficientnet-lite4-int8/efficientnet-lite4-int8.sh)|tensorflow|ImageNet|Top1|78.2|78.32|13.64|[More Info](https://www.kaggle.com/models/google/efficientnet/tensorFlow1/b4-classification/1)|
|[yolov5m](ultralytics/yolov5m/yolov5m.sh)|ultralytics|Object Detection|mAP(COCO)|56.43|56.54|2.42|[More Info](https://github.com/ultralytics/yolov5)|
|[yolov5m.relu](ultralytics/yolov5m.relu/yolov5m.relu.sh)|ultralytics|Object Detection|mAP(COCO)|54.59|54.51|6.41|[More Info](https://github.com/ultralytics/yolov5)|
|[yolov5n](ultralytics/yolov5n/yolov5n.sh)|ultralytics|Object Detection|mAP(COCO)|36.51|36.17|18.97|[More Info](https://github.com/ultralytics/yolov5)|
|[yolov5n.relu](ultralytics/yolov5n.relu/yolov5n.relu.sh)|ultralytics|Object Detection|mAP(COCO)|32.29|32.26|46.75|[More Info](https://github.com/ultralytics/yolov5)|
|[yolov5n6u](ultralytics/yolov5n6u/yolov5n6u.sh)|ultralytics|ImageNet|mAP(COCO)|50.51|49.28|1.8|[More Info](https://github.com/ultralytics/ultralytics/)|
|[yolov5nu](ultralytics/yolov5nu/yolov5nu.sh)|ultralytics|ImageNet|mAP(COCO)|42.19|38.69|7.77|[More Info](https://github.com/ultralytics/ultralytics/)|
|[yolov5s](ultralytics/yolov5s/yolov5s.sh)|ultralytics|Object Detection|mAP(COCO)|46.86|46.66|6.53|[More Info](https://github.com/ultralytics/yolov5)|
|[yolov5s.relu](ultralytics/yolov5s.relu/yolov5s.relu.sh)|ultralytics|Object Detection|mAP(COCO)|48.76|48.65|16.65|[More Info](https://github.com/ultralytics/yolov5)|
|[yolov8n](ultralytics/yolov8n/yolov8n.sh)|ultralytics|ImageNet|mAP(COCO)|44.47|42.53|7.73|[More Info](https://github.com/ultralytics/ultralytics/)|
|[yolov8n_cut](ultralytics/yolov8n_cut/yolov8n_cut.sh)|ultralytics|ImageNet|mAP(COCO)|44.47|42.53|13.95|[More Info](https://github.com/ultralytics/ultralytics/)|
|[yolov8n_512x288_cut](ultralytics/yolov8n_512x288_cut/yolov8n_512x288_cut.sh)|ultralytics|ImageNet|mAP(COCO)|||36.9|[More Info](https://github.com/ultralytics/ultralytics/)|
|[yolov8n-cls](ultralytics/yolov8n-cls/yolov8n-cls.sh)|ultralytics|ImageNet|Top1|61.78|61.48|165.02|[More Info](https://github.com/ultralytics/ultralytics/)|
|[yolov8s](ultralytics/yolov8s/yolov8s.sh)|ultralytics|ImageNet|mAP(COCO)|53.53|51.71|3.57|[More Info](https://github.com/ultralytics/ultralytics/)|
|[yolov8s_cut](ultralytics/yolov8s_cut/yolov8s_cut.sh)|ultralytics|ImageNet|mAP(COCO)|53.53|51.71|4.52|[More Info](https://github.com/ultralytics/ultralytics/)|
|[yolov9s](ultralytics/yolov9s/yolov9s.sh)|ultralytics|ImageNet|mAP(COCO)|54.94|53.98|3.58|[More Info](https://github.com/ultralytics/ultralytics/)|
|[yolov9s_cut](ultralytics/yolov9s_cut/yolov9s_cut.sh)|ultralytics|ImageNet|mAP(COCO)|54.94|53.98|4.51|[More Info](https://github.com/ultralytics/ultralytics/)|
|[yolov9t](ultralytics/yolov9t/yolov9t.sh)|ultralytics|ImageNet|mAP(COCO)|46.75|44.64|6.8|[More Info](https://github.com/ultralytics/ultralytics/)|
|[yolov9t_cut](ultralytics/yolov9t_cut/yolov9t_cut.sh)|ultralytics|ImageNet|mAP(COCO)|46.75|44.64|11.2|[More Info](https://github.com/ultralytics/ultralytics/)|
  
  
</div>


