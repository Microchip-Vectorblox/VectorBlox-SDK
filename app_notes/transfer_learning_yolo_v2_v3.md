# Yolo v2/v3 Transfer Learning via DarkNet

## Motivation 

YOLO (You-Only-Look-Once) networks are popular object-detection networks that are capable of finding objects in real-time.
This application note explains how to take a YOLO v2/v3 network, and retrain it on your own labelled data.
This network can then be ran through the VectorBlox SDK and deployed on your Microchip FPGA.


This application note modifies [Tiny YoloV3](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/darknet/yolov3-tiny/yolov3-tiny.sh), which detects objects from the 80-category [Common Objects in Context](https://cocodataset.org/#home) (COCO) dataset, and modifies it into single cateogry face detector, using faces from the [Face Detection Data Set and Benchmark](http://vis-www.cs.umass.edu/fddb/) (FDDB).


## Prerequisites


A GPU w/ CUDA setup is highly recommended, as retraining would take multiple days to run on just a CPU.


## Step 1: Clone and build DarkNet

```
git clone https://github.com/AlexeyAB/darknet.git
cd darknet
```

Edit the `Makefile`, setting `GPU=1` and `CUDNN=1` and build via `make`.


## Step 2: Gather images and generate annotations

For our custom object detection task, we use images and annotations from [FDDB](http://vis-www.cs.umass.edu/fddb/) and modify them for the format DarkNet expects. To download and extract the images and annotations, run the following:

```
cd data
wget http://vis-www.cs.umass.edu/fddb/originalPics.tar.gz
wget http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz
tar xfz originalPics.tar.gz
tar xfz FDDB-folds.tgz
cd ..
```

A list of images to train the network on is required, along with 
a individual `.txt` file for every image. The text file must contain a line for each object found in the image with the following format `<class> <x> <y> <width> <height>`. A utility script is provided to turn FDDB annotations into DarkNet annotations.

```
python face_annotations_darknet.py data fddb
```

If instead you have a collection of images you would like to label, either Microsoft's [Visual Object Tagging Tool](http://github.com/microsoft/VoTT) or [Yolo_mark](https://github.com/AlexeyAB/Yolo_mark) are popular choices for labelling images.


## Step 3: Setup configuration files for transfer learning

The following files are needed for training DarkNet networks:
- a network config file (`.cfg`)
- an optional pretrained weights file (`.weights`)
- a file containing a list of images to train the network on (`faces.txt`), generated in the previous step
- a file containing class names (`obj.names`)
- a summary file which points to these important files (`obj.data`)

By modifying the Yolo v3 Tiny `.cfg` and using [preloaded weights](https://drive.google.com/file/d/18v36esoXCh-PsOKwyP2GWrpYDptDY8Zf/view?usp=sharing), users can quickly generate a custom neural network. 
We need to adjust the `classes` and the previous layer's `filters`. For a single category network, `classes = 1`, and `filters = 18` (matching the formula `(5+classes)*num`).
Download these files and modify the config as follows:

```diff
@@ -124,7 +124,7 @@ activation=leaky
 size=1
 stride=1
 pad=1
-filters=255
+filters=18
 activation=linear
 
 
@@ -132,7 +132,7 @@ activation=linear
 [yolo]
 mask = 3,4,5
 anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
-classes=80
+classes=1
 num=6
 jitter=.3
 ignore_thresh = .7
@@ -168,13 +168,13 @@ activation=leaky
 size=1
 stride=1
 pad=1
-filters=255
+filters=18
 activation=linear
 
 [yolo]
 mask = 0,1,2
 anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
-classes=80
+classes=1
 num=6
 jitter=.3
 ignore_thresh = .7
```



obj.names
```
face
```

obj.data
```
classes = 1
train  = data/fddb.txt
names = data/obj.names
backup = backup/
```

## Step 5: Launch transfer learning

Training is launched using the `detector train` subcommand. The data file, network config and pretrained weights are added as arguments.

```
./darknet detector train data/obj.data cfg/faces.cfg data/yolov3.conv.11
```

## Step 6: (Optional) Gather more data and repeat Steps 2-5

If you are not satisfied with the performance of your network, please repeat steps 2-5 w/ more data, or perhaps target a different configuration.

## Step 7: Run your custom network through the VectorBlox SDK

The custom network can now be ran through the SDK with the following commands. Please consult our [Running Custom Yolo V2/V3 application note](./running_custom_yolo_v2_v3.md) for more details.
