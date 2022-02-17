# Running custom YOLO v2/v3 Networks

## Motivation 

YOLO (You-Only-Look-Once) networks are popular object-detection networks
that largely replace the compute-intensive R-CNNs (recursive convolutional neural networks).
With R-CNNs you had to run many passes of a classifer to detect objects,
while with YOLO networks, a single pass produces multiple bounding boxes (w/ prediction strength and category).


This application note explains how to take a YOLO v2/v3 network, pass it through the VectorBlox SDK to generate an embedded model and run post-processing.
Key parameters concerning the network’s inputs + outputs must be passed to both the VectorBlox SDK and post-processing.


Please note that while this application note explains these key parameters and how to gather them manually, **we provide a [utility script](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/darknet/darknet_to_onnx.py) that does this _automatically_** for Darknet YOLO networks. It converts the model from Darknet to ONNX and gathers all the parameters needed. This script can be seen in action in our provided YOLO tutorials: `python $VBX_SDK/tutorials/darknet/darknet_to_onnx.py yolo.cfg`.


This application note uses the following examples:

- [YOLO v2 Tiny VOC](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/darknet/yolov2-tiny-voc/yolov2-tiny-voc.sh), which detects objects from the 20-category [Pascal Visual Object Classes](http://host.robots.ox.ac.uk/pascal/VOC/) (VOC) dataset.

- [YOLO v3 Tiny COCO](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/darknet/yolov3-tiny/yolov3-tiny.sh), which detects objects from the 80-category [Common Objects in Context](https://cocodataset.org/#home) (COCO) dataset.


## Prerequisites

Please ensure you have the SDK installed, as well as [Netron](https://netron.app/) to explore the model.

The Vectorblox SDK consumes neural networks from most [frameworks, listed here](https://docs.openvinotoolkit.org/2021.1/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).
 If the framework is not directly supported, the user will have to convert the model to a supported framework. 


## Step 1: Understanding key parameters in your YOLO network  


### Parameter Overview

By inspecting our models in [Netron](https://netron.app/), we can gather the information needed.

> [Click here](https://netron.app/?url=https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2-tiny-voc.cfg) to inspect YOLO v2 Tiny VOC.

Each output layer (`region` layers for `v2` or `yolo` layers for `v3`) contains many parameters.
The following are important parameters you need to know:

- *Input layer shapes*: the dimensions of inputs
`(channels, rows, columns) or (c,h,w)`

- *Output layer names*: must provide output layers to the SDK, to cut them from the graph

- *Output layer shapes*: the dimensions of each output map
`(channels, rows, columns) or (c,h,w)`

- *Coords*: # of coordinates produced for each bounding box
(default is 4, corresponding to xmin, xmax, ymin, ymax) 

- *Classes*: # of classes, or object-types your model can detect
(COCO uses 80 classes, VOC uses 20) 

- *Num*: number of outputs (boxes) that can be detected for each cell in the `h,w` output map. 
(recorded as the total per network, not per layer)

- *Anchors*: height and width offsets to the bounding box coordinates.
(if `Num = N`, there will be `2N` anchors)

- *Masks*: specifies which anchor pairs are used for each set of outputs
(used when networks have more than 1 output)


![Gathering parameters via Netron for YOLO v2 Tiny VOC (yolov2-tiny-voc.cfg)](images/yolo_v2_tiny_voc_region.png)


### Version Differences

Some important _gotchas_ where parameters vary across different YOLO versions

- YOLO output layers in Darknet are `region` layers for `v2` or `yolo` layers for `v3`
- For YOLO `v2` anchors are not multipled by the ratio of input maps size to output maps size `(416 / 13 == 32)`,
  but for `v3` they already are.
- `Num` represents the total number of outputs sets across all output layers, not the number of output sets for an individual output layers
 (multiple outputs, refer to the number of entries in the `mask` field to see how many output sets for that layer)
- Like `Num`, all `anchors` for all layers are listed in each layer. The `mask` field to is used to select the correct anchors in post-processing.  


## Step 2: (Optional) Verify parameters match your model

To verify you have the correct values, you can check the number of output channels (or filters) of the convolution layer prior to each output layer matches the following formula: `(1+Coords+Classes)*Num = Output Channels`

> YOLO v2 Tiny VOC
```
Coords: 4
Classes: 20
Num: 5
(1+4+20)*5 = 125 Output Channels
```


> YOLO v3 Tiny COCO 
```
Coords: 4
Classes: 80
Num: 6 (3 per layer)
(1+4+80)*3 = 255 Output Channels (per layer)
```

## Step 3: Providing outputs to the SDK to generate the embedded model 

The Vectorblox SDK uses OpenVINO to produce an inference-optimized model, that is then translated to an embedded representation (`.vnnx`). We cut the network outputs immediately before the YOLO layers. If you are coming from Darknet, and used our utility script, the generated `ONNX` network has cut off the `YOLO` layers.


      
## Step 4: Providing parameters to YOLO post-processing  

As part of the SDK, we provide sample post-processing in Python and C. The `YOLO` layer parameters must be passed to post-processing. For Python, the codes consumes a `JSON` object, while our C code requires the information to be provided as arguments to post-processing function. 


### Python

If you are coming from Darknet, and used our utility script, you can use the file generated for you.  Otherwise you must create a `JSON` file, that contains a list of objects for each output, with the following fields:`"type",  "outputName", "anchors", "classes", "coords", "num", "c", "h", "w"`.


> YOLO v2 Tiny VOC
```.json
[
    {
        "type": "region",
        "outputName": "Y14",
        "anchors": "1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52",
        "classes": 20,
        "coords": 4,
        "num": 5,
        "c": 125,
        "h": 13,
        "w": 13
    }

]
```

> YOLO v3 Tiny COCO
```.json
[
    {
        "type": "yolo",
        "outputName": "Y15",
        "mask": "3, 4, 5",
        "anchors": "10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319",
        "classes": 80,
        "num": 6,
        "c": 255,
        "h": 13,
        "w": 13
    },
    {
        "type": "yolo",
        "outputName": "Y22",
        "mask": "0, 1, 2",
        "anchors": "10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319",
        "classes": 80,
        "num": 6,
        "c": 255,
        "h": 26,
        "w": 26
    }
]
```

### C

Similar to the code above, we provide the information in the form of arguments to `postprocessing_yolo`. Most information is encoded in `yolo_info_t` structs (one for each output).

> YOLO v2 Tiny VOC
```.c

  int num_outputs = 1;
  float anchors[] ={1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.620001, 10.52};

  yolo_info_t cfg_0 = {
	  .version = 2,
	  .input_dims = {3, 416, 416},
	  .output_dims = {125, 13, 13},
	  .coords = 4,
	  .classes = 20,
	  .num = 5,
	  .anchors_length = 10,
	  .anchors = anchors,
  };

  yolo_info_t cfg[] = {cfg_0};
  fix16_t *all_outputs[] = {outputs};
  *detections = post_process_yolo(all_outputs, num_outputs, cfg, 0.3, 0.4, fix16_boxes, max_boxes);

```

> YOLO v3 Tiny COCO
```.c
  int num_outputs = 2;
  float anchors[] = {10,14,23,27,37,58,81,82,135,169,344,319}; // 2*num
  int mask_0[] = {3,4,5};
  int mask_1[] = {1,2,3};

  yolo_info_t cfg_0 = {
	  .version = 3,
	  .input_dims = {3, 416, 416},
	  .output_dims = {255, 13, 13},
	  .coords = 4,
	  .classes = 80,
	  .num = 6,
	  .anchors_length = 12,
	  .anchors = anchors,
	  .mask_length = 3,
	  .mask = mask_0,
  };

  yolo_info_t cfg_1 = {
	  .version = 3,
	  .input_dims = {3, 416, 416},
	  .output_dims = {255, 26, 26},
	  .coords = 4,
	  .classes = 80,
	  .num = 6,
	  .anchors_length = 12,
	  .anchors = anchors,
	  .mask_length = 3,
	  .mask = mask_1,
  };

  yolo_info_t cfg[] = {cfg_0, cfg_1};

  fix16_t *all_outputs[] = {outputs0, outputs1};
  *detections = post_process_yolo(all_outputs, num_outputs, cfg, 0.3, 0.4, fix16_boxes, max_boxes);
```
