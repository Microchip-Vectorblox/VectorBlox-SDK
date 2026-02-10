# Inference Simulation

Refer to the sdk_toolkit.md and vbx_graph_generation.md documents for instructions to create a binary file that is ready to run on the functionally accurate VBX simulator. Example scripts which call the simulator and run post-processing are provided for classification and object detection tasks. For more details on simulator use and post-processing that these scripts leverage, and are found in the subsequent sections of this guide.

## C Simulation
The following script runs for the C version of the simulator, and prints out the checksum of the model based on either a sample image (.JPG) or the default test_data if a sample image is not passed as an argument.
```
(vbx_env) ~/sdk$ $VBX_SDK/example/sim-c/sim-run-model MODEL_NAME.vnnx IMAGE.jpg POSTPROCESSTYPE
```

## Python Simulation
The following scripts run the indicated models with the Python version of the simulator and print the inference results. They also generate an annotated output image.

Arguments passed for the following Python commands are found/explained in their corresponding Python files.

This is a list of a few examples, for exact details on which python commands to run for specific models, refer to the tutorials [here](../tutorials)

• Run a classifier model on the simulator.
```
(vbx_env) ~/sdk$ python $VBX_SDK/example/python/classifier.py MODEL_NAME.vnnx IMAGE.jpg
```
• Run a YOLO model (for example, YOLOv5n) on the simulator.
```
(vbx_env) ~/sdk$ python $VBX_SDK/example/python/yoloInfer.py yolov5n_V1000_ncomp.vnnx IMAGE.jpg -j yolov5n.json -v 5 -l coco.names -t 0.25 
```
• Run a YOLOv8 model (for example, yolov8n) on the simulator.
```
(vbx_env) ~/sdk$ python $VBX_SDK/example/python/yoloInfer.py yolov8n_V1000_ncomp.vnnx IMAGE.jpg --v 8 -l coco.names 
```
• Run a pose detection model (for example, posenet) on the simulator.
```
(vbx_env) ~/sdk$ python $VBX_SDK/example/python/posenetInfer.py posenet_V1000_ncomp.vnnx -i IMAGE.jpg 
```
