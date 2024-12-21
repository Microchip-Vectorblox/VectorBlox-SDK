# Face Recognition Demo Summary

## Description

This demo demonstrates a face recognition system. The demo includes two deep learning models that are run in series: a face detection model and a face recognition model. It also includes software to link the models and track results over time.

![](images/face_demo_functional_diagram.png)

Image data comes from either an HDMI or camera video input. The 1920x1080 input is resized to 512x288 pixels. This smaller image is run through the RetinaFace face detection model on the VectorBlox accelerator. The raw output is post-processed on the soft processor to generate a bounding box and facial feature points (eyes, nose, mouth) for each detection.

The tracker software takes the detections and tracks them over time using Kalman filters. Each frame, the tracker can choose one detected face to run through a recognition model. If there are multiple faces in a scene, the tracker will rotate through the faces, detecting one per frame. This allows a consistent high frame rate while maintaining identification for multiple faces.

The recognition model takes a face image and outputs an embedding vector. The recognition model requires that the face in the input image to be a specific size and orientation. The tracker converts a set of facial keypoints into a transformation matrix. This matrix is given to an accelerator that will crop, resize, and rotate the input image. It is required that this input frame be the same frame that was run through the detection model. The transformed face image is run through the ArcFace face recognition model on the VectorBlox accelerator. The output face embedding is a 128-element vector. In software, this embedding is compared to a database of embeddings. If a match in the database is found, a name or identification is sent back to the tracker.

Finally, the tracker sends bounding boxes and identifying names to the output display. Bounding boxes and face points are drawn for each detected face. If a face is recognized, the drawing is done in green and the name is drawn below the box. Otherwise, the box and face points are drawn in blue.

The database of known faces and names contains 4 entries labeled `John`, `Nancy`, `Tina`, and `Bob`. The database embeddings were created from a different video.

The demo uses the following models:  
- [SCRFD](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/onnx/scrfd_500m_bnkps/scrfd_500m_bnkps.sh), which detects faces.  
- [ArcFace Mobile](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/mxnet/mobilefacenet-arcface/mobilefacenet-arcface.sh), which recognizes a face.  

<!-- Update once updated tutorials are released, tested, and measured
## Performance

| Core Vectorblox Configuration | V1000 | V500 | V250 |
| --- | --- | --- | --- |
|FPGA Total LUT/FF/DSP/uSRAM/LSRAM | 160K/153K/353/1061/507 | 145K/132K/225/837/443 | 128K/112K/145/726/406 |
| FPGA CoreVectorblox LUT/FF/DSP/uSRAM/LSRAM | 60K/69K/292/531/148 | 45K/48K/164/307/84 | 26K/28K/84/16/47 |
| Detection Network Runtime | 16 ms | 19 ms | 39 ms |
| Recognition Network Runtime | 11 ms | 14 ms | 29 ms |
| System FPS | 19 FPS (52 ms) | 17 FPS (58 ms) | 12 FPS  (83 ms) |

> Performance was tested on MPF300-VIDEO-KIT-NS MPF300T-1FCG1152E.

<div style="page-break-after: always;"></div>

## Networks

| Model | Task | Input Resolution | MParams | GOPs |
| --- | --- | --- | --- | -- |
| RetinaFace Mobilenet  | Detection | 3x512x288 | 0.423 | 0.717 |
| Arcface Mobile | Recognition | 3x112x112 | 0.993 | 0.448 | -->

