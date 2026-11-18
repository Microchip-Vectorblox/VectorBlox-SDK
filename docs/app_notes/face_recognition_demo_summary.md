# Face Recognition Demo Summary

## Description

This demo showcases a face recognition system. The demo includes two deep learning models that are run in series: a face detection model and a face recognition model. It also includes software to link the models and track results over time.

![](../images/face_demo_functional_diagram.png)

>Note: The diagram shows SCRFD and ArcFace. The demo instead uses YuNet for facial detection and SFace for facial recognition.

Image data comes from either an HDMI or camera video input. The 1920x1080 input is resized to 512x288 pixels. This smaller image is run through the YuNet face detection model on the VectorBlox accelerator. The raw output is post-processed on the soft processor to generate a bounding box and facial feature points (eyes, nose, mouth) for each detection.

The tracker software takes the detections and tracks them over time using Kalman filters. Each frame, the tracker can choose one detected face to run through a recognition model. If there are multiple faces in a scene, the tracker will rotate through the faces, recognizing one per frame. This allows a consistent high frame rate while maintaining identification for multiple faces.

The recognition model takes a face image and outputs an embedding vector. The recognition model requires the face in the input image to be a specific size and orientation. The tracker converts a set of facial keypoints into a transformation matrix. This matrix is given to an accelerator that will crop, resize, and rotate the input image. It is required that this input frame be the same frame that was run through the detection model. The transformed face image is run through the SFace face recognition model on the VectorBlox accelerator. The output face embedding is a 128-element vector. In software, this embedding is compared to a database of embeddings. If a match in the database is found, a name or identification is sent back to the tracker.

Finally, the tracker sends bounding boxes and identifying names to the output display. Bounding boxes and face points are drawn for each detected face. If a face is recognized, the drawing is done in green and the name is drawn below the box. Otherwise, the box and face points are drawn in blue.

The database of known faces and names contains 4 entries labeled `John`, `Nancy`, `Tina`, and `Bob`. The database embeddings were created from a different video.

The demo uses the following models:  
- [YuNet](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/onnx/face_detection_yunet_2023mar/face_detection_yunet_2023mar.sh), which detects faces.  
- [SFace](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/onnx/face_recognition_sface_2021dec/face_recognition_sface_2021dec.sh), which recognizes a face.  
