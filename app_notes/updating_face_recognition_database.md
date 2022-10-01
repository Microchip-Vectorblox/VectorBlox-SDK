# Updating the Face Recognition demo

This document describes the Python code used to create the face recognition demo, including scripts to create a new face database. The code is located in the folder below.

```
$VBX_SDK/example/python/faceDemo
```

Before using code in this folder, make sure the Python virtual environment `vbx_env` is activated. See `README.md` in the SDK root directory for setup instructions.

## Preparing models

The Python scripts in this folder require one model for face detection and another model for face recognition. The tutorials for these models should be run so that the files are available.

```
cd $VBX_SDK/tutorials/onnx/scrfd_500m_bnkps
bash scrfd_500m_bnkps.sh
```
```
cd $VBX_SDK/tutorials/mxnet/mobilefacenet-arcface
bash mobilefacenet-arcface.sh
```
```
cd $VBX_SDK/tutorials/onnx/genderage
bash genderage.sh
```

## Creating a new database

Use the script below to create a new database. The database contains pairs of names and face embeddings. In this demo, the database is created offline.

```
cd $VBX_SDK/example/python/faceDemo
python createDb.py
```

The information used to create the database come from the folder `dbImages`. Each sub-folder in `dbImages` corresponds to an entry in the database. The name of the folder will become the name of the database entry. Each subfolder contains one or more images of the same person. Using multiple images per person is recommended to increase accuracy. If multiple images are present, their embeddings are combined to make one database entry.

The created database is saved to `faceDb.npy`, which is a numpy save file containing a Python dictionary. Note that this is not exactly the same as the database used in the current demo.

## Processing an image

Use the script below to process an image with the created database.

```
python processImage.py
```

This will process the image `garden.jpg` using the database `faceDb.npy`.  
Processing includes:
1. Load the image from file  
2. Resize the image if necessary  
3. Run the detection model  
4. If detected, align and crop faces  
5. Run the recognition model on each face  
6. Compare the recognition output to the database to identify faces  
7. Draw bounding boxes and names on the image  
8. Save the output image to file

## Exporting the database

Use the script below to export the Python database.

```
python exportDb.py
```

This creates `faceDb.c` containing C-code representing the embeddings. This code can be integrated into your project. For the VideoKit project, embeddings are found at the top of `faceDetection/faceDetectDemo.c`. Replace the default embeddings with the embeddings in `faceDb.c` to recognize your new faces.
