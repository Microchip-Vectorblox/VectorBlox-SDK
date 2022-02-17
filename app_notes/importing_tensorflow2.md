# Importing TensorFlow 2 Networks

## Motivation 

TensorFlow is one of the most popular frameworks for creating neural networks. With the release of a new major version in 2019, many changes were introduced. This application note explains how to take a TensorFlow 2 network (in SavedModel format or Keras format) and pass it through the VectorBlox SDK to generate an embedded model.

Please note that while this application note explains how to convert models via our SDK  **we provide TFHub-based tutorials that run these TF2 networks through our flow**.

This application note uses the following examples:

- [MobileNet v2 140 SavedModel](https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/5), which classifies objects from the 1000-category [ImageNet Challenge](https://www.image-net.org/challenges/LSVRC/) dataset.

- [ResNet 50 Keras](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50), which classifies objects from the 1000-category [ImageNet Challenge](https://www.image-net.org/challenges/LSVRC/) dataset.


## Step 1: Download model

First download a TensorFlow 2 model. You can grab a variety of models from [TFHub](https://tfhub.dev/) or use the `keras.applications` functions.

> MobileNetV2 140 (SavedModel)
```
wget https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/5?tf-hub-format=compressed -O mobilenet_v2_140_224.tar.gz
mkdir -p mobilenet_v2_140_224
tar -xzf mobilenet_v2_140_224.tar.gz -C mobilenet_v2_140_224
```

> Resnet50 (Keras)
```
import tensorflow as tf
model = tf.keras.applications.ResNet50(weights="imagenet")
model.save('resnet50.keras')
```

## Step 2: (Optional) Convert Keras model to SavedModel format

Users can quickly swap between SavedModel and Kera formats. For the remainder of this application note we assume the model is converted to SavedModel format.

> Resnet50 (Keras)
```
model = tf.keras.models.load_model('resnet50.keras')
model.save('resnet50')
```

## Step 3: Ensure default signature set for SavedModel

Signatures are used in OpenVINO's Model Optimizer, which is used as the entry point of the VectorBlox SDK. They are used to determine the networks inputs and outputs.  We can check if the default signature is set with the following python snippet:

```
model = tf.saved_model.load(DIRECTORY)
print(model.signatures.keys())
```

You should see `serving_default` in the list.  If the `serving_default` signature doesn't exist, we can add it to our model and resave:

```
model = tf.saved_model.load(DIRECTORY)
call = model.__call__.get_concrete_function(tf.TensorSpec(None, tf.float32))
tf.saved_model.save(model, DIRECTORY, signatures=call)
```

## Step 4: Inspect model to gather input and output names

With the default signature set, we use can use `tensorboard` to inspect our model and gather the input and output names that are needed when converting the model. 

```
mo --tensorboard_logdir DIRECTORY --saved_model_dir DIRECTORY
```

```
tensorboard serve --logdir DIRECTORY
```

## Step 5: Pass these parameters through the VectorBlox SDK

With the default signature set and I/O known, we can convert the TF2 model to OpenVINO format, and then generate our embedded network.

> Note: the user will have to determine mean and scale values based on required preprocessing, along w/ specifying the input shape (including batch = 1) 

> MobilenetV2
```
mo --saved_model_dir mobilenet_v2_140_224 \
   --input=inputs \
   --output=Identity \
   --input_shape [1,224,224,3] \
   --mean_values [127.5,127.5,127.5] \
   --scale_values [127.5]

generate_vnnx -x saved_model.xml  -c V1000 -f ../sample_images -o mobilenet_v2_140_224.vnnx
```

> ResNet 50
```
mo --saved_model_dir resnet50 \
   --input=input_1 \
   --output=Identity \
   --input_shape [1,224,224,3] \
   --mean_values [127.5,127.5,127.5] \
   --scale_values [127.5]

generate_vnnx -x saved_model.xml  -c V1000 -f ../sample_images -o resnet_50.vnnx
```
