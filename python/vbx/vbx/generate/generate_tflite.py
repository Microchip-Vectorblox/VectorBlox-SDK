import os
import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
import argparse
import cv2
import tqdm
import subprocess
import shlex
import sys
from tensorflow.lite.python.convert import mlir_quantize
from tensorflow.python.framework import dtypes


def representative_dataset_numpy(numpy_file_path, mean=0., scale=1.):
    def representative_dataset():
        calib_data = np.load(numpy_file_path)
        for data in calib_data:
            normalized_calib_data = (data - mean) / scale
            normalized_calib_data = np.expand_dims(normalized_calib_data, axis=0)
            yield [normalized_calib_data.astype(np.float32)]
    return representative_dataset

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.compat.v2.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and return it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def analyze_inputs_outputs(graph):
    ops = graph.get_operations()
    outputs_set = set(ops)
    inputs = []
    for op in ops:
        if len(op.inputs) == 0 and op.type != 'Const':
            inputs.append(op)
        else:
            for input_tensor in op.inputs:
                if input_tensor.op in outputs_set:
                    outputs_set.remove(input_tensor.op)
    outputs = list(outputs_set)
    inputs = [i.name.replace("prefix/", "") for i in inputs]
    outputs = [o.name.replace("prefix/", "") for o in outputs]
    return inputs, outputs

def convert_tflite_model(converter):
    calibrated_tflite_model = converter.convert()
    quantized_tflite_model = mlir_quantize(
        calibrated_tflite_model, fully_quantize=True,
        input_data_type=dtypes.int8,
        output_data_type=dtypes.int8)
    return quantized_tflite_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('tflite')
    parser.add_argument('-d', '--data')
    parser.add_argument('-c', '--count', type=int, default=128)
    parser.add_argument('-s', '--shape', type=int, nargs='+')
    parser.add_argument('-u', '--unsigned', action='store_true')
    parser.add_argument('-m', '--mean', type=float, nargs='+', default=0.)
    parser.add_argument('-sc', '--scale', type=float, nargs='+', default=1.)
    parser.add_argument('--float32', action='store_true')
    parser.add_argument('-no', '--no_optimizations', action='store_true', help='Skip optimizations')
    args = parser.parse_args()

    if os.path.isdir(args.model): # saved_model dir
        model = tf.saved_model.load(args.model)
        if args.shape: # TODO assuming single input
            concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
            concrete_func.inputs[0].set_shape(args.shape)
            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        else:
            converter = tf.lite.TFLiteConverter.from_saved_model(args.model, signature_keys=model.signatures.keys())
    elif '.h5' in args.model: # keras H5
        loaded = tf.keras.models.load_model(args.model)

        if args.shape: # TODO assuming single input
            if type(loaded.input) == list:
                input = [i.name for i in loaded.input]
            else:
                input = [loaded.input.name]
            converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(args.model, input_shapes={input[0]:args.shape})
        else:
            converter = tf.lite.TFLiteConverter.from_keras_model(loaded)
    elif '.pb' in args.model: # keras H5
        graph = load_graph(args.model)
        input, output = analyze_inputs_outputs(graph)
        if args.shape: # TODO assuming single input
            converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(args.model, input, output, input_shapes={input[0]:args.shape})
        else:
            converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(args.model, input, output)

    if not args.no_optimizations:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.representative_dataset = representative_dataset_numpy(
        numpy_file_path=args.data,
        mean=args.mean,
        scale=args.scale
    )

    if not args.float32:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        if args.unsigned:
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
        else:
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
    quantized_tflite_model = convert_tflite_model(converter)
    with open(args.tflite, 'wb') as f:
        f.write(quantized_tflite_model)

if __name__ == "__main__":
    main()
