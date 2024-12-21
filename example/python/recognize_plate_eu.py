import numpy as np
import vbx.sim
import argparse
import os
import cv2
import vbx.postprocess.lpr as lpr
import json
from vbx.generate.utils import openvino_infer, openvino_input_shape
from vbx.generate.utils import load_input


def get_vnnx_io_shapes(vnxx):
    with open(vnxx, 'rb') as mf:
        model = vbx.sim.Model(mf.read())
    return model.input_shape[0], model.output_shape

def vnnx_infer(vnnx_model, input_array):
    model = vbx.sim.model.Model(open(vnnx_model,"rb").read())
    
    input_array = input_array.astype(np.float32)
    inputs_resized = (input_array / model.input_scale_factor[0]) + model.input_zeropoint[0]
    flattened = inputs_resized.flatten().astype(model.input_dtypes[0])

    outputs = model.run([flattened])
    outputs[0] = model.output_scale_factor[0] * (outputs[0].astype(np.float32) - model.output_zeropoint[0])
    output = outputs[0].reshape((37,18))

    bw = model.get_bandwidth_per_run()
    print("Bandwidth per run = {} Bytes ({:.3} MB/s at 100MHz)".format(bw,bw/100E6))
    print("Estimated {} seconds at 100MHz".format(model.get_estimated_runtime(100E6)))
    print("If running at another frequency, scale these numbers appropriately")

    return output

    
  



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', default = 'lpr_eu_v3.vnnx')
    parser.add_argument('image', default = '../../test_images/A358CC82.jpg')
    parser.add_argument('-b', '--bgr', action='store_true')
    args = parser.parse_args()


    if not os.path.isfile(args.image):
        print('Error: {} could not be read'.format(args.image))
        os._exit(1)
    img = cv2.imread(args.image)
    if '.tflite' in args.model:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        interpreter= tf.lite.Interpreter(model_path=args.model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = tuple(input_details[0]["shape"][-3:])
        h, w = input_shape[0], input_shape[1]
        img = cv2.imread(args.image)
        if img.shape != input_shape:
            img_resized = cv2.resize(img, (w, h))
        else:
            img_resized = img
        img_resized = np.expand_dims(img_resized, axis=0)
        img_resized = img_resized.astype(np.float32)
        input_scale, input_zero_point = input_details[0].get('quantization', (0.0, 0))
        if  input_scale != 0.0:
            img_resized = (img_resized / input_scale) + input_zero_point
        img_resized = img_resized.astype(input_details[0]['dtype'])
        interpreter.set_tensor(input_details[0]['index'], img_resized)
        interpreter.invoke()
        outputs = []
        for o in range(len(output_details)):
            output_scale, output_zero_point = output_details[o].get('quantization', (0.0, 0))
            output = interpreter.get_tensor(output_details[o]['index'])
            if  output_scale != 0.0:
                # output = output_scale * (output.astype(output_details[o]['dtype']) - output_zero_point)
                output = output_scale * (output.astype(np.float32) - output_zero_point)
            output = output.transpose((0,3,1,2))
            outputs.append(output)
            # outputs.append(output.flatten())
        output = np.squeeze(outputs[0])

    elif '.vnnx' in args.model:
        input_shape , _ = get_vnnx_io_shapes(args.model)
        input_array = load_input(args.image, 1., input_shape, (not args.bgr)) #automatically resizes img to input dims
        output = vnnx_infer(args.model,input_array)
    elif '.xml' in args.model:
        weights=args.model.replace('.xml', '.bin')
        input_shape = openvino_input_shape(args.model, weights)[0]
        input_array = load_input(args.image, 1., input_shape, (not args.bgr))
        output = openvino_infer(args.model, input_array)
    
    plateID,conf = lpr.PlateDecodeCStyle(output)
    print("Plate ID: ", plateID,"    Recognition Score: {:3.4f}".format(conf))


if __name__ == "__main__":
    main()
