import vbx.postprocess.dataset as dataset
import vbx.sim
import argparse
import cv2
import os,os.path
import math
import numpy as np

from vbx.generate.utils import openvino_infer, openvino_input_shape
from vbx.generate.utils import load_input

def get_vnnx_io_shapes(vnxx):
    with open(vnxx, 'rb') as mf:
        model = vbx.sim.Model(mf.read())
    return model.input_shape[0], model.output_shape


def vnnx_infer(vnnx_model, input_array):
    with open(vnnx_model, "rb") as mf:
        model = vbx.sim.Model(mf.read())

    flattened = (input_array.flatten()).astype(model.input_dtypes[0])
    outputs = model.run([flattened])
    for idx, o in enumerate(outputs):
        out_scaled = model.output_scale_factor[idx] * (o.astype(np.float32) - model.output_zeropoint[idx])
        outputs[idx] = out_scaled.reshape(model.output_shape[idx])

    bw = model.get_bandwidth_per_run()
    print("Bandwidth per run = {} Bytes ({:.3} MB/s at 100MHz)".format(bw,bw/100E6))
    print("Estimated {} seconds at 100MHz".format(model.get_estimated_runtime(100E6)))
    print("If running at another frequency, scale these numbers appropriately")

    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    parser.add_argument('-b', '--bgr', action='store_true')
    parser.add_argument('--dataset',choices=['COCO','VOC'],default='VOC')
    parser.add_argument('-o', '--output', default="output.png")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print('Error: {} could not be read'.format(args.image))
        os._exit(1)
    if args.model.endswith('.vnnx'):
        input_shape, _ = get_vnnx_io_shapes(args.model)
        h, w = input_shape[1], input_shape[2]
        input_array = load_input(args.image, 1., input_shape, (not args.bgr))
        outputs = vnnx_infer(args.model, input_array)
    elif args.model.endswith('.xml'):
        weights=args.model.replace('.xml', '.bin')
        input_shape = openvino_input_shape(args.model, weights)[0]
        h, w = input_shape[2], input_shape[3]
        input_array = load_input(args.image, 1., input_shape, (not args.bgr))
        outputs = openvino_infer(args.model, input_array)

    elif args.model.endswith('.tflite'):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        interpreter= tf.lite.Interpreter(model_path=args.model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        img = cv2.imread(args.image)
        h, w = input_details[0]['shape'][1:3]
        if img.shape != (h, w, 3):
            img = cv2.resize(img, (h, w)).clip(0, 255)
        if not args.bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = img.astype(np.float32)
        img_resized = np.expand_dims(img_resized, axis=0)
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
            # output = output.transpose((0,3,1,2))
            outputs.append(output)

    assert(len(outputs)==1)
    # TopK k=1, max, decreasing order
    output = outputs[0].reshape((21, h, w)).astype('int32')
    ids = np.argsort(output, axis=0)[::-1][0,:,:]
    # values = np.sort(output, axis=0)[::-1][0,:,:]
    
    #add None Colour at start of array
    colours = np.asarray([[0, 0, 0]] + dataset.voc_colors, dtype="uint8")

    #get top category, map that to colour
    mask = colours[ids]

    img = cv2.imread(args.image)
    if img.shape != (h, w, 3):
        img = cv2.resize(img, (h, w)).clip(0, 255)

    output_img=((0.3 * img) + (0.7 * mask)).astype("uint8")
    cv2.imwrite(args.output, output_img)
