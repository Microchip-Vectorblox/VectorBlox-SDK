import vbx.sim
import vbx.postprocess.dataset as dataset
import vbx.postprocess.classifier as classifier
import cv2
import numpy as np
import argparse
import os
import math
import scipy.special

from vbx.generate.openvino_infer import openvino_infer, get_model_input_shape as get_xml_input_shape
from vbx.generate.onnx_infer import onnx_infer, load_input
from vbx.generate.onnx_helper import get_model_input_shape as get_onnx_input_shape

colors = np.array([[0,0,0,0.3],[0,255,0,0.5],[0,0,255,0.5],[0,255,255,0.7]])#,dtype='uint8')

def vnnx_infer(vnxx, input_array):
    with open(vnxx, 'rb') as mf:
        model = vbx.sim.Model(mf.read())
    flattened = input_array.flatten().astype('uint8')
    outputs = model.run([flattened])

    bw = model.get_bandwidth_per_run()
    print("Bandwidth per run = {} Bytes ({:.3} MB/s at 100MHz)".format(bw,bw/100E6))
    print("Estimated {} seconds at 100MHz".format(model.get_estimated_runtime(100E6)))
    print("If running at another frequency, scale these numbers appropriately")

    return [o.astype('float32') * sf for o,sf in zip(outputs, model.output_scale_factor)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    parser.add_argument('--output', '-o', default="output", help='output image to write labels to')
    parser.add_argument('--height', type=int, default=288, help='expected height of image')
    parser.add_argument('--width', type=int, default=800, help='expected width of image')
    parser.add_argument('--channels', type=int, default=3, help='number of channels of image')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print('Error: {} could not be read'.format(args.image))
        os._exit(1)

    if args.model.endswith('.vnnx'):
        input_shape = (args.channels, args.height, args.width)
        input_array = load_input(args.image, 1., input_shape)
        outputs = vnnx_infer(args.model, input_array)
        output = outputs[0].reshape((1, 201, 18, 4))
    elif args.model.endswith('.xml'):
        weights=args.model.replace('.xml', '.bin')
        input_shape = get_xml_input_shape(args.model, weights)
        input_array = load_input(args.image, 1., input_shape)
        outputs = openvino_infer(args.model, input_array)
        output = outputs[0]
    elif args.model.endswith('.onnx'):
        input_shape = get_onnx_input_shape(args.model)
        input_array = load_input(args.image, 1., input_shape)  
        outputs = onnx_infer(args.model, input_array)[0]

    img = cv2.imread(args.image)

    #cfg
    cfg_griding_num = 200
    cls_num_per_lane = 18
    row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]
    img_h = 590
    img_w = 1640

    #from demo.py
    col_sample = np.linspace(0, 800 - 1, cfg_griding_num)
    col_sample_w = col_sample[1] - col_sample[0]
    
    out_j = output[0,:,:,:]
    out_j = out_j[:, ::-1, :]
    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
    idx = np.arange(cfg_griding_num) + 1
    idx = idx.reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)
    out_j = np.argmax(out_j, axis=0)
    loc[out_j == cfg_griding_num] = 0
    out_j = loc

    for i in range(out_j.shape[1]):
        if np.sum(out_j[:, i] != 0) > 2:
            for k in range(out_j.shape[0]):
                if out_j[k, i] > 0:
                    ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                    cv2.circle(img,ppp,5,(0,255,0),-1)

    cv2.imwrite(args.output+'_output.png', img)
