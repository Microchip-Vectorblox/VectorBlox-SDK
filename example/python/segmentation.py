import vbx.postprocess.dataset as dataset
import vbx.sim
import argparse
import cv2
import os,os.path
import math
import numpy as np

from vbx.generate.utils import openvino_infer, openvino_input_shape
from vbx.generate.utils import onnx_infer, onnx_input_shape
from vbx.generate.utils import load_input
import vbx.sim.model_run as mr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    parser.add_argument('-b', '--bgr', action='store_true')
    parser.add_argument('-m', '--mean', type=float, nargs='+', default=[0.])
    parser.add_argument('-sc', '--scale', type=float, nargs='+', default=[1.])
    parser.add_argument('-d', '--dataset',choices=['VOC','COCO','cityscapes','depth'],default='VOC')
    parser.add_argument('-inj', '--injected-pixels', action='store_true')
    parser.add_argument('-o', '--output', default="output.png")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print('Error: {} could not be read'.format(args.image))
        os._exit(1)

    scale = args.scale
    img = cv2.imread(args.image)
    arr, input_height, input_width, channels_last = mr.preprocess_img_to_input_array(img, args.model, args.bgr, scale)
    outputs, output_shapes = mr.model_run(arr, args.model)
    
    output=outputs[0].squeeze()

    if args.injected_pixels:
        assert(len(output.shape) == 2)

        arr = np.asarray(bytearray(output), dtype='uint8')
        arr = arr.reshape((output.shape[-2], output.shape[-1], 4))
        mask = arr[:,:,:3]
    else:
        if len(output.shape) != 2:
            if len(output.shape) == 3 and output.shape[-1] < output.shape[-3]:
                output = output.transpose((2,0,1))

            # resize bilinear if not matching network input dims
            if (input_height, input_width) != (output.shape[1], output.shape[2]):
                scaled = np.zeros((output.shape[0], input_height, input_width))
                for c,channel in enumerate(output):
                    scaled[c] = cv2.resize(channel, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
                output = scaled
                
            # TopK k=1 (argmax)
            output = np.argmax(output, axis=0)
        
        #add None Colour at start of array of 20 categories
        colors = np.asarray([[0, 0, 0]] + dataset.voc_colors, dtype="uint8")
        if args.dataset == 'VOC':
            colors = np.asarray([[0, 0, 0]] + dataset.voc_colors, dtype="uint8")
        elif args.dataset == 'COCO':
            colors = np.asarray([[0, 0, 0]] + dataset.coco_colors, dtype="uint8")
        elif args.dataset == 'cityscapes':
            rgb2bgr = lambda x: (x[2],x[1],x[0])
            colors = np.asarray([rgb2bgr(_["color"]) for _ in dataset.city_groups], dtype="uint8")   
        elif args.dataset == 'depth':
            colors = cv2.applyColorMap(np.arange(256).astype('uint8'), cv2.COLORMAP_PLASMA).reshape((256,3))
            output = output - np.min(output) 
            output = (output / np.max(output) * 255.)

        mask = colors[output.astype('int32')] #map top class to colour

    h,w = output.shape[-2], output.shape[-1]
    img = cv2.imread(args.image)
    if img.shape != (h, w, 3):
        img = cv2.resize(img, (w, h)).clip(0, 255)

    output_img=((0.3 * img) + (0.7 * mask)).astype("uint8")
    cv2.imwrite(args.output, output_img)
    print("Saved simulation result to ", args.output)
    print()
