import numpy as np
import vbx.sim
import cv2
import os
import json
import onnxruntime
import argparse

import vbx.postprocess.lpd as lpd
from vbx.generate.onnx_infer import onnx_infer, load_input
from vbx.generate.openvino_infer import openvino_infer, get_model_input_shape as get_xml_input_shape
from vbx.generate.onnx_infer import onnx_infer, load_input
from vbx.generate.onnx_helper import get_model_input_shape as get_onnx_input_shape

def preProcess(img,input_shape):
    imgDims = np.array(img.shape[:2]) #img dimensions
    input_shape = input_shape[-2:] #model input dimensions

    resizeRatio = np.min(np.array(input_shape)/imgDims)
    resizeDims = np.round(imgDims * resizeRatio).astype('int')

    padTop = int((input_shape[0]-resizeDims[0])/2)    # if cropping, these values may be negative
    padBottom = input_shape[0]-resizeDims[0] - padTop
    padLeft = int((input_shape[1]-resizeDims[1])/2)
    padRight = input_shape[1]-resizeDims[1] - padLeft
    meta = {'imageX':imgDims[1],
            'imageY':imgDims[0],
            'inputX':input_shape[1],
            'inputY':input_shape[0],
            'resizeX':resizeDims[1],
            'resizeY':resizeDims[0],
            'padTop':padTop,
            'padBottom':padBottom,
            'padLeft':padLeft,
            'padRight':padRight}
    
    imgResize = cv2.resize(img, (resizeDims[1],resizeDims[0]), interpolation=cv2.INTER_LINEAR)
    arr = imgResize.swapaxes(1, 2).swapaxes(0, 1).astype(np.float32)
    return arr,meta



def get_vnnx_io_shapes(vnxx):
    with open(vnxx, 'rb') as mf:
        model = vbx.sim.Model(mf.read())
    return model.input_dims[0], model.output_dims


def vnnx_infer(vnnx_model, input_array):
    model = vbx.sim.model.Model(open(vnnx_model,"rb").read())
    input_array = input_array.astype(np.uint8)
    outputs = model.run([input_array.flatten()])
    outputs = [o/(1<<16) for o in outputs]
    

    bw = model.get_bandwidth_per_run()
    print("Bandwidth per run = {} Bytes ({:.3} MB/s at 100MHz)".format(bw,bw/100E6))
    print("Estimated {} seconds at 100MHz".format(model.get_estimated_runtime(100E6)))
    print("If running at another frequency, scale these numbers appropriately")
    return outputs

def main():
    detectThresh = .55
    maxIou = 0.2

    parser = argparse.ArgumentParser()
    parser.add_argument('model', default = 'lpd_exp42s_mod.vnnx')
    parser.add_argument('image', default = '../../test_images/parked_cars.png')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print('Error: {} could not be read'.format(args.image))
        os._exit(1)
    img = cv2.imread(args.image)
    img_height = int(img.shape[0]/2)
    cropTop = img_height
    imgLower = img[cropTop:, :, :]
    color = (0,250,0)
    width = int(np.ceil(min(img.shape[0], img.shape[1]) / 500))

    if '.vnnx' in args.model:
        input_shape , _ = get_vnnx_io_shapes(args.model)
        input_array, meta = preProcess(imgLower, input_shape) #takes lower half of img and resizes
        output = vnnx_infer(args.model,input_array)
    elif '.xml' in args.model:
        weights=args.model.replace('.xml', '.bin')
        input_shape = get_xml_input_shape(args.model, weights)
        input_array, meta = preProcess(imgLower, input_shape)
        output = openvino_infer(args.model, input_array)
    elif '.onnx' in args.model:
        input_shape = get_onnx_input_shape(args.model)
        input_array, meta = preProcess(imgLower, input_shape)  
        output = onnx_infer(args.model, input_array)      
    objs = lpd.postprocess_lpd(output,288,1024,detectThresh,maxIou)
    #resizing values back to original size
    for obj in objs:
        box = obj['box']
        box = np.array([box[0]-box[2]/2, box[1]-box[3]/2, box[0]+box[2]/2, box[1]+box[3]/2])
        box[0::2] = (box[0::2] - meta['padLeft'])/meta['resizeX']*meta['imageX']  # scale to original image
        box[1::2] = (box[1::2] - meta['padTop'])/meta['resizeY']*meta['imageY']  # scale to original image
            
        obj['box'] = box
        obj['kps'][0::2] = (obj['kps'][0::2] - meta['padLeft'])/meta['resizeX']*meta['imageX']
        obj['kps'][1::2] = (obj['kps'][1::2] - meta['padTop'])/meta['resizeY']*meta['imageY']


        p1 = (int(box[0]), int(box[1] + cropTop))
        p2 = (int(box[2]), int(box[3] + cropTop))
        cv2.rectangle(img, p1, p2, color, width)
        print("Plate Found at: ", *box.astype(int), "w/ confidence {:3.4f}".format(obj['detectScore']))

        kps = obj['kps'].reshape(4, 2).round().astype('int32')
        kps[:, 1] += cropTop
        cv2.polylines(img, [kps[[0, 1, 3, 2], :]], True, (0, 0, 255), max(1,width-1), cv2.LINE_AA)

        detText = '{:0.4f} s{:n} {:0.0f}x{:0.0f}'.format(obj['detectScore'], obj['stride'], box[2]-box[0], box[3]-box[1])
        p4 = (int(box[0]), int(box[1] - 4 * width + cropTop))
        cv2.putText(img, detText, p4, cv2.FONT_HERSHEY_SIMPLEX, width * .25, color, int(width * .75), cv2.LINE_AA)

    

    cv2.imwrite('output.png',img)

if __name__ == "__main__":
    main()