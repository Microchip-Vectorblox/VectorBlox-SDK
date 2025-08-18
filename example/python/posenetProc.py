import json
import openvino.inference_engine as ie
import vbx.sim
import numpy as np
import cv2
import os
from collections import namedtuple
import posenet_python.posenet as pnpy
import matplotlib.pyplot as plt

from vbx.generate.utils import openvino_infer, openvino_input_shape

GeometricOperationMetadata = namedtuple('GeometricOperationMetadata', ['type', 'parameters']) # matching openvino

def preprocessImage(data, rgb=None, mean=0., in_scale=1.):
    model_h = 273
    model_w = 481
    image_h, image_w = data.shape[:2]
    scale_h = model_h/image_h
    scale_w = model_w/image_w
    scale = min(scale_h,scale_w)    # keep aspect ratio
    scale = min(1,scale)    # don't upsample
    resize_w = round(scale*image_w)
    resize_h = round(scale*image_h)
    data = cv2.resize(data, (resize_w, resize_h), interpolation=2)

    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=-1)
    pad_top = int((model_h - resize_h)/2)
    pad_bottom = model_h - pad_top - resize_h
    pad_left = int((model_w - resize_w)/2)
    pad_right = model_w - pad_left - resize_w
    data = cv2.copyMakeBorder(data, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(128,128,128))
    if rgb:
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    data = (data - mean) / in_scale

    meta = {}
    meta['image_size'] = [image_h, image_w, 3]

    meta_resize = {}
    meta_resize['scale_y'] = resize_h/image_h
    meta_resize['scale_x'] = resize_w/image_w
    meta_resize['original_height'] = image_h
    meta_resize['original_width'] = image_w
    meta_resize['image_info'] = [resize_h, resize_w, 1]
    meta.setdefault('geometric_operations', []).append(GeometricOperationMetadata('resize', meta_resize))

    pad = [pad_top, pad_left, pad_bottom, pad_right]
    meta['padding'] = pad
    meta.setdefault('geometric_operations', []).append(GeometricOperationMetadata('padding',
        {'pad': pad,
         'dst_width': None,
         'dst_height': None,
         'pref_width': model_w,
         'pref_height': model_h,
         'width': resize_w,
         'height': resize_h}))

    data = data.swapaxes(1,2).swapaxes(0,1).astype(np.float32)
    data = np.expand_dims(data, axis=0)     # add an axis for the batch size
    return data, meta


def postProcess(prediction, meta, imageId=0):
    pose_scores, keypoint_scores, keypoint_coords = pnpy.decode_multiple_poses(
        prediction['heatmap'].squeeze(0).swapaxes(0,1).swapaxes(1,2),   # heatmaps
        prediction['offset'].squeeze(0).swapaxes(0,1).swapaxes(1,2),  # offsets
        prediction['displacement_fwd'].squeeze(0).swapaxes(0,1).swapaxes(1,2),  # displacement_fwd
        prediction['displacement_bwd'].squeeze(0).swapaxes(0,1).swapaxes(1,2),  # displacement_bwd
        output_stride=16,
        max_pose_detections=10,
        min_pose_score=0.25)
    
    keypoint_coords[:,:,0] += 0 - meta['geometric_operations'][1].parameters['pad'][0]
    keypoint_coords[:,:,1] += 0 - meta['geometric_operations'][1].parameters['pad'][1]
    keypoint_coords[:,:,0] *= 1.0/meta['geometric_operations'][0].parameters['scale_y']
    keypoint_coords[:,:,1] *= 1.0/meta['geometric_operations'][0].parameters['scale_x']

    res = []  # COCO result format; this will be a list of dictionaries
    for n in range(0, pose_scores.size):    # loop through number of people found
        if pose_scores[n]>0:
            newRes = {}
            newRes['image_id'] = imageId
            newRes['category_id'] = 1
            newRes['keypoints'] = [None]*51
            newRes['keypoints'][0::3] = keypoint_coords[n,:,1]
            newRes['keypoints'][1::3] = keypoint_coords[n,:,0]
            newRes['keypoints'][2::3] = keypoint_scores[n,:]
            newRes['score'] = pose_scores[n]
            res.append(newRes)
    return res


def vnnx_infer(model, input_array):
    with open(model,'rb') as mf:
        m = vbx.sim.Model(mf.read())
    input_array = input_array.astype(np.float32)
    input_array = (input_array / m.input_scale_factor[0]) + m.input_zeropoint[0]
    # input_flat = input_array.swapaxes(2,3).swapaxes(1,2).flatten().astype(m.input_dtypes[0])
    input_flat = input_array.flatten().astype(m.input_dtypes[0])

    outputs = m.run([input_flat])
    for idx, o in enumerate(outputs):
        out_scaled = m.output_scale_factor[idx] * (o.astype(np.float32) - m.output_zeropoint[idx])
        outputs[idx] = out_scaled

    bw = m.get_bandwidth_per_run()
    print("Bandwidth per run = {} Bytes ({:.3} MB/s at 100MHz)".format(bw,bw/100E6))    
    print("Estimated {} seconds at 100MHz".format(m.get_estimated_runtime(100E6)))
    print("If running at another frequency, scale these numbers appropriately")

    return outputs

def tflite_infer(model, input_array):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    interpreter= tf.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # img_resized = np.expand_dims(img_resized, axis=0)
    input_array = input_array.astype(np.float32)
    input_scale, input_zero_point = input_details[0].get('quantization', (0.0, 0))
    if  input_scale != 0.0:
        input_array = (input_array / input_scale) + input_zero_point
    input_array = input_array.astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], input_array)
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
    return outputs

def model_infer(model, input_array):
    if '.vnnx' in model:
        outputs = vnnx_infer(model, input_array)
        outputs = [outputs[2], outputs[3], outputs[1], outputs[0]]
    elif '.xml' in model:
        outputs = openvino_infer(model, input_array)
        outputs = [outputs[2], outputs[3], outputs[1], outputs[0]]
    elif '.tflite' in model:
        input_array = input_array.swapaxes(1, 2).swapaxes(2, 3).astype(np.float32) # nchw -> nhwc
        outputs = tflite_infer(model, input_array)
        outputs = [outputs[2], outputs[3], outputs[1], outputs[0]]
    prediction = {
        'heatmap':outputs[0].reshape(1,17,18,31),
        'offset':outputs[1].reshape(1,34,18,31),
        'displacement_fwd':outputs[2].reshape(1,32,18,31),
        'displacement_bwd':outputs[3].reshape(1,32,18,31)}
    return prediction


def drawRes(res):
    ax = plt.gca()
    ax.set_autoscale_on(False)
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    sks = np.array(skeleton)-1
    for r in res:
        c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
        kp = r['keypoints']
        x = kp[0::3]
        y = kp[1::3]
        v = kp[2::3]
        for sk in sks:
            if v[sk[0]]>0 and v[sk[1]]>0:
                plt.plot([x[sk[0]],x[sk[1]]],[y[sk[0]],y[sk[1]]], linewidth=3, color=c)
        for n in range(len(x)):
            if v[n]>1:
                plt.plot(x[n], y[n],'o',markersize=8, markerfacecolor=c, markeredgecolor=c, markeredgewidth=2)
            elif v[n]>0:
                plt.plot(x[n], y[n],'o',markersize=8, markerfacecolor=c, markeredgecolor='k',markeredgewidth=2)
