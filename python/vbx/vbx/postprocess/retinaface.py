import vbx.sim
import openvino.inference_engine as ie
import argparse
import os
import numpy as np
import cv2
import json
from  vbx.postprocess.blazeface import blazeface
from math import ceil
from itertools import product as product


def softmax(x, axis=1):
    if len(x.shape) == 2:
        s = np.max(x, axis=1)
        s = s[:, np.newaxis] # necessary step to do broadcasting
        e_x = np.exp(x - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis] # dito
        return e_x / div
    else:
        e_x = np.exp(x- np.max(x))
        return e_x / e_x.sum(axis=axis)
def logistic(x):
    return 1/(1+np.exp(-x))

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def decode(loc, priors, variances):
    boxes = np.concatenate((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:], priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:,:2] -= boxes[:,2:]/2
    boxes[:,2:] += boxes[:,:2]
    return boxes


def decode_landm(pre, priors, variances):
    landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), 1)
    return landms


def gen_priors(image_size=[320,320], steps=[8,16,32], min_sizes=[[16,32], [64,128], [256,512]]):
    anchors = []
    feature_maps = [[ceil(image_size[0]/step), ceil(image_size[1]/step)] for step in steps]
    for k, f in enumerate(feature_maps):
        _min_sizes = min_sizes[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in _min_sizes:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                dense_cx = [x * steps[k]  / image_size[1] for x in [j + 0.5]]
                dense_cy = [y * steps[k]  / image_size[0] for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors += [cx, cy, s_kx, s_ky]

    anchors = np.asarray(anchors).reshape(-1, 4)
    return anchors


def plot_detections(img, detections, with_keypoints=True):
        output_img = img


        print("Found %d faces" % len(detections))
        for i in range(len(detections)):
            ymin = detections[i][ 0] * img.shape[0]
            xmin = detections[i][ 1] * img.shape[1]
            ymax = detections[i][ 2] * img.shape[0]
            xmax = detections[i][ 3] * img.shape[1]
            p1 = (int(xmin),int(ymin))
            p2 = (int(xmax),int(ymax))
            print(p1,p2)
            cv2.rectangle(output_img, p1, p2, (0,50,0))

            if with_keypoints:
                for k in range(6):
                    kp_x = int(detections[i, 4 + k*2    ] * img.shape[1])
                    kp_y = int(detections[i, 4 + k*2 + 1] * img.shape[0])
                    cv2.circle(output_img,(kp_x,kp_y),1,(0,0,255))
        cv2.imwrite("output.png",output_img)

        return output_img


def openvino_infer(model_file, image):
    import openvino.inference_engine as ie
    weights=model_file.replace('.xml', '.bin')
    core = ie.IECore()
    net = core.read_network(model=model_file, weights=weights)
    assert(len(net.input_info) == 1)
    i0 = [k for k in net.input_info.keys()][0]
    outputs = [k for k in net.outputs.keys()]

    exec_net = core.load_network(network=net, device_name="CPU")
    input_height=exec_net.requests[0].input_blobs[i0].buffer.shape[-2]
    input_width=exec_net.requests[0].input_blobs[i0].buffer.shape[-1]
    img = cv2.imread(image)
    if img.shape != (input_height,input_width,3):
        img = cv2.resize(img,(input_width,input_height))
    input_array = img.swapaxes(1, 2).swapaxes(0, 1).astype(np.float32)
    input_array = np.expand_dims(input_array, axis=0)
    exec_net.requests[0].input_blobs[i0].buffer[:] = input_array
    exec_net.requests[0].infer()
    return [exec_net.requests[0].output_blobs[o].buffer.flatten() for o in outputs]


def onnx_infer(model_file, image, scale=1.0, shift=0.0, io=None):
    import onnxruntime
    session = onnxruntime.InferenceSession(model_file, None)
    input_name = session.get_inputs()[0].name
    input_height = session.get_inputs()[0].shape[-2]
    input_width = session.get_inputs()[0].shape[-1]

    if not io is None:
        with open(io) as f:
            io_dict = json.load(f)
            output_scale_factors = io_dict['output_scale_factors']
            input_scale_factors = io_dict['input_scale_factors']
    if '.npy' in image:
        input_array = np.load(image)
    else:
        img = cv2.imread(image)
        if img.shape != (input_height,input_width,3):
            img = cv2.resize(img,(input_width,input_height))
        input_array = img.swapaxes(1, 2).swapaxes(0, 1).astype(np.float32)
        if not(io is None):
            input_array /= input_scale_factors[0]
        else:
            input_array = (input_array / scale) - shift

    input_array = np.expand_dims(input_array, axis=0)
    if io is None:
        return [o.flatten() for o in session.run([], {input_name: input_array})]
    else:
        return [o.flatten() * sf for o,sf in zip(session.run([], {input_name: input_array}), output_scale_factors)]


def retinaface(outputs, image_width, image_height, confidence_threshold=0.9, nms_threshold=0.4):
    outputs = [o for o in outputs]
    if len(outputs) == 3:
        #probably ran the onnx model
        loc = outputs[0].reshape((-1, 4))
        conf = outputs[1].reshape((-1, 2))
        landms = outputs[2].reshape((-1, 10))
    else:
        factor_h = (image_height//320)
        factor_w = (image_width//320)
        ah,bh,ch = 40*factor_h, 20*factor_h, 10*factor_h
        aw,bw,cw = 40*factor_w, 20*factor_w, 10*factor_w

        outputs[0] = outputs[0].reshape((1,8,ah,aw)).transpose((0,2,3,1)).reshape((-1,4))
        outputs[1] = outputs[1].reshape((1,8,bh,bw)).transpose((0,2,3,1)).reshape((-1,4))
        outputs[2] = outputs[2].reshape((1,8,ch,cw)).transpose((0,2,3,1)).reshape((-1,4))

        outputs[3] = outputs[3].reshape((1,4,ah,aw)).transpose((0,2,3,1)).reshape((-1,2))
        outputs[4] = outputs[4].reshape((1,4,bh,bw)).transpose((0,2,3,1)).reshape((-1,2))
        outputs[5] = outputs[5].reshape((1,4,ch,cw)).transpose((0,2,3,1)).reshape((-1,2))

        outputs[6] = outputs[6].reshape((1,20,ah,aw)).transpose((0,2,3,1)).reshape((-1,10))
        outputs[7] = outputs[7].reshape((1,20,bh,bw)).transpose((0,2,3,1)).reshape((-1,10))
        outputs[8] = outputs[8].reshape((1,20,ch,cw)).transpose((0,2,3,1)).reshape((-1,10))
        loc = np.concatenate((outputs[0], outputs[1], outputs[2]))
        conf = np.concatenate((outputs[3], outputs[4], outputs[5]))

        #print(np.where(np.isclose(conf,3.52770996)))
        sm_conf = softmax(conf)
        #conf = logistic(conf)
        conf =sm_conf
        landms = np.concatenate((outputs[6], outputs[7], outputs[8]))

    variance = [0.1, 0.2]

    priors = gen_priors([image_height,image_width])

    scores = conf[:, 1]
    boxes = decode(loc, priors, variance) * [image_width, image_height, image_width, image_height]
    landms = decode_landm(landms, priors, variance) * [image_width, image_height, image_width, image_height, image_width, image_height, image_width, image_height, image_width, image_height]

    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]
    top_k = 5000
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)

    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]
    #print(dets)
    # keep top-K faster NMS
    keep_top_k = 750
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]
    dets = np.concatenate((dets, landms), axis=1)
    return [ {"box":d[:4],
              "score":d[4],
              "landmarks":d[5:].reshape((-1,2))} for d in dets]


if __name__ == "__main__":
    main()
