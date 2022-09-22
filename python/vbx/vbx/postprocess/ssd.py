import numpy as np
import math


coco91 = [
'unlabeled',
'person',
'bicycle',
'car',
'motorcycle',
'airplane',
'bus',
'train',
'truck',
'boat',
'traffic light',
'fire hydrant',
'street sign',
'stop sign',
'parking meter',
'bench',
'bird',
'cat',
'dog',
'horse',
'sheep',
'cow',
'elephant',
'bear',
'zebra',
'giraffe',
'hat',
'backpack',
'umbrella',
'shoe',
'eye glasses',
'handbag',
'tie',
'suitcase',
'frisbee',
'skis',
'snowboard',
'sports ball',
'kite',
'baseball bat',
'baseball glove',
'skateboard',
'surfboard',
'tennis racket',
'bottle',
'plate',
'wine glass',
'cup',
'fork',
'knife',
'spoon',
'bowl',
'banana',
'apple',
'sandwich',
'orange',
'broccoli',
'carrot',
'hot dog',
'pizza',
'donut',
'cake',
'chair',
'couch',
'potted plant',
'bed',
'mirror',
'dining table',
'window',
'desk',
'toilet',
'door',
'tv',
'laptop',
'mouse',
'remote',
'keyboard',
'cell phone',
'microwave',
'oven',
'toaster',
'sink',
'refrigerator',
'blender',
'book',
'clock',
'vase',
'scissors',
'teddy bear',
'hair drier',
'toothbrush',
'hair brush',
]

prior0 = {
       'shape': [1,12,19,19],
       'height': [30.0,42.42640750334631,84.85281500669262],
       'width' : [30.0,84.85281500669265,42.426407503346326],
       'offset' : 0.5,
       'variance': [0.1,0.1,0.2,0.2]
       }


prior1 = {
       'shape': [1,24,10,10],
       'height': [104.99999999994,74.24621202454506,148.49242404909012,60.62177826487607,181.87443025249192,125.49900360603824],
       'width' : [104.99999999994,148.49242404909015,74.24621202454507,181.8653347946282,60.61874659720808,125.49900360603824],
       'offset' : 0.5,
       'variance': [0.1,0.1,0.2,0.2]
       }
prior2 = {
       'shape': [1,24,5,5],
       'height': [149.99999910588,106.06601654574379,212.13203309148759,86.60253986222344,259.8206130978267,171.02631247097506],
       'width' : [149.99999910588,212.1320330914876,106.0660165457438,259.8076195866703,86.59820890843783,171.02631247097506],
       'offset' : 0.5,
       'variance': [0.1,0.1,0.2,0.2]
       }
prior3 = {
       'shape': [1,24,3,3],
       'height': [194.99999821182,137.88582106694255,275.7716421338851,112.58330145957083,337.7667959431616,216.3330743270663],
       'width' : [194.99999821182,275.77164213388517,137.88582106694258,337.7499043787124,112.57767121966761,216.3330743270663],
       'offset' : 0.5,
       'variance': [0.1,0.1,0.2,0.2]
       }
prior4 = {
       'shape': [1,24,2,2],
       'height': [239.99999731775998,169.7056255881413,339.4112511762826,138.5640630569182,415.71297878849646,261.5339335100698],
       'width' : [239.99999731775998,339.41125117628263,169.70562558814132,415.69218917075455,138.55713353089737,261.5339335100698],
       'offset' : 0.5,
       'variance': [0.1,0.1,0.2,0.2]
       }
prior5 = {
       'shape': [1,24,1,1],
       'width' : [284.9999964237,201.52543010934002,403.05086021868004,164.5448246542656,493.6591616338313,292.40382850966574],
       'height': [284.9999964237,403.05086021868016,201.52543010934008,493.6344739627967,164.53659584212716,292.40382850966574],
       'offset' : 0.5,
       'variance': [0.1,0.1,0.2,0.2]
       }


def convert_to_fixedpoint(data, dtype):
    # this should go away eventually, and always input uint8 rather than fixedpoint Q1.7
    if dtype == np.int16:
        shift_amt = 13
    elif dtype == np.int8:
        shift_amt = 7
    clip_max, clip_min = (1 << shift_amt)-1, -(1 << shift_amt)
    float_img = flattened.astype(np.float32)/255 * (1 << shift_amt) + 0.5

    fixedpoint_img = np.clip(float_img, clip_min, clip_max).astype(dtype)
    return fixedpoint_img


def logistic(x):
    return 1. / (1. + np.exp(-x))


def get_scores(values, confidence, top_k=None):

    scores = []
    for idx, value in enumerate(values):
        if value > confidence:
            scores.append((idx, value))

    scores.sort(reverse=True, key=lambda x: x[1])
    if top_k:
        scores = scores[:top_k]

    return scores

def intersect_box(bbox0, bbox1):
    if bbox1[0] > bbox0[2] or bbox1[2] < bbox0[0] or bbox1[1] > bbox0[3] or bbox1[3] < bbox0[1]:
        return [0., 0., 0., 0.]
    else:
        return [max(bbox0[0], bbox1[0]), max(bbox0[1], bbox1[1]), min(bbox0[2], bbox1[2]), min(bbox0[3], bbox1[3])]


def jaccard_overlap(bbox0, bbox1):
    inter_box = intersect_box(bbox0, bbox1)
    inter_w = inter_box[2] - inter_box[0]
    inter_h = inter_box[3] - inter_box[1]

    if inter_h > 0 and inter_w > 0:
        inter_size = inter_h * inter_w
        return inter_size / (bbox0[-1] + bbox1[-1] - inter_size)
    else:
        return 0.



def detections(boxes, classes, priors, num_classes, size, confidence_threshold=0.3, nms_threshold=0.6, top_k=100):
    '''
    https://github.com/openvinotoolkit/openvino/blob/d18073260bc742d7bf14d262d6919a1b660e2b61/ngraph/core/reference/include/ngraph/runtime/reference/detection_output.hpp
    clip_after_nms=True
    clip_before_nms=False
    code_type="caffe.PriorBoxParameter.CENTER_SIZE"
    confidence_threshold=0.300000011921
    input_height=1
    input_width=1
    keep_top_k=100
    nms_threshold=0.600000023842
    normalized=True
    num_classes=91
    pad_mode="caffe.ResizeParameter.CONSTANT"
    resize_mode="caffe.ResizeParameter.WARP"
    share_location=True
    top_k=100
    variance_encoded_in_target = False
    '''

    boxes = boxes[0]
    classes = classes[0]
    pboxes = priors[0]
    pvars = priors[1]

    decoded = []
    for box, pbox, pvar in zip(boxes, pboxes, pvars):
        box = box[0]
        pw = pbox[2] - pbox[0]
        ph = pbox[3] - pbox[1]
        pcx = (pbox[2] + pbox[0])/2
        pcy = (pbox[3] + pbox[1])/2

        dcx = pvar[0]*box[0] * pw + pcx
        dcy = pvar[1]*box[1] * ph + pcy
        dw = np.exp(pvar[2]*box[2]) * pw
        dh = np.exp(pvar[3]*box[3]) * ph

        d_xmin = dcx - dw / 2
        d_ymin = dcy - dh / 2
        d_xmax = dcx + dw / 2
        d_ymax = dcy + dh / 2
        d_size = 0
        if d_xmax > d_xmin and d_ymax > d_ymin:
            d_size = (d_xmax - d_xmin) * (d_ymax - d_ymin)

        decoded.append([d_xmin, d_ymin, d_xmax, d_ymax, d_size])

    decoded = np.asarray(decoded)

    valid = []
    for c in range(1, num_classes):
        scores = get_scores(classes[:,c], confidence_threshold, top_k)
        if len(scores):
            kept = []
            for idx, score in scores:
                keep = True
                for kept_idx, kept_score in kept:
                    overlap = jaccard_overlap(decoded[idx], decoded[kept_idx])
                    if overlap > nms_threshold:
                        keep = False
                        break
                if keep:
                    kept.append((idx, score))

            for idx, score in kept:
                valid.append({
                    'index': idx,
                    'class_id': c,
                    'confidence': score,
                    'xmin': min(1, max(0, decoded[idx][0])) * size,
                    'ymin': min(1, max(0, decoded[idx][1])) * size,
                    'xmax': min(1, max(0, decoded[idx][2])) * size,
                    'ymax': min(1, max(0, decoded[idx][3])) * size,
                    })

    return valid


def reverse_prior(prior, shape, repeat, img_size):
    img_h, img_w = img_size
    h = shape[-2]
    w = shape[-1]
    step_h = img_h / h
    step_w = img_w / w
    offset = 0.5
    center_x = offset * step_w
    center_y = offset * step_h

    height = []
    width = []
    for r in range(shape[1]//repeat):
        xmin = prior[r][0]
        ymin = prior[r][1]
        xmax = prior[r][2]
        ymax = prior[r][3]

        # box_w = -2*(xmin * img_w - center_x)
        # box_h = -2*(ymin * img_h - center_y)
        box_w = 2*(xmax * img_w - center_x)
        box_h = 2*(ymax * img_h - center_y)
        width.append(box_w)
        height.append(box_h)

    return height, width


def reverse_priors(priors_npy, raw_shapes, img_size):
    params = []
    priors = np.load(priors_npy).reshape((2,-1,4))[0]
    sum_raw_shapes = np.sum([np.prod(_) for _ in raw_shapes])
    assert(sum_raw_shapes % priors.shape[0] == 0)
    repeat =  sum_raw_shapes // priors.shape[0]

    idx = 0
    for shape in raw_shapes:
        inc = shape[-2]*shape[-1] * (shape[1]//repeat)
        height, width = reverse_prior(priors[idx:idx+inc], shape, repeat, img_size)
        param = {
               'shape': shape,
               'height': height,
               'width' : width,
               'offset' : 0.5,
               'variance': [0.1,0.1,0.2,0.2]
               }
        params.append(param)
        idx += inc
    return params


def get_priors(params, img_size=(300,300)):
    '''
    https://github.com/openvinotoolkit/openvino/blob/master/ngraph/core/reference/include/ngraph/runtime/reference/prior_box_clustered.hpp
    '''
    boxes = []
    variances = []
    img_h, img_w = img_size

    h = params['shape'][-2]
    w = params['shape'][-1]

    step_w = img_w / w
    step_h = img_h / h 
    offset = 0.5
    for y in range(h):
        for x in range(w):
            center_x = (x + offset) * step_w
            center_y = (y + offset) * step_h

            for (box_h, box_w) in zip(params['height'], params['width']):
                xmin = (center_x - box_w/2) / (img_w)
                ymin = (center_y - box_h/2) / (img_h)
                xmax = (center_x + box_w/2) / (img_w)
                ymax = (center_y + box_h/2) / (img_h)

                if xmin < 0.0:
                    xmin = 0.0
                if ymin < 0.0:
                    ymin = 0.0
                if xmax > 1.0:
                    xmax = 1.0
                if ymax > 1.0:
                    ymax = 1.0

                boxes.append([xmin, ymin, xmax, ymax])
    boxes = np.expand_dims(np.asarray(boxes), axis=0)

    for y in range(h):
        for x in range(w):
            for (box_h, box_w) in zip(params['height'], params['width']):
                variances.append(params['variance'])
    variances = np.expand_dims(np.asarray(variances), axis=0)

    return np.concatenate((boxes, variances))


def ssdv2_predictions(outputs, output_scale_factor, confidence_threshold=0.3, nms_threshold=0.3, top_k=100, num_classes=91):
    # reshape to original
    try:
        elem = int(math.sqrt(outputs[0].size/12))
        outputs[0] = np.reshape(outputs[0], (1,12,elem,elem))
    except:
        outputs.reverse()
        elem = int(math.sqrt(outputs[0].size/12))
        outputs[0] = np.reshape(outputs[0], (1,12,elem,elem))
    elem = int(math.sqrt(outputs[1].size/273))
    outputs[1] = np.reshape(outputs[1], (1,273,elem,elem))
    for i in range(1,6):
        elem = int(math.sqrt(outputs[2*i].size/24))
        outputs[2*i] = np.reshape(outputs[2*i], (1,24,elem,elem))
        elem = int(math.sqrt(outputs[2*i+1].size/546))
        outputs[2*i+1] = np.reshape(outputs[2*i+1], (1,546,elem,elem))

    # scale factor 
    for i in range(12):
        outputs[i] = outputs[i] * output_scale_factor[i]


    priors = get_priors(prior0)
    priors = np.concatenate((priors,get_priors(prior1)), axis=1)
    priors = np.concatenate((priors,get_priors(prior2)), axis=1)
    priors = np.concatenate((priors,get_priors(prior3)), axis=1)
    priors = np.concatenate((priors,get_priors(prior4)), axis=1)
    priors = np.concatenate((priors,get_priors(prior5)), axis=1)

    return predictions(outputs, priors, 300, confidence_threshold, nms_threshold, top_k, num_classes)


def predictions(outputs, priors, size, confidence_threshold=0.3, nms_threshold=0.3, top_k=100, num_classes=91):

    # transpose 
    for i in range(12):
        outputs[i] = np.transpose(outputs[i], (0,2,3,1))
    
    # reshape
    boxes = []
    classes = []

    # outputs = vino_outputs

    for i in range(6):
        bsize = outputs[i*2].size
        elem = bsize // 4
        boxes.append(np.reshape(outputs[i*2], (1,elem,1,4)))

    for i in range(6):
        csize = outputs[i*2+1].size
        elem = csize // num_classes
        classes.append(np.reshape(outputs[i*2+1], (1,elem,num_classes)))


    concat_boxes = np.concatenate(boxes, axis=1)
    concat_classes = np.concatenate(classes, axis=1)
    sig_classes = logistic(concat_classes)

    return detections(concat_boxes, sig_classes, priors, num_classes, size, confidence_threshold, nms_threshold, top_k)
