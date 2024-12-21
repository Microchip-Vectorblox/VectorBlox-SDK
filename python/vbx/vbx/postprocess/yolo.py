"""
modified from
"""
from math import exp
import numpy as np
import cv2


class Parser:
    def __init__(self, prob_threshold=0.5, iou_threshold=0.4):
        self.objects = []
        self.IOU_THRESHOLD = iou_threshold
        self.PROB_THRESHOLD = prob_threshold

    def scale_bbox(self, x, y, h, w, class_id, confidence, h_scale, w_scale):
        xmin = int((x - w / 2) * w_scale)
        ymin = int((y - h / 2) * h_scale)
        xmax = int(xmin + w * w_scale)
        ymax = int(ymin + h * h_scale)

        return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=confidence)


    def logistic(self, x):
      return 1 / (1 + np.exp(-x))


    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def entry_index(self, side, coord, classes, location, entry):
        side_power_2 = side ** 2
        n = location // side_power_2
        loc = location % side_power_2
        return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)

    def intersection_over_union(self, box_1, box_2):
        width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
        height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
        if width_of_overlap_area < 0 or height_of_overlap_area < 0:
            area_of_overlap = 0
        else:
            area_of_overlap = width_of_overlap_area * height_of_overlap_area
        box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
        box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
        area_of_union = box_1_area + box_2_area - area_of_overlap
        if area_of_union == 0:
            return 0
        return area_of_overlap / area_of_union


    def sort_objects(self):
        self.objects = sorted(self.objects, key=lambda obj : obj['confidence'], reverse=True)

        for i in range(len(self.objects)):
            if self.objects[i]['confidence'] == 0:
                continue
            for j in range(i + 1, len(self.objects)):
                if self.intersection_over_union(self.objects[i], self.objects[j]) > self.IOU_THRESHOLD:
                    self.objects[j]['confidence'] = 0

    def parse_yolo_region(self, blob: 'np.ndarray', original_shape: list, network_shape: list, params: dict, version=2, apply_activation=False) -> list:

        num = params['num']
        coords = params['coords']
        classes = params['classes']
        # -----------------

        _, _, out_blob_h, out_blob_w = blob.shape
        assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                         "be equal to width. Current height = {}, current width = {}" \
                                         "".format(out_blob_h, out_blob_w)

        # ------ Extracting layer parameters --
        orig_im_h, orig_im_w = original_shape
        net_h, net_w = original_shape
        predictions = blob.flatten()
        side_square = params['side'] * params['side']

        # ------ Parsing YOLO Region output --
        for i in range(side_square):
            row = i // params['side']
            col = i % params['side']
            for n in range(num):
                # -----entry index calcs------
                obj_index = self.entry_index(params['side'], coords, classes, n * side_square + i, coords)
                if apply_activation:
                    predictions[obj_index] = self.logistic(predictions[obj_index])
                scale = predictions[obj_index]
                if scale < self.PROB_THRESHOLD:
                    continue
                box_index = self.entry_index(params['side'], coords, classes, n * side_square + i, 0)

                # Network produces location predictions in absolute coordinates of feature maps.
                # Scale it to relative coordinates.
                if apply_activation:
                    predictions[box_index + 0 * side_square] = self.logistic(predictions[box_index + 0 * side_square])
                    predictions[box_index + 1 * side_square] = self.logistic(predictions[box_index + 1 * side_square])
                if version > 3:
                    # ultralytics yolov5 (and master yolov3)
                    x = (col + predictions[box_index + 0 * side_square] * 2 - 0.5) / params['side'] * net_w
                    y = (row + predictions[box_index + 1 * side_square] * 2 - 0.5) / params['side'] * net_h
                    h_exp = (self.logistic(predictions[box_index + 3 * side_square]) * 2) ** 2
                    w_exp = (self.logistic(predictions[box_index + 2 * side_square]) * 2) ** 2
                else:
                    x = (col + predictions[box_index + 0 * side_square]) / params['side'] * net_w
                    y = (row + predictions[box_index + 1 * side_square]) / params['side'] * net_h
                    # Value for exp is very big number in some cases so following construction is using here
                    try:
                        h_exp = exp(predictions[box_index + 3 * side_square])
                        w_exp = exp(predictions[box_index + 2 * side_square])
                    except OverflowError:
                        continue

                w = w_exp * params['anchors'][2 * n]
                h = h_exp * params['anchors'][2 * n + 1]

                if apply_activation:
                    activated_classes = []
                    for j in range(classes):
                        class_index = self.entry_index(params['side'], coords, classes, n * side_square + i,
                                                  coords + 1 + j)
                        activated_classes.append(predictions[class_index])

                    if version < 3:
                        activated_classes = self.softmax(activated_classes)
                    else:
                        activated_classes = [self.logistic(_) for _ in activated_classes]

                    for j in range(classes):
                        class_index = self.entry_index(params['side'], coords, classes, n * side_square + i,
                                                  coords + 1 + j)
                        predictions[class_index] = activated_classes[j]

                for j in range(classes):
                    class_index = self.entry_index(params['side'], coords, classes, n * side_square + i,
                                              coords + 1 + j)
                    confidence = scale * predictions[class_index]
                    if confidence < self.PROB_THRESHOLD:
                        continue

                    self.objects.append(self.scale_bbox(x=x,
                                                        y=y,
                                                        h=h,
                                                        w=w,
                                                        class_id=j,
                                                        confidence=confidence,
                                                        h_scale=(orig_im_h/net_h),
                                                        w_scale=(orig_im_w/net_w)))


def nms_post_process(output, confidence_thres=0.5, iou_thres=0.5, input_width=416, input_height=416, do_nms=True, num_classes=80):
    # Transpose and squeeze the output to match the expected shape
    outputs = np.squeeze(output)
    # if (outputs.shape[0] < outputs.shape[1]): # should be #box, box (8400,84)
    if (outputs.shape[1] != num_classes + 4 and outputs.shape[0] == num_classes + 4): # should be #box, box (8400,84)
        outputs = np.transpose(outputs)

    # Get the number of rows in the outputs array
    rows = outputs.shape[0]

    # Lists to store the bounding boxes, scores, and class IDs of the detections
    boxes = []
    scores = []
    class_ids = []

    # Calculate the scaling factors for the bounding box coordinates
    x_factor = input_width
    y_factor = input_height

    # Iterate over each row in the outputs array
    for i in range(rows):
        # Extract the class scores from the current row
        classes_scores = outputs[i][4:]

        # Find the maximum score among the class scores
        max_score = np.amax(classes_scores)

        # If the maximum score is above the confidence threshold
        if max_score >= confidence_thres:
            # Get the class ID with the highest score
            class_id = np.argmax(classes_scores)

            # Extract the bounding box coordinates from the current row
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

            # Calculate the scaled coordinates of the bounding box
            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)

            # Add the class ID, score, and box coordinates to the respective lists
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])

    # Apply non-maximum suppression to filter out overlapping bounding boxes
    if do_nms:
        indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)
    else:
        indices = range(len(boxes)) #TODO verify

    results = []
    for i in indices:
        confidence = scores[i]
        label = class_ids[i]
        box = boxes[i]
        xmin = box[0]
        xmax = box[0] + box[2]
        ymin = box[1]
        ymax = box[1] + box[3]

        # Enforcing extra checks for bounding box coordinates
        xmin = max(0,xmin)
        ymin = max(0,ymin)
        xmax = min(xmax,input_width)
        ymax = min(ymax,input_height)

        results.append(dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=label, confidence=confidence))
    return results


def ultralytics_process_box(arr, x, y):
    _,bw,bh = arr.shape

    t = arr[:,x,y].reshape(4,16)
    v = np.zeros(4)

    softmax = lambda x : np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum(axis=0)
    v[0] = np.sum(softmax(t[0]) * np.asarray(range(16)))
    v[1] = np.sum(softmax(t[1]) * np.asarray(range(16)))
    v[2] = np.sum(softmax(t[2]) * np.asarray(range(16)))
    v[3] = np.sum(softmax(t[3]) * np.asarray(range(16)))

    v /= [bh,bw,bh,bw] #v *= 1 / np.sqrt(bh*bw)

    cx = 1/bh/2 + y/bh
    cy = 1/bw/2 + x/bw

    r = (cx - v[0])
    lo = (cy - v[1])
    l = (cx + v[2])
    u = (cy + v[3])

    x = (l + r) / 2
    y = (lo + u) / 2

    w = (l - r)
    h = (u - lo)

    return [x,y,w,h]


def ultralytics_post_process(conv_outputs, threshold=0.5, num_classes=80):

    conv_outputs = [o.squeeze() for o in conv_outputs]
    # order outputs in classes and boxes of descending size
    classes = []
    boxes = []
    for o,output in enumerate(conv_outputs):
        if output.shape[0] == num_classes:
            if len(classes) == 0 or output.shape[1]*output.shape[2] < classes[-1].shape[1]*classes[-1].shape[2]:
                classes += [output]
            else:
                classes = [output] + classes
        elif output.shape[0] == 4*16:
            if len(boxes) == 0 or output.shape[1]*output.shape[2] < boxes[-1].shape[1]*boxes[-1].shape[2]:
                boxes += [output]
            else:
                boxes = [output] + boxes


    processed_boxes = []
    processed_classes = []

    valid_count = 0
    valid_coords = []
    logistic = lambda x: 1 / (1 + np.exp(-x))
    for c,cls in enumerate(classes):
        classes[c] = logistic(classes[c])
        for j in range(classes[c].shape[1]):
            for i in range(classes[c].shape[2]):
                valid = False
                for x in range(num_classes):
                    if classes[c][x,j,i] > threshold:
                        valid = True
                        valid_count += 1
                        break
                if valid:
                    processed_boxes.append(ultralytics_process_box(boxes[c], j, i))
                    processed_classes.append(classes[c][:,j,i].flatten())
                else:
                    classes[c][:,j,i] = np.zeros(num_classes)
    # print('valid', valid_count)

    processed_classes = np.asarray(processed_classes)
    processed_boxes = np.asarray(processed_boxes)
    processed = np.concatenate((processed_boxes, processed_classes), axis=1)

    return processed

def yolo_post_process(blobs, params, height, width, threshold, iou, version=2, do_nms=False, do_activations=False):

        parser = Parser(threshold, iou)

        for name in params:
            arr = blobs[name]
            arr =  np.reshape(arr, params[name]['shape'])
            parser.parse_yolo_region(arr, (height, width), (height, width), params[name], version, do_activations)

        if do_nms:
            parser.sort_objects()

        results = []
        for obj in parser.objects:
            if obj['confidence'] >= parser.PROB_THRESHOLD:
                label = obj['class_id']
                xmin = obj['xmin']
                xmax = obj['xmax']
                ymin = obj['ymin']
                ymax = obj['ymax']

                # Enforcing extra checks for bounding box coordinates
                xmin = max(0,xmin)
                ymin = max(0,ymin)
                xmax = min(xmax,width)
                ymax = min(ymax,height)

                results.append(dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=label, confidence=obj['confidence']))
        return results


yolo_v3_params = {
        'detector/yolo-v3/Conv_6/BiasAdd/YoloRegion':  {
            'side': 19,
            'anchors': [116,90,156,198,373,326],
            'classes': 80,
            'num': 3,
            'coords': 4,
            'shape': [1, 255, 19, 19],
            },
        'detector/yolo-v3/Conv_14/BiasAdd/YoloRegion': {
            'side': 38,
            'anchors': [30,61,62,45,59,119],
            'classes': 80,
            'num': 3,
            'coords': 4,
            'shape': [1, 255, 38, 38],
            },
        'detector/yolo-v3/Conv_22/BiasAdd/YoloRegion': {
            'side': 76, 
            'anchors': [10,13,16,30,33,23],
            'classes': 80,
            'num': 3,
            'coords': 4,
            'shape': [1, 255, 76, 76],
            },
        }

yolo_v3_voc_params = {
        'detector/yolo-v3/Conv_6/BiasAdd/YoloRegion':  {
            'side': 13,
            'anchors': [116,90,156,198,373,326],
            'classes': 20,
            'num': 3,
            'coords': 4,
            'shape': [1, 75, 13, 13],
            },
        'detector/yolo-v3/Conv_14/BiasAdd/YoloRegion': {
            'side': 26,
            'anchors': [30,61,62,45,59,119],
            'classes': 20,
            'num': 3,
            'coords': 4,
            'shape': [1, 75, 26, 26],
            },
        'detector/yolo-v3/Conv_22/BiasAdd/YoloRegion': {
            'side': 52, 'anchors': [10,13,16,30,33,23],
            'classes': 20,
            'num': 3,
            'coords': 4,
            'shape': [1, 75, 52, 52],
            },
        }

yolo_v3_tiny_params = {
        'detector/yolo-v3-tiny/Conv_9/BiasAdd/YoloRegion': {
            'id': '26',
            'side': 13,
            'anchors': [81,82,135,169,344,319],
            'classes': 80,
            'num': 3,
            'coords': 4,
            'shape': [1, 255, 13, 13],
            'scale': 29.526176,
            },
        'detector/yolo-v3-tiny/Conv_12/BiasAdd/YoloRegion': {
            'id': '34',
            'side': 26,
            'anchors': [23,27,37,58,81,82],
            'classes': 80,
            'num': 3,
            'coords': 4,
            'shape': [1, 255, 26, 26],
            'scale': 28.07609,
            },
        }

yolov2_tiny_voc_params = {
        'output/YoloRegion':  {
            'side': 13,
            'anchors': [_*32 for _ in [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]],
            'classes': 20,
            'num': 5,
            'scale': 21.9720645,
            'coords': 4,
            'shape': [1, 125, 13, 13],
            'id': '24',
            },
        }

yolov2_voc_params = {
        'output/YoloRegion':  {
            'side': 13,
            'anchors': [_*32 for _ in [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]],
            'classes': 20,
            'num': 5,
            'coords': 4,
            'shape': [1, 125, 13, 13],
            },
        }

yolov2_tiny_params = {
        'output/YoloRegion':  {
            'side': 13,
            'classes': 80,
            'num': 5,
            'coords': 4,
            'shape': [1, 425, 13, 13],
            'anchors': [_*32 for _ in [0.57273,0.677385,1.87446,2.06253,3.33843,5.47434,7.88282,3.52778,9.77052,9.16828]],
            },
        }

yolov2_params = {
        'output/YoloRegion':  {
            'side': 19,
            'classes': 80,
            'num': 5,
            'coords': 4,
            'shape': [1, 425, 19, 19],
            'anchors': [_*32 for _ in [0.57273,0.677385,1.87446,2.06253,3.33843,5.47434,7.88282,3.52778,9.77052,9.16828]],
            },
        }

def get_param_by_output_size(output,parameters):
    from functools import reduce
    import operator
    size = output.flatten().shape[0]
    for p in parameters:
        if size == reduce(operator.mul,parameters[p]['shape']):
            return p


def yolov2_coco(outputs, scale_factors, threshold=0.3, iou=0.4, do_activations=True):
    blobs = {'output/YoloRegion': np.reshape(outputs[0].astype(np.float32)*scale_factors[0], (425, 19, 19))}
    return yolo_post_process(blobs, yolov2_params, 608, 608, threshold, iou, 2, True, do_activations)


def yolov2_voc(outputs, scale_factors, threshold=0.3, iou=0.4, do_activations=True):
    blobs = {'output/YoloRegion': np.reshape(outputs[0].astype(np.float32)*scale_factors[0], (125, 13, 13))}
    return yolo_post_process(blobs, yolov2_voc_params, 416, 416, threshold, iou, 2, True, do_activations)


def yolov2_tiny_voc(outputs, scale_factors, threshold=0.3, iou=0.4):
    blobs = {'output/YoloRegion': np.reshape(outputs[0].astype(np.float32)*scale_factors[0], (125, 13, 13))}
    return yolo_post_process(blobs, yolov2_tiny_voc_params, 416, 416, threshold, iou, 2, True, True)


def yolov2_tiny_coco(outputs, scale_factors, threshold=0.3, iou=0.4):
    blobs = {}
    params = yolov2_tiny_params
    for out, scale in zip(outputs,scale_factors):
        p = get_param_by_output_size(out,params)
        blobs[p] = out.astype(np.float32).reshape(params[p]['shape']) * scale

    return yolo_post_process(blobs, params, 416, 416, threshold, iou, 3, True, True)


def yolov3_tiny_voc(outputs,scale_factors):
    raise NotImplementedError


def yolov3_tiny_coco(outputs, scale_factors, threshold=0.3, iou=0.4):

    blobs = {}
    params = yolo_v3_tiny_params
    for out, scale in zip(outputs,scale_factors):
        p = get_param_by_output_size(out,params)
        blobs[p] = out.astype(np.float32).reshape(params[p]['shape']) * scale
    return yolo_post_process(blobs, params, 416, 416, threshold, iou, 3, True, True)


def yolov3_coco(outputs, scale_factors, threshold=0.3, iou=0.4):
    blobs = {}
    params = yolo_v3_params
    for out, scale in zip(outputs,scale_factors):
        p = get_param_by_output_size(out,params)
        blobs[p] = out.astype(np.float32).reshape(params[p]['shape']) * scale

    return yolo_post_process(blobs, params, 608, 608, threshold, iou, 3, True, True)


def demo_post_process(outputs, scale=1, network='tiny_voc', threshold=0.3, iou=0.4, do_nms=True, do_softmax=True):

    if network == 'tiny_voc':
        params =  yolov2_tiny_voc_params
        height, width = 416, 416
        blobs = {'output/YoloRegion': outputs*scale}
        version = 2

    return yolo_post_process(blobs, params, height, width, version, threshold, iou, do_nms, do_softmax)
