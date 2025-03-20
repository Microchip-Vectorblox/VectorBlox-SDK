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
        if x > 0:
            return 1 / (1 + np.exp(-x))
        else:
            z = np.exp(x)
            return z / (1 + z)


    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def entry_index(self, height, width, coord, classes, location, entry):
        region_size = height * width
        n = location // region_size
        loc = location % region_size
        return int(region_size * (n * (coord + classes + 1) + entry) + loc)

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

    def parse_yolo_region(self, blob: 'np.ndarray', original_shape: list, network_shape: list, anchors: list, version='2', apply_activation=False) -> list:

        coords = 4
        num = len(anchors) // 2
        classes = (blob.shape[1] // num) - (1+coords)
        height = blob.shape[2]
        width = blob.shape[3]
        
        # ------ Extracting layer parameters --
        orig_im_h, orig_im_w = original_shape
        net_h, net_w = original_shape
        predictions = blob.flatten()
        region_size = height * width

        # ------ Parsing YOLO Region output --
        for i in range(region_size):
            row = i // width
            col = i % width
            for n in range(num):
                # -----entry index calcs------
                obj_index = self.entry_index(height, width, coords, classes, n * region_size + i, coords)
                if apply_activation:
                    predictions[obj_index] = self.logistic(predictions[obj_index])
                scale = predictions[obj_index]
                if scale < self.PROB_THRESHOLD:
                    continue
                box_index = self.entry_index(height, width, coords, classes, n * region_size + i, 0)

                # Network produces location predictions in absolute coordinates of feature maps.
                # Scale it to relative coordinates.
                if apply_activation:
                    predictions[box_index + 0 * region_size] = self.logistic(predictions[box_index + 0 * region_size])
                    predictions[box_index + 1 * region_size] = self.logistic(predictions[box_index + 1 * region_size])
                if version in ['ultra5', '5']:
                    # ultralytics yolov5 (and master yolov3)
                    x = (col + predictions[box_index + 0 * region_size] * 2 - 0.5) / width * net_w
                    y = (row + predictions[box_index + 1 * region_size] * 2 - 0.5) / height * net_h
                    h_exp = (self.logistic(predictions[box_index + 3 * region_size]) * 2) ** 2
                    w_exp = (self.logistic(predictions[box_index + 2 * region_size]) * 2) ** 2
                else:
                    x = (col + predictions[box_index + 0 * region_size]) / width * net_w
                    y = (row + predictions[box_index + 1 * region_size]) / height * net_h
                    # Value for exp is very big number in some cases so following construction is using here
                    try:
                        h_exp = exp(predictions[box_index + 3 * region_size])
                        w_exp = exp(predictions[box_index + 2 * region_size])
                    except OverflowError:
                        continue

                w = w_exp * anchors[2 * n]
                h = h_exp * anchors[2 * n + 1]

                if apply_activation:
                    activated_classes = []
                    for j in range(classes):
                        class_index = self.entry_index(height, width, coords, classes, n * region_size + i,
                                                  coords + 1 + j)
                        activated_classes.append(predictions[class_index])

                    if version == '2':
                        activated_classes = self.softmax(activated_classes)
                    else:
                        activated_classes = [self.logistic(_) for _ in activated_classes]

                    for j in range(classes):
                        class_index = self.entry_index(height, width, coords, classes, n * region_size + i,
                                                  coords + 1 + j)
                        predictions[class_index] = activated_classes[j]

                for j in range(classes):
                    class_index = self.entry_index(height, width, coords, classes, n * region_size + i,
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


def nms_post_process(output, confidence_thres=0.5, iou_thres=0.5, input_width=416, input_height=416, do_nms=True, num_classes=80, prescaled=False):
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
    x_factor, y_factor = input_width, input_height
    if prescaled:
        x_factor, y_factor = 1, 1

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


def ultralytics_process_points(arr, x, y, image_height=640, image_width=640):
    logistic = lambda x: 1 / (1 + np.exp(-x))

    _,bh,bw = arr.shape

    t = arr[:,x,y].reshape(17,3)

    t[:,2] = logistic(t[:,2])

    # x2
    t[:,:2] *= 2

    # plus offset
    t[:,0] += y
    t[:,1] += x

    # scale 
    t[:,0] *= image_width//bw
    t[:,1] *= image_height//bh

    return t.reshape((51,))


def ultralytics_process_box(arr, x, y, ang=None):
    _,bw,bh = arr.shape

    t = arr[:,x,y].reshape(4,16)
    v = np.zeros(4)

    softmax = lambda x : np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum(axis=0)
    v[0] = np.sum(softmax(t[0]) * np.asarray(range(16)))
    v[1] = np.sum(softmax(t[1]) * np.asarray(range(16)))
    v[2] = np.sum(softmax(t[2]) * np.asarray(range(16)))
    v[3] = np.sum(softmax(t[3]) * np.asarray(range(16)))

    v /= [bh,bw,bh,bw]

    cx = 1/bh/2 + y/bh
    cy = 1/bw/2 + x/bw

    x = (v[2] - v[0]) / 2
    y = (v[3] - v[1]) / 2
    if not ang is None:
        x0,y0 = x,y
        x = np.cos(ang)*x0 - np.sin(ang)*y0
        y = np.sin(ang)*x0 + np.cos(ang)*y0
    x += cx
    y += cy

    w = v[2] + v[0]
    h = v[3] + v[1]

    return [x,y,w,h]


def ultralytics_post_process(conv_outputs, input_height, input_width, threshold=0.5, num_classes=80, is_pose=False, is_obb=False, is_seg=False):

    conv_outputs = [o.squeeze() for o in conv_outputs]
    # order outputs in classes and boxes of descending size
    classes = []
    boxes = []
    points = []
    angles = []
    seg = []

    for o,output in enumerate(conv_outputs):
        if len(output.shape) == 2:
            output = np.expand_dims(output, axis=0)

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
        elif is_pose and output.shape[0] == 3*17:
            if len(points) == 0 or output.shape[1]*output.shape[2] < points[-1].shape[1]*points[-1].shape[2]:
                points += [output]
            else:
                points = [output] + points
        elif is_obb and output.shape[0] == 1:
            if len(angles) == 0 or output.shape[1]*output.shape[2] < angles[-1].shape[1]*angles[-1].shape[2]:
                angles += [output]
            else:
                angles = [output] + angles
        elif is_seg and output.shape[0] == 32:
            if len(seg) == 0 or output.shape[1]*output.shape[2] < seg[-1].shape[1]*seg[-1].shape[2]:
                seg += [output]
            else:
                seg = [output] + seg
    
    size_sort = lambda x: x.shape[-1]*x.shape[-2]
    classes.sort(reverse=True, key=size_sort)
    boxes.sort(reverse=True, key=size_sort)
    if is_pose:
        points.sort(reverse=True, key=size_sort)
    if is_obb:
        angles.sort(reverse=True, key=size_sort)
    if is_seg:
        seg.sort(reverse=True, key=size_sort)

    processed_boxes = []
    processed_classes = []
    processed_points = []
    processed_angles = []
    processed_seg = []
    processed= []
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
                    processed_classes.append(classes[c][:,j,i].flatten())
                    if is_obb:
                        angle = (logistic(angles[c][0,j,i]) - 0.25) * 3.141592741
                        processed_boxes.append(ultralytics_process_box(boxes[c], j, i, angle))
                        processed_angles.append([angle])
                    else:
                        processed_boxes.append(ultralytics_process_box(boxes[c], j, i))
                    if is_pose:
                        processed_points.append(ultralytics_process_points(points[c], j, i, input_height, input_width))
                    if is_seg:
                        processed_seg.append(seg[c][:,j, i])
                else:
                    classes[c][:,j,i] = np.zeros(num_classes)
    if valid_count>0:
        processed_classes = np.asarray(processed_classes)
        processed_boxes = np.asarray(processed_boxes)
        processed = np.concatenate((processed_boxes, processed_classes), axis=1)
        if is_obb:
            processed_angles = np.asarray(processed_angles)
            processed = np.concatenate((processed_boxes, processed_classes, processed_angles), axis=1)
        if is_pose:
            processed_points = np.asarray(processed_points)
            processed = np.concatenate((processed_boxes, processed_classes, processed_points), axis=1)
        if is_seg:
            processed_seg = np.asarray(processed_seg)
            processed = np.concatenate((processed_boxes, processed_classes, processed_seg), axis=1)
    
    return processed


def yolo_post_process(arrs, anchors, height, width, threshold, iou, version='2', do_nms=False, do_activations=False):

        parser = Parser(threshold, iou)

        for arr,anchor in zip(arrs, anchors):
            parser.parse_yolo_region(arr, (height, width), (height, width), anchor, version, do_activations)

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
