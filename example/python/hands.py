import vbx.sim
import argparse
import os
import numpy as np
import cv2

from vbx.generate.openvino_infer import openvino_infer, get_model_input_shape as get_xml_input_shape
from vbx.generate.onnx_infer import onnx_infer, load_input
from vbx.generate.onnx_helper import get_model_input_shape as get_onnx_input_shape

import numpy as np

# SSDAnchorOptions, calculate_scale, generate_anchors functions from 
# https://github.com/geaxgx/openvino_hand_tracker/blob/main/mediapipe_utils.py
from collections import namedtuple
SSDAnchorOptions = namedtuple('SSDAnchorOptions',[
        'num_layers',
        'min_scale',
        'max_scale',
        'input_size_height',
        'input_size_width',
        'anchor_offset_x',
        'anchor_offset_y',
        'strides',
        'aspect_ratios',
        'reduce_boxes_in_lowest_layer',         # default = false
        'interpolated_scale_aspect_ratio',      # default = 1.0
        'fixed_anchor_size'])                   # default = false

def calculate_scale(min_scale, max_scale, stride_index, num_strides):
    if num_strides == 1:
        return (min_scale + max_scale) / 2
    else:
        return min_scale + (max_scale - min_scale) * stride_index / (num_strides - 1)

def generate_anchors(options):
    """
    option : SSDAnchorOptions
    # https://github.com/google/mediapipe/blob/master/mediapipe/calculators/tflite/ssd_anchors_calculator.cc
    """
    anchors = []
    layer_id = 0
    n_strides = len(options.strides)
    while layer_id < n_strides:
        anchor_height = []
        anchor_width = []
        aspect_ratios = []
        scales = []
        # For same strides, we merge the anchors in the same order.
        last_same_stride_layer = layer_id
        while last_same_stride_layer < n_strides and \
                options.strides[last_same_stride_layer] == options.strides[layer_id]:
            scale = calculate_scale(options.min_scale, options.max_scale, last_same_stride_layer, n_strides)
            if last_same_stride_layer == 0 and options.reduce_boxes_in_lowest_layer:
                # For first layer, it can be specified to use predefined anchors.
                aspect_ratios += [1.0, 2.0, 0.5]
                scales += [0.1, scale, scale]
            else:
                aspect_ratios += options.aspect_ratios
                scales += [scale] * len(options.aspect_ratios)
                if options.interpolated_scale_aspect_ratio > 0:
                    if last_same_stride_layer == n_strides -1:
                        scale_next = 1.0
                    else:
                        scale_next = calculate_scale(options.min_scale, options.max_scale, last_same_stride_layer+1, n_strides)
                    scales.append(np.sqrt(scale * scale_next))
                    aspect_ratios.append(options.interpolated_scale_aspect_ratio)
            last_same_stride_layer += 1
        
        for i,r in enumerate(aspect_ratios):
            ratio_sqrts = np.sqrt(r)
            anchor_height.append(scales[i] / ratio_sqrts)
            anchor_width.append(scales[i] * ratio_sqrts)

        stride = options.strides[layer_id]
        feature_map_height = np.ceil(options.input_size_height / stride).astype(np.int)
        feature_map_width = np.ceil(options.input_size_width / stride).astype(np.int)

        for y in range(feature_map_height):
            for x in range(feature_map_width):
                for anchor_id in range(len(anchor_height)):
                    x_center = (x + options.anchor_offset_x) / feature_map_width
                    y_center = (y + options.anchor_offset_y) / feature_map_height
                    # new_anchor = Anchor(x_center=x_center, y_center=y_center)
                    if options.fixed_anchor_size:
                        new_anchor = [x_center, y_center, 1.0, 1.0]
                        # new_anchor.w = 1.0
                        # new_anchor.h = 1.0
                    else:
                        new_anchor = [x_center, y_center, anchor_width[anchor_id], anchor_height[anchor_id]]
                        # new_anchor.w = anchor_width[anchor_id]
                        # new_anchor.h = anchor_height[anchor_id]
                    anchors.append(new_anchor)
        
        layer_id = last_same_stride_layer
    return np.array(anchors)

# https://github.com/google/mediapipe/blob/master/mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt
HANDPOSE_SSD_OPTIONS = SSDAnchorOptions(
                            num_layers=4, 
                            min_scale=0.1484375,
                            max_scale=0.75,
                            input_size_height=128,
                            input_size_width=128,
                            anchor_offset_x=0.5,
                            anchor_offset_y=0.5,
                            strides=[8, 16, 16, 16],
                            aspect_ratios= [1.0],
                            reduce_boxes_in_lowest_layer=False,
                            interpolated_scale_aspect_ratio=1.0,
                            fixed_anchor_size=True)

# https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_detection/pose_detection_cpu.pbtxt
BLAZEPOSE_SSD_OPTIONS = SSDAnchorOptions(
                            num_layers=5, 
                            min_scale=0.1484375,
                            max_scale=0.75,
                            input_size_height=224,
                            input_size_width=224,
                            anchor_offset_x=0.5,
                            anchor_offset_y=0.5,
                            strides=[8, 16, 32, 32, 32],
                            aspect_ratios= [1.0],
                            reduce_boxes_in_lowest_layer=False,
                            interpolated_scale_aspect_ratio=1.0,
                            fixed_anchor_size=True)

# https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/hands_connections.py
HAND_PALM_CONNECTIONS = ((0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17))
HAND_THUMB_CONNECTIONS = ((1, 2), (2, 3), (3, 4))
HAND_INDEX_FINGER_CONNECTIONS = ((5, 6), (6, 7), (7, 8))
HAND_MIDDLE_FINGER_CONNECTIONS = ((9, 10), (10, 11), (11, 12))
HAND_RING_FINGER_CONNECTIONS = ((13, 14), (14, 15), (15, 16))
HAND_PINKY_FINGER_CONNECTIONS = ((17, 18), (18, 19), (19, 20))

HAND_CONNECTIONS = [
    HAND_PALM_CONNECTIONS, HAND_THUMB_CONNECTIONS,
    HAND_INDEX_FINGER_CONNECTIONS, HAND_MIDDLE_FINGER_CONNECTIONS,
    HAND_RING_FINGER_CONNECTIONS, HAND_PINKY_FINGER_CONNECTIONS
]


# https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/pose_connections.py
POSE_CONNECTIONS = frozenset([(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
                              (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
                              (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                              (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                              (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                              (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                              (29, 31), (30, 32), (27, 31), (28, 32)])



# adapted from posenetInfer , posenetProc.py
def preprocessImage(data, model_h=128, model_w=128):

    image_h, image_w = data.shape[:2]
    scale_h = model_h/image_h
    scale_w = model_w/image_w
    scale = min(scale_h,scale_w)    # keep aspect ratio
    scale = min(1,scale)    # don't upsample
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resize_w = round(scale*image_w)
    resize_h = round(scale*image_h)
    data = cv2.resize(data, (resize_w, resize_h), interpolation=interpolation)
    data = data.astype(np.float32) 
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=-1)
    pad_top = int((model_h - resize_h)/2)
    pad_bottom = model_h - pad_top - resize_h
    pad_left = int((model_w - resize_w)/2)
    pad_right = model_w - pad_left - resize_w
    data = cv2.copyMakeBorder(data, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(128,128,128))

    data = data.swapaxes(1,2).swapaxes(0,1).astype(np.float32)
    data = np.expand_dims(data, axis=0)     # add an axis for the batch size
    return data

def overlap_similarity(box, other_boxes):
    """Computes the IOU between a bounding box and set of other boxes."""
    def union(A,B):
        x1,y1,x2,y2 = A
        a = (x2-x1)*(y2-y1)
        x1,y1,x2,y2 = B
        b = (x2-x1)*(y2-y1)
        ret= a+b - intersect(A,B)
        return ret

    def intersect(A,B):
        x1= max(A[0],B[0])
        y1= max(A[1],B[1])
        x2= min(A[2],B[2])
        y2= min(A[3],B[3])
        return  max(0,x2-x1)*max(0,y2-y1)
    
    def iou(A,B):
        i = intersect(A,B)
        u = union(A,B)
        if u>0:
            return i/u
        else:
            return 0

    ret = np.array([iou(box,b) for b in other_boxes])
    return ret

class DetectorTrackerPipeline():
    """
    Adapted from BlazeFace class in our blazeface.py script, 
    which was based on code from https://github.com/tkat0/PyTorch_BlazeFace/ 
    and https://github.com/google/mediapipe/
    """
    def __init__(self, orig_img_shape, ssd_size, ssd_thresh, region_size, bp=False):
        # latest palm detector model settings:
        # https://github.com/google/mediapipe/blob/master/mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt

        # NOTE instead, these are based on the settings from the MediaPipe example graph for face
        # we do not have the latest palm detector working
        # https://github.com/google/mediapipe/blob/b133b0f2004c831c2be004291d3fcf7a378a48f4/mediapipe/graphs/face_detection/face_detection_mobile_gpu.pbtxt
        self.num_classes = 1
        self.num_anchors = 896
        self.num_coords = 18
        self.score_clipping_thresh = 100.0
        self.x_scale = ssd_size
        self.y_scale = ssd_size
        self.h_scale = ssd_size
        self.w_scale = ssd_size
        self.min_score_thresh = ssd_thresh
        self.min_suppression_threshold = 0.3
        self.blending = True
        
        self.num_keypoints = 7
        self.num_lm_keypoints = 21
        self.region_size = region_size                  # landmark model expected size
        self.orig_h, self.orig_w = orig_img_shape[:2]
        
        # https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_detection/pose_detection_cpu.pbtxt
        self.bp = bp
        if bp:
            self.num_coords = 12
            self.num_keypoints = 4
            self.num_lm_keypoints = 39
            if ssd_size==224:
                self.num_anchors = 2254


    def tensors_to_detections(self, raw_box_tensor, raw_score_tensor, anchors):
        """The output of the neural network is a tensor of shape (b, 896, 18)
        containing the bounding box regressor predictions, as well as a tensor
        of shape (b, 896, 1) with the classification confidences.
        This function converts these two "raw" tensors into proper detections.
        Returns a list of (num_detections, 19) tensors, one for each image in
        the batch.
        This is based on the source code from:
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto
        """
        assert len(raw_box_tensor.shape) == 3
        assert raw_box_tensor.shape[1] == self.num_anchors
        assert raw_box_tensor.shape[2] == self.num_coords

        assert len(raw_box_tensor.shape) == 3
        assert raw_score_tensor.shape[1] == self.num_anchors
        assert raw_score_tensor.shape[2] == self.num_classes

        assert raw_box_tensor.shape[0] == raw_score_tensor.shape[0]

        detection_boxes = self._decode_boxes(raw_box_tensor, anchors)

        thresh = self.score_clipping_thresh
        raw_score_tensor = raw_score_tensor.clip(-thresh, thresh)

        from scipy.special import expit
        detection_scores = expit(raw_score_tensor).squeeze(axis=-1)
        # detection_scores = 1/(1 + np.exp(-raw_score_tensor)).squeeze(axis=-1)
        print(np.max(detection_scores))

        # Note: we stripped off the last dimension from the scores tensor
        # because there is only has one class. Now we can simply use a mask
        # to filter out the boxes with too low confidence.
        mask = detection_scores >= self.min_score_thresh

        # Because each image from the batch can have a different number of
        # detections, process them one at a time using a loop.
        output_detections = []
        for i in range(raw_box_tensor.shape[0]):
            boxes = detection_boxes[i, mask[i]]
            scores = np.expand_dims(detection_scores[i, mask[i]],axis=-1)
            output_detections.append(np.concatenate((boxes, scores), axis=-1))

        return output_detections

    def _decode_boxes(self, raw_boxes, anchors):
        """Converts the predictions into actual coordinates using
        the anchor boxes. Processes the entire batch at once.
        """
        boxes = np.zeros(raw_boxes.shape)

        x_center = raw_boxes[..., 0] / self.x_scale * anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[..., 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]

        w = raw_boxes[..., 2] / self.w_scale * anchors[:, 2]
        h = raw_boxes[..., 3] / self.h_scale * anchors[:, 3]

        boxes[..., 0] = y_center - h / 2.  # ymin
        boxes[..., 1] = x_center - w / 2.  # xmin
        boxes[..., 2] = y_center + h / 2.  # ymax
        boxes[..., 3] = x_center + w / 2.  # xmax

        for k in range(self.num_keypoints):
            offset = 4 + k*2
            keypoint_x = raw_boxes[..., offset    ] / self.x_scale * anchors[:, 2] + anchors[:, 0]
            keypoint_y = raw_boxes[..., offset + 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]
            boxes[..., offset    ] = keypoint_x
            boxes[..., offset + 1] = keypoint_y

        return boxes

    def weighted_non_max_suppression(self, detections):
        """The alternative NMS method as mentioned in the BlazeFace paper:
        "We replace the suppression algorithm with a blending strategy that
        estimates the regression parameters of a bounding box as a weighted
        mean between the overlapping predictions."
        The original MediaPipe code assigns the score of the most confident
        detection to the weighted detection, but we take the average score
        of the overlapping detections.
        The input detections should be a Tensor of shape (count, 19).
        Returns a list of PyTorch tensors, one for each detected face.

        This is based on the source code from:
        mediapipe/calculators/util/non_max_suppression_calculator.cc
        mediapipe/calculators/util/non_max_suppression_calculator.proto
        """
        if len(detections) == 0: return []
        output_detections = []


        # Sort the detections from highest to lowest score.

        remaining = np.argsort(-detections[:, self.num_coords])
        while len(remaining) > 0:
            detection = detections[remaining[0]]

            # Compute the overlap between the first box and the other
            # remaining boxes. (Note that the other_boxes also include
            # the first_box.)
            first_box = detection[:4]
            other_boxes = detections[remaining, :4]
            ious = overlap_similarity(first_box, other_boxes)

            # If two detections don't overlap enough, they are considered
            # to be from different faces.
            mask = ious > self.min_suppression_threshold
            overlapping = remaining[mask]
            remaining = remaining[~mask]
            if self.blending:
                # Take an average of the coordinates from the overlapping
                # detections, weighted by their confidence scores.
                weighted_detection = detection.copy()
                if len(overlapping) > 1:
                    coordinates = detections[overlapping, :self.num_coords]
                    scores = detections[overlapping, self.num_coords:self.num_coords+1]
                    total_score = scores.sum()
                    weighted = (coordinates * scores).sum(axis=0) / total_score
                    weighted_detection[:self.num_coords] = weighted
                    weighted_detection[self.num_coords] = total_score / len(overlapping)
                output_detections.append(weighted_detection)
            else:
                output_detections.append(detection)

        return output_detections

    def get_detections(self, raw_output_a, raw_output_b, anchors):
        if raw_output_a.size == self.num_anchors:
            raw_score_tensor = raw_output_a
            raw_box_tensor = raw_output_b
        else:
            raw_score_tensor = raw_output_b
            raw_box_tensor = raw_output_a

        assert(raw_score_tensor.size==self.num_anchors)
        assert(raw_box_tensor.size==self.num_anchors*self.num_coords) 

        # Postprocess the raw predictions
        raw_score_tensor = raw_score_tensor.reshape(1,self.num_anchors,1)
        raw_box_tensor = raw_box_tensor.reshape(1,self.num_anchors,self.num_coords)
        detections = self.tensors_to_detections(raw_box_tensor, raw_score_tensor, anchors)

        # Non-Maximum Suppression (NMS) to remove overlapping detections
        filtered_detections = []
        for i in range(len(detections)):
            faces = self.weighted_non_max_suppression(detections[i])
            faces = np.stack(faces) if len(faces) > 0 else np.zeros((0, self.num_coords+1))
            filtered_detections.append(faces)

        return filtered_detections

    def plot_detections(self, img, detections, with_keypoints=True):
        output_img = img.copy()

        cropped_regions = []
        print("Found %d detections" % len(detections))
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
                radius = 5

                # NOTE original mediapipe pose detector is designed to detect only one person in each frame
                # only drawing first two keypoints + circle for full body blazepose
                if self.bp:
                    center_x = int(detections[i, 4] * img.shape[1])
                    center_y = int(detections[i, 5] * img.shape[0])
                    scale_x = int(detections[i, 6] * img.shape[1])
                    scale_y = int(detections[i, 7] * img.shape[0])
                    cv2.circle(output_img,(center_x,center_y),radius,(0,0,255))
                    cv2.circle(output_img,(scale_x,scale_y),radius,(0,0,255))
                    radius = int(np.sqrt(((scale_x - center_x )**2) + ((scale_y-center_y)**2)))
                    cv2.circle(output_img,(center_x,center_y),radius,(0,0,255))

                else:
                    for k in range(self.num_keypoints): 
                        kp_x = int(detections[i, 4 + k*2    ] * img.shape[1])
                        kp_y = int(detections[i, 4 + k*2 + 1] * img.shape[0])
                        cv2.circle(output_img,(kp_x,kp_y),radius,(0,0,255))
            
            rect_points= get_roi(detections[i], self.orig_w, self.orig_h, self.bp )
    
            # pass in top left, top right, bottom right
            # TODO rewrite warp_rect_img
            # from https://github.com/geaxgx/openvino_hand_tracker/blob/main/mediapipe_utils.py
            src = np.array(rect_points[1:], dtype=np.float32) # rect_points[0] is left bottom point !
            dst = np.array([(0, 0), (self.region_size, 0), (self.region_size, self.region_size)], dtype=np.float32)
            mat = cv2.getAffineTransform(src, dst)
            warp_dst = cv2.warpAffine(img, mat, (self.region_size, self.region_size))
            # ---------------------

            region_name = "region{}.png".format(i)
            cv2.imwrite(region_name, warp_dst)
            cropped_regions.append((region_name, rect_points))

        if self.bp:
            cv2.imwrite("pose_bbox.png",output_img)
        else:
            cv2.imwrite("palm_bbox.png",output_img)
    
        return cropped_regions


    def plot_landmarks(self, image, outputs, rect_points):
        coords = None

        if self.bp:
            for k in range(5):
                if outputs[k].size == 195:
                    coords = outputs[k]
                    break
        else:
            for k in range(4):
                if outputs[k].size == 63: # TODO 3d world coords output is also size 63, need to find the right output
                    coords = outputs[k]
                    break

       
        landmarks=[]
        for i in range(self.num_lm_keypoints):
            if self.bp:
                # x y z visibility presence, only care about x y z for now
                curr = 5*i
                landmarks.append(coords[curr:curr+3] / self.region_size) 
            else:
                # x,y,z
                curr = 3*i
                landmarks.append(coords[curr:curr+3] / self.region_size)

        # TODO rewrite ?
        # https://github.com/geaxgx/openvino_hand_tracker/blob/06703fa935b71a57fc21df54f804cd9878bcae10/HandTracker.py#L247
        src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
        dst = np.array([ (x, y) for x,y in rect_points[1:]], dtype=np.float32) # region.rect_points[0] is left bottom point !
        mat = cv2.getAffineTransform(src, dst)
        
        lm_xy = np.expand_dims(np.array([(l[0], l[1]) for l in landmarks]), axis=0)
        lm_xy = np.squeeze(cv2.transform(lm_xy, mat)).astype(np.int)
        # -------------------
        
        all_lines = []
        if self.bp:
            for line in POSE_CONNECTIONS:
                all_lines.append([np.array([lm_xy[point] for point in line])])
        else:
            for connection in HAND_CONNECTIONS:
                all_lines.append([np.array([lm_xy[point] for point in line]) for line in connection])


        for lines in all_lines:
            color = ((np.random.random((1,3))*0.6+0.4)*255).astype('int').tolist()[0]
            image = cv2.polylines(image, lines, False, color, 2, cv2.LINE_AA)

        radius = 6
        color = (255, 0, 0)
        for landmark in lm_xy:
            cv2.circle(image, tuple(landmark), radius, color)


def get_roi(detection, w, h, bp=False):
    # palm detection reference:
    # https://github.com/google/mediapipe/blob/master/mediapipe/modules/hand_landmark/palm_detection_detection_to_roi.pbtxt
    # https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/detections_to_rects_calculator.cc

    # pose detection reference:
    # https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_landmark/pose_detection_to_roi.pbtxt
    # https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/alignment_points_to_rects_calculator.cc

    
    # detection = [ymin xmin ymax xmax (keypoints x and y) ]

    # rotation_vector_target_angle_degrees: 90
    target_angle = np.pi / 2
    rect = {}

    if bp:
        center_x, center_y = detection[4], detection[5] # hip center keypoint
        scale_x, scale_y = detection[6], detection[7] # scale point(size, rotation) for full body detection
        box_size = np.sqrt((scale_x-center_x)**2 + (scale_y-center_y)**2) * 2
        rect['width'] = box_size
        rect['height'] = box_size
        rect['x_center'] = center_x
        rect['y_center'] = center_y

        rotation = target_angle - np.arctan2(-(scale_y - center_y), scale_x - center_x)
    else:
        xmin = detection[1]
        xmax = detection[3]
        ymin = detection[0]
        ymax = detection[2]

        rect['width'] = xmax - xmin
        rect['height'] = ymax - ymin
        rect['x_center'] = (xmin + xmax) / 2
        rect['y_center'] = (ymin + ymax) / 2

        x0, y0 = detection[4], detection[5] # Center of wrist.
        x1, y1 = detection[8], detection[9] # MCP of middle finger.
        rotation = target_angle - np.arctan2(-(y1 - y0), x1 - x0)

    def normalize_radians(angle):
        return angle - 2 * np.pi * np.floor((angle + np.pi) / (2 * np.pi))
    rotation = normalize_radians(rotation)

    # https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/rect_transformation_calculator.cc
    if bp:
        scale_x = 1.25
        scale_y = 1.25
        shift_x = 0
        shift_y = 0
        square_long = True
    else:
        scale_x = 2.6
        scale_y = 2.6
        shift_x = 0
        shift_y = -0.5
        square_long = True

    # we need to account for original img w and h for getting ROI 
    if rotation == 0:
        rect['x_center'] = (rect['x_center'] + rect['width'] * shift_x) * w
        rect['y_center'] = (rect['y_center'] + rect['height'] * shift_y) * h
    else:
        x_shift = (w * rect['width'] * shift_x * np.cos(rotation) - h * rect['height'] * shift_y * np.sin(rotation))
        y_shift = (w * rect['width'] * shift_x * np.sin(rotation) + h * rect['height'] * shift_y * np.cos(rotation))
        rect['x_center'] = rect['x_center']*w + x_shift
        rect['y_center'] = rect['y_center']*h + y_shift

    if square_long:
        long_side = max(rect['width'] * w, rect['height'] * h)
        rect['width'] = long_side * scale_x
        rect['height'] = long_side * scale_y
    else:
        short_side = min(rect['width'] * w, rect['height'] * h)
        rect['width'] = short_side * scale_x
        rect['height'] = short_side * scale_y


    # get corners of ROI box with center of box translated to 0, 0
    points = []

    # bottom left
    x0 = - rect['width']/2
    y0 = rect['height']/2
    points.append((x0,y0))

    # top left
    x1 = - rect['width']/2
    y1 = - rect['height']/2
    points.append((x1,y1))
    
    # top right
    x2 = rect['width']/2
    y2 = - rect['height']/2
    points.append((x2,y2))

    # bottom right
    x3 = rect['width']/2
    y3 = rect['height']/2
    points.append((x3,y3))

    for idx, point in enumerate(points):
        x_rotated = point[0] * np.cos(rotation) - point[1] * np.sin(rotation)
        y_rotated = point[0] * np.sin(rotation) + point[1] * np.cos(rotation)

        points[idx] = (x_rotated + rect['x_center'], y_rotated + rect['y_center'])

    return points


def vnnx_infer(vnnx_model, input_array):
    model = vbx.sim.model.Model(open(vnnx_model,"rb").read())

    flattened = input_array.flatten().astype('uint8')
    outputs = model.run([flattened])
    outputs = [o/(1<<16) for o in outputs]

    bw = model.get_bandwidth_per_run()
    print("Bandwidth per run = {} Bytes ({:.3} MB/s at 100MHz)".format(bw,bw/100E6))
    print("Estimated {} seconds at 100MHz".format(model.get_estimated_runtime(100E6)))
    print("If running at another frequency, scale these numbers appropriately")

    return outputs

def vnnx_infer_landmarks(model, input_array):
    with open(model,'rb') as mf:
        m = vbx.sim.Model(mf.read())
    input_flat = input_array.flatten().astype(np.uint8)
    outputs = m.run([input_flat])

    for n in range(len(outputs)):
        outputs[n] = outputs[n].astype(np.float32) * m.output_scale_factor[n]

    bw = m.get_bandwidth_per_run()
    print("Bandwidth per run = {} Bytes ({:.3} MB/s at 100MHz)".format(bw,bw/100E6))    
    print("Estimated {} seconds at 100MHz".format(m.get_estimated_runtime(100E6)))
    print("If running at another frequency, scale these numbers appropriately")

    return outputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('detection_model')
    parser.add_argument('-i', '--image', default='../../hands.960.640.jpg')

    parser.add_argument('--landmark-model', type=str, default=None)
    parser.add_argument('--det-only', action='store_true', help='run only detector model of pipeline')

    parser.add_argument('--det-threshold', type=float, default=0.5)
    parser.add_argument('--landmark-threshold', type=float, default=0.5)

    parser.add_argument('--det-size', type=int, default=128)
    parser.add_argument('--landmark-size', type=int, default=256)

    parser.add_argument('--bp', action='store_true', help="Blazepose processing")

    args = parser.parse_args()

    img = cv2.imread(args.image)
    input_array = preprocessImage(img, args.det_size, args.det_size)

    if args.detection_model.endswith('.vnnx'):
        outputs = vnnx_infer(args.detection_model, input_array)
    elif args.detection_model.endswith('.xml'):
        outputs = openvino_infer(args.detection_model, input_array)
    elif args.detection_model.endswith('.onnx'):
        # NOTE: to test pinto onnx, pinto needs BGR to RGB conversion and scaling, transposing
        if args.detection_model.endswith('model_float32.onnx'):
            cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_array = preprocessImage(cvt_img, args.det_size, args.det_size)
            input_array = (input_array - 127.5) / 127.5
            if not args.bp:
                input_array = np.transpose(input_array, (0,2,3,1))
        outputs = onnx_infer(args.detection_model, input_array)

    if len(outputs) == 4:
        a = np.concatenate((outputs[0], outputs[1]))
        b = np.concatenate((outputs[2], outputs[3]))
    else:
        a = outputs[0]
        b = outputs[1]

    if args.bp and args.det_size==224:
        anchor_options = BLAZEPOSE_SSD_OPTIONS
    else:
        anchor_options = HANDPOSE_SSD_OPTIONS  
    anchors = generate_anchors(anchor_options)

    pipeline = DetectorTrackerPipeline(img.shape, args.det_size, args.det_threshold, args.landmark_size, args.bp)
    detections = pipeline.get_detections(a, b, anchors)
    cropped_ROIs = pipeline.plot_detections(img, detections[0])

    # landmark model processing
    if not args.det_only:
        output_img = img.copy()
        for region_img_name, rect_points in cropped_ROIs:
            if args.landmark_model.endswith('.vnnx'):
                input_shape = (3, args.landmark_size, args.landmark_size)
                input_array = load_input(region_img_name, 1., input_shape)
                outputs = vnnx_infer_landmarks(args.landmark_model, input_array)

            elif args.landmark_model.endswith('.xml'):
                weights=args.landmark_model.replace('.xml', '.bin')
                input_shape = get_xml_input_shape(args.landmark_model, weights)
                input_array = load_input(region_img_name, 1., input_shape)
                outputs = openvino_infer(args.landmark_model, input_array)

            elif args.landmark_model.endswith('.onnx'):
                input_shape = get_onnx_input_shape(args.landmark_model)
                # NOTE: pinto needs BGR2RGB conversion, scaling, transpose
                input_array = load_input(region_img_name, 1., input_shape)
                if args.landmark_model.endswith('model_float32.onnx'):
                    input_array /= 255.0
                    if args.bp:
                        input_array = np.transpose(input_array, (0,2,3,1))
                
                outputs = onnx_infer(args.landmark_model, input_array)

            if outputs[0][0] > args.landmark_threshold:
                pipeline.plot_landmarks(output_img, outputs, rect_points)

        cv2.imwrite("landmarks.png",output_img)

if __name__ == "__main__":
    main()
