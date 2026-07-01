import cv2
import numpy as np
import math
import os
import onnx
import onnxruntime as rt
import vbx.postprocess.dataset as dataset
import argparse
import tensorflow as tf
import vbx.sim.model_run as mr
import vbx.generate.utils as utils
from vbx.postprocess.linemod import *
import vbx.postprocess.yolo as yolo


def create_anchors(image_size=(768, 512), scales=[2**0, 2**(1.0/3.0), 2** (2.0/3.0)], ratios=[[1.0,1.0],[1.4,0.7],[0.7,1.4]]):
    """
    adapted and modified from https://github.com/google/automl/blob/master/efficientdet/anchors.py by Zylo117

    Returns:
        anchor_boxes: [N x 4] anchor boxes [xmin, ymin, xmax, ymax] (pix) in input image dimension
        strides: record of strides associated with eacn anchor boxes
    """
    # Basic scaling factor
    anchor_scale = 4

    # Pyramid levels from EfficientDet to use
    # Even if segmentation head exists that uses P2, ignore it
    # (otherwise too many anchor boxes)
    pyramid_levels = [3, 4, 5, 6, 7]

    strides = [2 ** x for x in pyramid_levels]
    if scales is None: scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], dtype=np.float32)
    if ratios is None: ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]

    # Generates multiscale anchor boxes.
    H, W = image_size

    boxes_all   = []
    strides_all = []
    for stride in strides:
        boxes_level   = []
        for s,scale in enumerate(scales):
            for r,ratio in enumerate(ratios):
                if W % stride != 0:
                    raise ValueError('input size must be divisible by the stride.')
                base_anchor_size = anchor_scale * stride * scale
                anchor_size_x_2 = base_anchor_size * ratio[0] / 2
                anchor_size_y_2 = base_anchor_size * ratio[1] / 2

                x = np.arange(int(stride / 2), W, stride)
                y = np.arange(int(stride / 2), H, stride)
                xv, yv = np.meshgrid(x, y)
                xv = xv.reshape(-1)
                yv = yv.reshape(-1)

                # [xmin, ymin, xmax, ymax]
                boxes = np.vstack((xv - anchor_size_x_2, yv - anchor_size_y_2,
                                   xv + anchor_size_x_2, yv + anchor_size_y_2))
                boxes = np.swapaxes(boxes, 0, 1)
                boxes_level.append(np.expand_dims(boxes, axis=1))

                # strides
                [strides_all.append(stride) for _ in range(len(xv))]

        # concat anchors on the same level to the reshape NxAx4
        boxes_level = np.concatenate(boxes_level, axis=1)
        boxes_all.append(boxes_level.reshape([-1, 4]))

    anchor_boxes = np.vstack(boxes_all).astype(np.float32)
    return anchor_boxes, strides_all


def rot_6d_to_matrix(arr):
    r1_raw = arr[:3]
    r2_raw = arr[3:]

    # First column
    r1 = r1_raw / np.linalg.norm(r1_raw)

    # # Second column
    dot = np.sum(r1 * r2_raw)
    r2 = r2_raw - dot * r1
    r2 = r2 / np.linalg.norm(r2)

    r3 = np.cross(r1, r2)

    # # Into matrix
    dcm = np.stack((r1, r2, r3), axis=-1)

    return dcm


def delta_xy_tz_to_translation(translation_raw, image_size, input_size, anchors, strides, cameraMatrix):
    dx, dy, tz = translation_raw[0], translation_raw[1], translation_raw[2]

    cx = dx * strides + (anchors[0] + anchors[2])/2
    cy = dy * strides + (anchors[1] + anchors[3])/2

    # (cx, cy) are in terms of input image size, so convert to original size
    cx = cx / input_size[0] * image_size[0]
    cy = cy / input_size[1] * image_size[1]

    tx = (cx - cameraMatrix[0,2]) * tz / cameraMatrix[0,0]
    ty = (cy - cameraMatrix[1,2]) * tz / cameraMatrix[1,1]

    return np.stack((tx, ty, tz), axis=-1) # [B, N, 3]


def raw_output_to_bbox(bbox_raw, image_size, anchor):
    anchor_width    = anchor[2] - anchor[0]
    anchor_height   = anchor[3] - anchor[1]
    anchor_center_x = anchor[0] + anchor_width/2
    anchor_center_y = anchor[1] + anchor_height/2

    # Current box prediction (xc, yc, w, h)
    x = anchor_center_x + anchor_width * bbox_raw[0]
    y = anchor_center_y + anchor_height * bbox_raw[1]
    w = anchor_width * np.exp(bbox_raw[2])
    h = anchor_height * np.exp(bbox_raw[3])

    # Into [xmin, ymin, xmax, ymax]
    bbox = np.zeros(bbox_raw.shape)
    bbox[0] = np.clip(x - w/2, 0, image_size[0]-1)
    bbox[1] = np.clip(y - h/2, 0, image_size[1]-1)
    bbox[2] = np.clip(x + w/2, 0, image_size[0]-1)
    bbox[3] = np.clip(y + h/2, 0, image_size[1]-1)

    return bbox


def project_keypoints(dcm, r, K, dist, keypoints):
    """ Projecting 3D keypoints to 2D
        dcm: cosinematrix (np.array)
        r: position   (np.array)
        K: camera intrinsic (3,3) (np.array)
        dist: distortion coefficients (5,) (np.array)
        keypoints: N x 3 or 3 x N (np.array)
    """
    # Make sure keypoints are 3 x N
    if keypoints.shape[0] != 3:
        keypoints = np.transpose(keypoints)

    # Keypoints into 4 x N homogenous coordinates
    keypoints = np.vstack((keypoints, np.ones((1, keypoints.shape[1]))))

    # transformation to image frame
    pose_mat = np.hstack((np.transpose(dcm), np.expand_dims(r, 1)))
    xyz      = np.dot(pose_mat, keypoints) # [3 x N]
    x0, y0   = xyz[0,:] / xyz[2,:], xyz[1,:] / xyz[2,:] # [1 x N] each

    # apply distortion
    r2 = x0*x0 + y0*y0
    cdist = 1 + dist[0]*r2 + dist[1]*r2*r2 + dist[4]*r2*r2*r2
    x  = x0*cdist + dist[2]*2*x0*y0 + dist[3]*(r2 + 2*x0*x0)
    y  = y0*cdist + dist[2]*(r2 + 2*y0*y0) + dist[3]*2*x0*y0

    # apply camera matrix
    points2D = np.vstack((K[0,0]*x + K[0,2], K[1,1]*y + K[1,2]))

    return points2D


def draw_pose(img, points2D, scale_x=1.0, scale_y=1.0, thickness=2):
    """Draw the projected 3D pose wireframe onto the image.

    The 12 keypoints represent the Tango satellite:
        0: centroid
        1-4: front face corners
        5-8: body ring corners
        9-10: solar panel tips
        11: antenna

    Args:
        img: image to draw on
        points2D: 2 x N projected pixel coordinates
        scale_x, scale_y: scale from projected coords to output image coords
    """
    h, w = img.shape[:2]
    margin = max(w, h) * 50

    pts = []
    for i in range(points2D.shape[1]):
        x_val = points2D[0, i] * scale_x
        y_val = points2D[1, i] * scale_y
        if (np.isfinite(x_val) and np.isfinite(y_val)
                and -margin <= x_val <= w + margin
                and -margin <= y_val <= h + margin):
            pts.append((int(round(x_val)), int(round(y_val))))
        else:
            pts.append(None)

    edge_groups = [
        ([(1,2),(2,3),(3,4),(4,1)],       (0, 255, 0)),    # front face - green
        ([(5,6),(6,7),(7,8),(8,5)],       (255, 0, 0)),    # body ring - blue
        ([(1,5),(2,6),(3,7),(4,8)],       (0, 255, 255)),  # sides - yellow
        ([(2,9),(3,10),(4,11)],           (0, 165, 255)),  # solar panels - orange
        #([(11,5),(11,8)],                 (255, 0, 255)),  # antenna - magenta
    ]

    for edges, c in edge_groups:
        for (i, j) in edges:
            if i < len(pts) and j < len(pts) and pts[i] is not None and pts[j] is not None:
                cv2.line(img, pts[i], pts[j], c, thickness)

    for idx, pt in enumerate(pts):
        if pt is None:
            continue
        if idx == 0:
            cv2.circle(img, pt, 5, (0, 0, 255), -1)
        else:
            cv2.circle(img, pt, 3, (255, 255, 255), -1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    parser.add_argument('--mean', type=float, nargs='+', default=[0.])
    parser.add_argument('-sc', '--scale', type=float, nargs='+', default=[1.])
    parser.add_argument('-b', '--bgr', action='store_true')
    parser.add_argument('-t', '--append-transpose-first', action='store_true')
    parser.add_argument('-th', '--threshold', type=float, default=0.5)
    parser.add_argument('--output', '-o', default="output.png", help='output image to write labels to')
    args = parser.parse_args()

    anchors, strides = create_anchors()

    # from spnv2/core/utils/models/tangoPoints.mat
    cameraMatrix = np.asarray([[2988.5795163815555,0,960],[0,2988.3401159176124,600],[0,0,1]])
    distCoeffs = np.asarray([ -0.22383016606510672, 0.51409797089106379, -0.00066499611998340662, -0.00021404771667484594, -0.13124227429077406])
    Nu,Nv = 1920, 1200
    tangoKeypoints = np.asarray([[0.,-0.37, -0.37, 0.37, 0.37, -0.37, -0.37, 0.37, 0.37, -0.5427, 0.5427, 0.305],
                                 [0.,-0.385, 0.385, 0.385, -0.385, -0.264, 0.304, 0.304, -0.264, 0.4877, 0.4877, -0.579],
                                 [0.,0.3215, 0.3215, 0.3215, 0.3215, 0., 0., 0., 0., 0.2535, 0.2591, 0.2515]], dtype=np.float32)


    img = cv2.imread(args.image)
    arr, input_height, input_width, channels_last = mr.preprocess_img_to_input_array(img, args.model, args.bgr, args.scale, args.mean)
    outputs, output_shapes = mr.model_run(arr, args.model)


    bbox_pr = None
    if len(outputs) == 9: #yolo backbone
        if channels_last: #if model is NHWC outputs need to be converted to
            outputs=mr.transpose_outputs(outputs)

        scores, yolo_outputs, pose_outputs = [],[],[]
        max_idx, max_score, max_set, max_y, max_x = -1, -999, -1, -1, -1

        for o,output in enumerate(outputs):
            if output.shape[1] in [1,64]:
                yolo_outputs.append(output)
            else:
                pose_outputs.append(output)
            if output.shape[1] in [1]:
                scores.append(output)

        scores = sorted(scores, key=lambda x: -np.prod(x.shape))
        pose_outputs = sorted(pose_outputs, key=lambda x: -np.prod(x.shape))

        for o,output in enumerate(scores):
            score = np.max(output)
            if np.max(output) > max_score:
                max_score, max_idx, max_set = np.max(output), np.argmax(output), o
                max_y, max_x = max_idx // output.shape[-1], max_idx % output.shape[-1]

        max_pose = pose_outputs[max_set][0,:,max_y, max_x]
        stride = [8,16,32][max_set]
        # ax, ay = np.arange(input_width // stride), np.arange(input_height // stride)
        # anchor_x, anchor_y = ax[max_x]*stride + stride//2, ay[max_y]*stride + stride//2
        anchor_x, anchor_y = max_x*stride + stride//2, max_y*stride + stride//2

        t_pr = delta_xy_tz_to_translation(max_pose[:3], (Nu,Nv), (768,512), [anchor_x, anchor_y, anchor_x, anchor_y], stride, cameraMatrix)
        R_pr = rot_6d_to_matrix(max_pose[3:])

        post = yolo.ultralytics_post_process(yolo_outputs, input_height, input_width, args.threshold, num_classes=1)

        if len(post):
            [x,y,w,h,score] = sorted(post, key=lambda x: -x[-1])[0]

            left = (x - w / 2) * input_width
            top = (y - h / 2) * input_height
            width = (w * input_width)
            height = (h * input_height)
            bbox_pr = [left, top, left+width, top+height]
    else:
        if len(outputs) == 4: # concatenated efficientnet backbone
            classification, translation, bbox_prediction, rotation_raw = sorted(outputs, key=lambda x: np.prod(x.shape))
        else: # split efficientnet backbone
            if '.vnnx' in args.model or '.onnx' in args.model: 
                for o,output in enumerate(outputs):
                    outputs[o] = output.transpose((0,2,3,1))

            c,t0,t1,b,r = [],[],[],[],[]
            for o,output in enumerate(outputs):
                in_c = output.size in [_.size for _ in c]
                in_t0 = output.size in [_.size for _ in t0]

                if output.shape[-1] == 9 and not in_c and args.append_transpose_first is in_t0:
                    c.append(output.reshape(1,-1))
                elif output.shape[-1] == 9:
                    t0.append(output)
                elif output.shape[-1] == 18:
                    t1.append(output)
                elif output.shape[-1] == 9*4:
                    b.append(output.reshape(1,-1,4))
                elif output.shape[-1] == 9*6:
                    r.append(output.reshape(1,-1,6))

            c = sorted(c, key=lambda x: -np.prod(x.shape))
            b = sorted(b, key=lambda x: -np.prod(x.shape))
            r = sorted(r, key=lambda x: -np.prod(x.shape))
            t0 = sorted(t0, key=lambda x: -np.prod(x.shape))
            t1 = sorted(t1, key=lambda x: -np.prod(x.shape))
            tr = [np.concatenate([a,b], axis=-1).reshape(1,-1,3) for a,b in zip(t1,t0)]

            classification = np.concatenate(c,axis=1)
            translation = np.concatenate(tr, axis=1)
            bbox_prediction = np.concatenate(b,axis=1)
            rotation_raw = np.concatenate(r,axis=1)

        cls_argmax = np.argmax(classification.squeeze())
        bbox_pr = raw_output_to_bbox(bbox_prediction[0,cls_argmax], (768,512), anchors[cls_argmax])
        R_pr = rot_6d_to_matrix(rotation_raw[0,cls_argmax,:])
        t_pr = delta_xy_tz_to_translation(translation[0,cls_argmax], (Nu,Nv), (768,512), anchors[cls_argmax], strides[cls_argmax], cameraMatrix)


    if bbox_pr is None:
        print('No spacecraft detected')
    else:
        output_img = cv2.resize(cv2.imread(args.image), (768,512))

        # Draw bounding box
        [xmin, ymin, xmax, ymax] = bbox_pr
        p1 = (int(xmin), int(ymin))
        p2 = (int(xmax), int(ymax))
        cv2.rectangle(output_img, p1, p2, dataset.coco91_colors[0], 1)

        # Project 3D keypoints to 2D
        keypoints = project_keypoints(R_pr, t_pr, cameraMatrix, distCoeffs, tangoKeypoints)
        # Draw the projected pose wireframe.
        draw_pose(output_img, keypoints, scale_x=((768 / 1920)), scale_y=((512 / 1200)), thickness=2)
        cv2.imwrite('output.png', output_img)
