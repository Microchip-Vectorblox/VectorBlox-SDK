import numpy as np
import cv2
import os
import argparse
import math

import vbx.sim
from vbx.generate.utils import onnx_infer, onnx_input_shape
from vbx.generate.utils import load_input

import vbx.postprocess.dataset as dataset
import vbx.postprocess.classifier as classifier


colors = dataset.coco_colors

KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): (147, 20, 255),
    (0, 2): (255, 255, 0),
    (1, 3): (147, 20, 255),
    (2, 4): (255, 255, 0),
    (0, 5): (147, 20, 255),
    (0, 6): (255, 255, 0),
    (5, 7): (147, 20, 255),
    (7, 9): (147, 20, 255),
    (6, 8): (255, 255, 0),
    (8, 10): (255, 255, 0),
    (5, 6): (0, 255, 255),
    (5, 11): (147, 20, 255),
    (6, 12): (255, 255, 0),
    (11, 12): (0, 255, 255),
    (11, 13): (147, 20, 255),
    (13, 15): (147, 20, 255),
    (12, 14): (255, 255, 0),
    (14, 16): (255, 255, 0)
}


def draw_masks(image, boxes, class_ids, mask_alpha=0.3, mask_maps=None):
    mask_img = image.copy()

    # Draw bounding boxes and labels of detections
    for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # Draw fill mask image
        if mask_maps is None:
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
        else:
            crop_mask = mask_maps[i][y1:y2, x1:x2, np.newaxis]
            crop_mask_img = mask_img[y1:y2, x1:x2]
            crop_mask_img = crop_mask_img * (1 - crop_mask) + crop_mask * color
            mask_img[y1:y2, x1:x2] = crop_mask_img

    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)


def draw_keypoints_on_image(image, scale, keypoints, t):
    for edge, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        point1_index, point2_index = edge
        x1, y1, _ = keypoints[point1_index]
        x2, y2, _ = keypoints[point2_index]
        x1, y1, x2, y2 = int(x1*scale[1]), int(y1*scale[0]), int(x2*scale[1]), int(y2*scale[0])
            
        # Draw the line on the image
        cv2.line(image, (x1, y1), (x2, y2), color, 2)
        cv2.circle(image, (x1, y1), 4, color, -1) 

    return image


def draw_bbox_on_image(image, scale, x, y, w, h, color=[255,0,255], label=None, is_xywh=True):
    # Denormalize the coordinates
    x = int(x * scale[1])
    y = int(y * scale[0])
    w = int(w * scale[1])
    h = int(h * scale[0])

    # Draw the bounding box
    if is_xywh:
        p1, p2 = (x-w//2, y-h//2), (x+w//2, y+h//2)
    else:
        p1, p2 = (x, y), (w, h)
    cv2.rectangle(image, p1, p2, color, 2)

    if not label is None:
        p3 = (max(p1[0]-4, 4), max(p1[1]-4, 4))
        cv2.putText(image, label, p3, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    return image


def post_process(output, confidence_thres=0.5, iou_thres=0.5, num_classes=80):
    # Transpose and squeeze the output to match the expected shape
    outputs = np.transpose(np.squeeze(output))

    # Get the number of rows in the outputs array
    rows = outputs.shape[0]

    # Lists to store the bounding boxes, scores, and class IDs of the detections
    boxes = []
    scores = []
    class_ids = []

    # Calculate the scaling factors for the bounding box coordinates
    x_factor = 1.
    y_factor = 1.

    # Iterate over each row in the outputs array
    for i in range(rows):
        # Extract the class scores from the current row
        classes_scores = outputs[i][4:4+num_classes]

        # Find the maximum score among the class scores
        max_score = np.amax(classes_scores)

        # If the maximum score is above the confidence threshold
        if max_score >= confidence_thres:
            # Get the class ID with the highest score
            class_id = np.argmax(classes_scores)

            # Extract the bounding box coordinates from the current row
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

            # Calculate the scaled coordinates of the bounding box
            x = int(x * x_factor)
            y = int(y * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)

            # Add the class ID, score, and box coordinates to the respective lists
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([x, y, width, height])

    # Apply non-maximum suppression to filter out overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)

    # Iterate over the selected indices after non-maximum suppression
    valid_results = []
    for i in indices:
        # Get the box, score, and class ID corresponding to the index
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]

        valid_results.append((box, score, class_id))
    return valid_results


def postprocess_classifier(output, img):
    output = np.squeeze(output)
    sorted_classes = classifier.topk(output)

    if len(output)==1001:
        classes = dataset.imagenet_classes_with_nul
    elif len(output)==1000:
        classes = dataset.imagenet_classes
    else: # mnist
        classes = None

    i = 0
    output_img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    for cls, score in list(zip(*sorted_classes))[:5]:
        p3 = (4, (i+1)*(32+4))
        if classes is None:
            cv2.putText(output_img, '{}'.format(cls), p3,
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            class_name = classes[cls]
            short_name = class_name.split(',')[0]
            cv2.putText(output_img, '{} {}'.format(cls, short_name), p3,
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        i += 1

    cv2.imwrite('class.png', output_img)


def post_process_detection(output, img, input_height, input_width, threshold, iou, labels=None, scale_by_input_dims=False):
    if scale_by_input_dims:
        output[:,0] *= input_width
        output[:,2] *= input_width
        output[:,1] *= input_height
        output[:,3] *= input_height

    valid_results = post_process(output, threshold, iou)

    output_img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    scale =  1024. / input_height, 1024. / input_width

    for box, score, class_id in valid_results:
        x, y, w, h = box

        short_name = str(class_id)
        color = [255,0,255]
        if not labels is None:
            class_name = labels[class_id]
            short_name = class_name.split(',')[0]
            color = colors[class_id]

        draw_bbox_on_image(output_img, scale, x, y, w, h, color, short_name)

    cv2.imwrite('output.png', output_img)


def postprocess_pose(output, img, input_height, input_width, conf, iou, scale_input_dims=False, topk=1):
    output_t = output[0].T
    sorted_indices = np.argsort(output_t[:, 4])[::-1]
    # do nms not topk
    output_topk = output_t[sorted_indices[:topk]]

    output_img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    scale = 1024. / input_height, 1024. / input_width

    kpt_shape = (17, 3) #17 points of x,y pairs + visibility score
    for r in output_topk:
        xywh, score, keypoints = r[:4], r[4], r[5:].reshape(kpt_shape)
        if score > conf:
            output_img = draw_bbox_on_image(output_img, scale, xywh[0], xywh[1], xywh[2], xywh[3], label="person")
            output_img = draw_keypoints_on_image(output_img, scale, keypoints, 0.7)
    cv2.imwrite('pose.png', output_img)


def xywhr2xyxyxyxy(rbox):
    cos, sin = (np.cos, np.sin)

    ctr = rbox[:2]
    w, h, angle = (rbox[i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = np.concatenate(vec1, axis=-1)
    vec2 = np.concatenate(vec2, axis=-1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return np.asarray([pt1, pt2, pt3, pt4])


def postprocess_obb(outputs, img, input_height, input_width, conf, iou, labels=None, num_classes=15, scale_input_dims=False):
    output_img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    scale = 1024. / input_width, 1024. / input_height

    predictions = np.squeeze(outputs[0]).T

    scores = np.max(predictions[:, 4:4+num_classes], axis=1) # get max score
    predictions = predictions[scores > conf, :] # shrink predictions to > threshold
    scores = scores[scores > conf] # shrink scores to > threshold

    if len(scores) == 0:
        print('no object found')
        return

    # add nms_rotated 
    boxes = np.concatenate([predictions[..., :4], predictions[...,-1:]], axis=-1) #xywh + r  (first 4 dims, and last dims concatentated)
    class_ids = np.argmax(predictions[:, 4:num_classes+4], axis=1)
    for box, score, class_id in zip(boxes, scores, class_ids):
        p1, p2, p3, p4 = xywhr2xyxyxyxy(box)

        color = colors[class_id]
        for s,e in [(p1,p2), (p2,p3), (p3,p4), (p4,p1)]:
            sx = int(np.clip(s[0], 0, input_width))
            sy = int(np.clip(s[1], 0, input_height))
            ex = int(np.clip(e[0], 0, input_width))
            ey = int(np.clip(e[1], 0, input_height))
            cv2.line(output_img, (sx,sy), (ex,ey), color, 2)


    cv2.imwrite('obb.png', output_img)


def postprocess_instance_segmentation(outputs, img, input_height, input_width, conf, iou, labels=None, scale_input_dims=False):
    output_img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    scale = 1024. / input_width, 1024. / input_height

    boxes, scores, class_ids, mask_box_scores = process_box_output(outputs[0], conf, iou)
    boxes = rescale_boxes(boxes, (input_height, input_width), (output_img.shape[0], output_img.shape[1]))

    mask_maps = process_mask_output(mask_box_scores, outputs[1], boxes, output_img.shape[0], output_img.shape[1])

    output_img = draw_masks(output_img, boxes, class_ids, 0.3, mask_maps)

    for box, score, class_id in zip(boxes, scores, class_ids):
        x, y, w, h = box

        short_name = str(class_id)
        color = [255,0,255]
        if not labels is None:
            class_name = labels[class_id]
            short_name = class_name.split(',')[0]
            color = colors[class_id]

        output_img = draw_bbox_on_image(output_img, (1.,1.), x, y, w, h,color, short_name, is_xywh=False)

    cv2.imwrite('seg.png', output_img)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def process_mask_output(mask_predictions, mask_output, boxes, img_height, img_width):
    if mask_predictions.shape[0] == 0:
        return []

    mask_output = np.squeeze(mask_output)

    # Calculate the mask maps for each box
    num_mask, mask_height, mask_width = mask_output.shape  # CHW
    masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
    masks = masks.reshape((-1, mask_height, mask_width))

    # Downscale the boxes to match the mask size
    scale_boxes = rescale_boxes(boxes, (img_height, img_width), (mask_height, mask_width))

    # For every box/mask pair, get the mask map
    mask_maps = np.zeros((len(scale_boxes), img_height, img_width))
    blur_size = (int(img_width / mask_width), int(img_height / mask_height))
    for i in range(len(scale_boxes)):

        scale_x1 = int(math.floor(scale_boxes[i][0]))
        scale_y1 = int(math.floor(scale_boxes[i][1]))
        scale_x2 = int(math.ceil(scale_boxes[i][2]))
        scale_y2 = int(math.ceil(scale_boxes[i][3]))

        x1 = int(math.floor(boxes[i][0]))
        y1 = int(math.floor(boxes[i][1]))
        x2 = int(math.ceil(boxes[i][2]))
        y2 = int(math.ceil(boxes[i][3]))

        scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
        crop_mask = cv2.resize(scale_crop_mask,
                          (x2 - x1, y2 - y1),
                          interpolation=cv2.INTER_CUBIC)

        crop_mask = cv2.blur(crop_mask, blur_size)

        crop_mask = (crop_mask > 0.5).astype(np.uint8)
        mask_maps[i, y1:y2, x1:x2] = crop_mask

    return mask_maps


def rescale_boxes(boxes, input_shape, image_shape):
        # Rescale boxes to original image dimensions
        input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])

        return boxes


def process_box_output(box_output, conf_threshold, iou_threshold, num_masks=32):

    predictions = np.squeeze(box_output).T
    num_classes = box_output.shape[1] - num_masks - 4

    # Filter out object confidence scores below threshold
    scores = np.max(predictions[:, 4:4+num_classes], axis=1)
    predictions = predictions[scores > conf_threshold, :]
    scores = scores[scores > conf_threshold]

    if len(scores) == 0:
        return [], [], [], np.array([])

    box_predictions = predictions[..., :num_classes+4] # xywh+class_scores
    mask_predictions = predictions[..., num_classes+4:] # class_mask_scores

    # Get the class with the highest confidence
    class_ids = np.argmax(box_predictions[:, 4:], axis=1)

    # Get bounding boxes for each object
    boxes = box_predictions[:, :4]
    boxes = xywh2xyxy(boxes)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    indices = nms(boxes, scores, iou_threshold)

    return boxes[indices], scores[indices], class_ids[indices], mask_predictions[indices]


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    parser.add_argument('--task', choices=['detect', 'classify', 'pose', 'seg', 'obb'], default='detect')
    parser.add_argument('-l', '--labels')
    parser.add_argument('-t', '--threshold', type=float, default=0.5)
    parser.add_argument('-i', '--iou', type=float, default=0.5)
    parser.add_argument('-sc', '--scale', type=float, nargs='+', default=[1.])
    parser.add_argument('-b', '--bgr', action='store_true')
    parser.add_argument('-si', '--scale_inputs', action='store_true')

    args = parser.parse_args()
    img = cv2.imread(args.image)

    if '.onnx' in args.model:
        input_shape = onnx_input_shape(args.model)[0]
        input_height, input_width = input_shape[-2], input_shape[-1]

        arr = load_input(args.image, 1./ args.scale[0], input_shape, (not args.bgr))
        outputs = onnx_infer(args.model, arr)
    elif '.vnnx' in args.model:
        with open(args.model, 'rb') as mf:
            model = vbx.sim.Model(mf.read())

        input_shape = model.input_shape[0]
        input_height, input_width = input_shape[-2], input_shape[-1]
        if img.shape != (input_height, input_width, 3):
            img = cv2.resize(img, (input_width, input_height)).clip(0, 255)
        img_resized = img.astype(np.float32)
        if not args.bgr:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_resized = img_resized / args.scale
        input_scale, input_zero_point = model.input_scale_factor[0], model.input_zeropoint[0]
        if  input_scale != 0.0:
            img_resized = (img_resized / input_scale) + input_zero_point
        img_resized = img_resized.astype(model.input_dtypes[0])
        flattened = img_resized.swapaxes(1, 2).swapaxes(0, 1).flatten()
        outputs = model.run([flattened])
        for o, output in enumerate(outputs):
            outputs[o] = model.output_scale_factor[o] * (outputs[o].astype(np.float32) - model.output_zeropoint[o])
            outputs[o] = outputs[o].reshape(model.output_shape[o]).transpose((1,2,0))

    elif '.tflite' in args.model:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        interpreter= tf.lite.Interpreter(model_path=args.model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_height, input_width = input_details[0]['shape'][1], input_details[0]['shape'][2]
        if img.shape != (input_height, input_width, 3):
            img = cv2.resize(img, (input_width, input_height)).clip(0, 255)
        img_resized = img.astype(np.float32)
        if not args.bgr:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_resized = img_resized / args.scale
        img_resized = np.expand_dims(img_resized, axis=0)

        input_scale, input_zero_point = input_details[0].get('quantization', (0.0, 0))
        if  input_scale != 0.0:
            img_resized = (img_resized / input_scale) + input_zero_point
        img_resized = img_resized.astype(input_details[0]['dtype'])
        interpreter.set_tensor(input_details[0]['index'], img_resized)
        interpreter.invoke()
        outputs = []
        for o in range(len(output_details)):
            output_scale, output_zero_point = output_details[o].get('quantization', (0.0, 0))
            output = interpreter.get_tensor(output_details[o]['index'])
            if  output_scale != 0.0:
                output = output_scale * (output.astype(np.float32) - output_zero_point)
            outputs.append(output)

    labels = None
    if args.labels:
        with open(args.labels) as f:
            labels = f.readlines()
            labels = [x.strip() for x in labels]

    if args.task == 'detect':
        post_process_detection(outputs[0], img, input_height, input_width, args.threshold, args.iou, labels, args.scale_inputs)
    elif args.task == 'classify':
        postprocess_classifier(outputs[0], img, args.scale_inputs)
    elif args.task == 'pose':
        postprocess_pose(outputs[0], img, input_height, input_width, args.threshold, args.iou, args.scale_inputs)
    elif args.task == 'seg':
        postprocess_instance_segmentation(outputs, img, input_height, input_width, args.threshold, args.iou, labels, args.scale_inputs)
    elif args.task == 'obb':
        postprocess_obb(outputs[0], img, input_height, input_width, args.threshold, args.iou, labels, args.scale_inputs)
