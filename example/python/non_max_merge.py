import numpy as np


def box_iou_batch(boxes_true, boxes_detection):
    """
    Compute Intersection over Union (IoU) of two sets (x_min, y_min, x_max, y_max) boxes

        boxes_true (np.ndarray): 2D `np.ndarray` representing ground-truth boxes.
            `shape = (N, 4)` where `N` is number of true objects.
        boxes_detection (np.ndarray): 2D `np.ndarray` representing detection boxes.
            `shape = (M, 4)` where `M` is number of detected objects.

    Returns:
        np.ndarray: Pairwise IoU of boxes from `boxes_true` and `boxes_detection`.
            `shape = (N, M)` where `N` is number of true objects and
            `M` is number of detected objects.
    """

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area_true = box_area(boxes_true.T)
    area_detection = box_area(boxes_detection.T)

    top_left = np.maximum(boxes_true[:, None, :2], boxes_detection[:, :2])
    bottom_right = np.minimum(boxes_true[:, None, 2:], boxes_detection[:, 2:])

    area_inter = np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)
    ious = area_inter / (area_true[:, None] + area_detection - area_inter)
    ious = np.nan_to_num(ious)
    return ious


def group_overlapping_boxes(predictions, iou_threshold = 0.5):
    """
    Apply greedy version of non-maximum merging to avoid detecting too many
    overlapping bounding boxes for a given object.

            [x1, y1, x2, y2, score]
    Return merge group, where each group may have 1 or more elements.
    """
    merge_groups = []

    scores = predictions[:, 4]
    order = scores.argsort() #least -> most 

    while len(order) > 0:
        idx = int(order[-1]) #top confident

        order = order[:-1] #most -> least 
        if len(order) == 0:
            merge_groups.append([idx])
            break

        ious = []
        rorder = np.flip(order) #least -> most

        merge_group = [idx]
        for r,prediction in zip(rorder, predictions[rorder]):
            if calc_box_iou(prediction[:4], predictions[idx][:4]) > iou_threshold:
                merge_group.append(r)

        remaining = []
        for o in order:
            if o not in merge_group:
                remaining.append(o)

        merge_groups.append(merge_group)
        order = remaining

        '''
        merge_candidate = np.expand_dims(predictions[idx], axis=0)
        ious = box_iou_batch(predictions[order][:, :4], merge_candidate[:, :4])
        ious = ious.flatten()

        above_threshold = ious >= iou_threshold
        merge_group = [idx, *np.flip(order[above_threshold]).tolist()]
        merge_groups.append(merge_group)

        order = order[~above_threshold]
        '''
    return merge_groups


def box_non_max_merge(predictions, iou_threshold=0.5):
    """
    Apply greedy version of non-maximum merging per category to avoid detecting
    too many overlapping bounding boxes for a given object.
    
    Return groups of prediction indices to be merged (of 1+ elements).
    """
    if predictions.shape[1] == 5:
        return group_overlapping_boxes(predictions, iou_threshold)

    category_ids = predictions[:, 5]
    merge_groups = []
    for category_id in np.unique(category_ids):
        curr_indices = np.where(category_ids == category_id)[0]
        merge_class_groups = group_overlapping_boxes(
            predictions[curr_indices], iou_threshold
        )

        for merge_class_group in merge_class_groups:
            merge_groups.append(curr_indices[merge_class_group].tolist())

    for merge_group in merge_groups:
        if len(merge_group) == 0:
            raise ValueError(
                f"Empty group detected when non-max-merging "
                f"detections: {merge_groups}"
            )
    return merge_groups

def calc_box_iou(box1, box2):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])

    # Compute intersection area
    intersection_area = max(0, xmax - xmin) * max(0, ymax - ymin)

    # Compute union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def merge_inner_detection_objects(detections, threshold=0.5):

    """
    Given N detections into a single detection
    """
    xyxy_1, keypoints_1, conf_1, class_1 = detections[0]
    for xyxy_2, keypoints_2, conf_2, class_2 in detections[1:]:
        box_iou = calc_box_iou(xyxy_1, xyxy_2)
        if box_iou < threshold:
            break
        xyxy_1, keypoints_1, conf_1, class_1 = merge_inner_detection_object_pair(xyxy_1, keypoints_1, conf_1, class_1, xyxy_2, keypoints_2, conf_2, class_2)
    return xyxy_1, keypoints_1, conf_1, class_1


def merge_inner_detection_object_pair(xyxy_1, keypoints_1, conf_1, class_1, xyxy_2, keypoints_2, conf_2, class_2):
    """
    Merges two detections into a one
    """

    if conf_1 is None and conf_2 is None:
        merged_confidence = None
    else:
        detection_1_area = (xyxy_1[2] - xyxy_1[0]) * (xyxy_1[3] - xyxy_1[1])
        detections_2_area = (xyxy_2[2] - xyxy_2[0]) * (xyxy_2[3] - xyxy_2[1])
        merged_confidence = (
            detection_1_area * conf_1
            + detections_2_area * conf_2
        ) / (detection_1_area + detections_2_area)

    if keypoints_1 is None and keypoints_2 is None:
        merged_keypoints = None
    else:
        detection_1_area = (xyxy_1[2] - xyxy_1[0]) * (xyxy_1[3] - xyxy_1[1])
        detections_2_area = (xyxy_2[2] - xyxy_2[0]) * (xyxy_2[3] - xyxy_2[1])
        merged_keypoints = (
            detection_1_area * keypoints_1
            + detections_2_area * keypoints_2
        ) / (detection_1_area + detections_2_area)

    merged_x1, merged_y1 = np.minimum(xyxy_1[:2], xyxy_2[:2])
    merged_x2, merged_y2 = np.maximum(xyxy_1[2:], xyxy_2[2:])
    merged_xyxy = np.array([merged_x1, merged_y1, merged_x2, merged_y2])

    merged_class = class_1
    if conf_1 is None and conf_2 is None:
        pass
    elif conf_1 >= conf_2:
        merged_class = class_1
    else:
        merged_class = class_2

    return merged_xyxy, merged_keypoints, merged_confidence, merged_class

