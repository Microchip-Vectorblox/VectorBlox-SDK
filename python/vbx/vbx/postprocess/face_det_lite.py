import numpy as np


# ---------------------------------------------------------------------------
# Postprocessing for the Qualcomm "Lightweight-Face-Detection" (face_det_lite)
# model.  This is a numpy port of qai_hub_models.models.face_det_lite.utils so
# the simulator output stays bit-comparable with the reference implementation.
#
# Model outputs (stride 8, input 480x640 -> grid 60x80):
#   heatmap  : (1, 1,  H, W)  face center confidence (pre-sigmoid logits)
#   bbox     : (1, 4,  H, W)  left/top/right/bottom offsets
#   landmark : (1, 10, H, W)  5 (x,y) facial landmarks
# ---------------------------------------------------------------------------


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def calcIouLTRB(A, B):
    # A and B are of the format [left, top, right, bottom]
    # mirrors vbx.postprocess.scrfd.calcIouLTRB
    left = max(A[0], B[0])
    top = max(A[1], B[1])
    right = min(A[2], B[2])
    bottom = min(A[3], B[3])
    intersect = max(0, right - left) * max(0, bottom - top)
    union = (A[2] - A[0]) * (A[3] - A[1]) + (B[2] - B[0]) * (B[3] - B[1]) - intersect
    if union > 0:
        return intersect / union
    else:
        return 0


def detect(heatmap, box, landmark, threshold=0.55, nms_iou=-1, stride=8, max_pre_detects=64):
    """heatmap (1,1,H,W), box (1,4,H,W), landmark (1,10,H,W) -> list of faces."""
    heatmap = sigmoid(heatmap[0, 0].astype(np.float64))

    # rank every cell by score, keep the top max_pre_detects,
    # and rely on the inline NMS below to collapse
    # the cluster of high cells around each face.
    flat = heatmap.reshape(-1)
    k = min(flat.shape[0], max_pre_detects)
    indices = np.argsort(flat)[::-1][:k]
    scores = flat[indices]

    _, heatmap_width = heatmap.shape
    ys = (indices // heatmap_width).astype(np.int32)
    xs = (indices % heatmap_width).astype(np.int32)

    # detections are visited in descending score order; NMS is applied inline
    # (greedy, suppress against already-kept faces) to match scrfd.scrfd
    faces = []
    for cx, cy, score in zip(xs, ys, scores):
        if score < threshold:
            break

        x, y, r, b = box[0, :, cy, cx].astype(np.float64)
        xyrb = ((np.array([cx, cy, cx, cy], dtype=np.float64) +
                 [-x, -y, r, b]) * stride).tolist()
        # normalize to [left, top, right, bottom]
        xyrb = [min(xyrb[0], xyrb[2]), min(xyrb[1], xyrb[3]),
                max(xyrb[0], xyrb[2]), max(xyrb[1], xyrb[3])]

        # NMS
        if nms_iou != -1:
            passNms = True
            for face in faces:
                if calcIouLTRB(face['box'], xyrb) > nms_iou:
                    passNms = False
                    break
            if not passNms:
                continue

        x5y5 = landmark[0, :, cy, cx].astype(np.float64)
        x5y5 = (x5y5 + ([cx] * 5 + [cy] * 5)) * stride
        landmarks = list(zip(x5y5[:5], x5y5[5:]))

        faces.append({'box': xyrb, 'score': float(score), 'landmarks': landmarks})

    return faces