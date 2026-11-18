"""YuNet (face_detection_yunet_2023mar) postprocess.

The tutorial graph is cut at the twelve head conv outputs, *before* the ONNX
Sigmoid/Transpose/Reshape tail, so each map arrives as a flattened [channel][y][x]
plane of raw logits (cls/obj) or raw regression values (bbox/kps).

YuNet's decode differs from SCRFD's in three ways that matter:
  * one anchor per pixel, not two;
  * bbox is (dx, dy, log_w, log_h) about the cell centre, not an l/t/r/b distance;
  * the score is sqrt(sigmoid(cls) * sigmoid(obj)) -- obj is the discriminative
    term, cls alone fires on roughly a quarter of the stride-8 cells.
"""
import numpy as np

MAP_STRIDES = [8, 16, 32]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def calcIouLTRB(A, B):
    # A and B are of the format [left, top, right, bottom]
    left = max(A[0], B[0])
    top = max(A[1], B[1])
    right = min(A[2], B[2])
    bottom = min(A[3], B[3])
    intersect = max(0, right - left) * max(0, bottom - top)
    union = (A[2] - A[0]) * (A[3] - A[1]) + (B[2] - B[0]) * (B[3] - B[1]) - intersect
    return intersect / union if union > 0 else 0


def order_outputs(outputs):
    """Order raw model outputs into the layout yunet() expects.

    Accepts NCHW arrays in any order and returns flattened maps as
    cls_8/16/32, obj_8/16/32, bbox_8/16/32, kps_8/16/32 (or obj/bbox/kps when the
    cls head was cut from the graph).

    Grouping is by channel count -- 1 = cls/obj, 4 = bbox, 10 = kps -- which is
    unambiguous.  Sorting by shape tuple, as the SCRFD path does, relies on the
    channel and stride values not colliding.

    cls and obj are both 1-channel and so indistinguishable by shape; the two
    stride-ordered triples may therefore come back swapped.  That is harmless
    because the score sqrt(cls*obj) is commutative -- what matters is only that
    each triple is stride-ordered so the pairing per stride is correct.
    """
    ones, fours, tens = [], [], []
    for out in outputs:
        c = out.shape[-3]
        if c not in (1, 4, 10):
            raise ValueError("unexpected yunet output with %d channels" % c)

        # Map the an output based on the value of c to the appropriate  [c]
        # c = 1, 4, or 10
        # [c] = ones, fours, tens
        {1: ones, 4: fours, 10: tens}[c].append(out)

    if len(fours) != 3 or len(tens) != 3:
        raise ValueError("expected 3 bbox and 3 kps maps, got %d and %d" % (len(fours), len(tens)))
    if len(ones) not in (3, 6):
        raise ValueError("expected 3 (obj only) or 6 (cls+obj) 1-channel maps, got %d" % len(ones))

    # descending width == ascending stride (8, 16, 32); sort is stable, so the two
    # same-width 1-channel maps keep their relative order and de-interleave cleanly
    for group in (ones, fours, tens):
        group.sort(key=lambda o: -o.shape[-1])
    if len(ones) == 6:
        ones = ones[0::2] + ones[1::2]

    return [np.asarray(o).flatten() for o in ones + fours + tens]


def yunet(modelOut, inputWidth, inputHeight, detectThresh=0.6, maxIou=0.2,
          maxPreDetects=500, presigmoid=True):
    """modelOut is 12 flattened maps, ordered cls_8/16/32, obj_8/16/32,
    bbox_8/16/32, kps_8/16/32.  Pass 9 maps (cls omitted) to score on obj alone.

    presigmoid=True when the graph was cut before the Sigmoid nodes.
    Returns a list of {'box': [l,t,r,b], 'landmarks': 5x2, 'score': float}.
    """
    assert inputHeight % 32 == 0 and inputWidth % 32 == 0
    h32, w32 = inputHeight // 32, inputWidth // 32
    mapSizes = [[4 * h32, 4 * w32], [2 * h32, 2 * w32], [h32, w32]]
    mapPixels = np.array([h * w for h, w in mapSizes])

    if len(modelOut) == 12:
        clsMaps, objMaps = modelOut[0:3], modelOut[3:6]
        locMaps, landMaps = modelOut[6:9], modelOut[9:12]
    elif len(modelOut) == 9:
        # cls head dropped from the graph; obj carries the whole score
        clsMaps = None
        objMaps, locMaps, landMaps = modelOut[0:3], modelOut[3:6], modelOut[6:9]
    else:
        raise ValueError("expected 12 (or 9) yunet output maps, got %d" % len(modelOut))

    def activate(maps):
        return [sigmoid(np.asarray(m, dtype=np.float64)) if presigmoid
                else np.clip(np.asarray(m, dtype=np.float64), 0, 1) for m in maps]

    obj = activate(objMaps)
    if clsMaps is None:
        scoreMaps = obj
    else:
        cls = activate(clsMaps)
        scoreMaps = [np.sqrt(c * o) for c, o in zip(cls, obj)]

    scores = np.concatenate(scoreMaps)
    order = scores.argsort()[::-1][:maxPreDetects]
    order = order[scores[order] > detectThresh]

    faces = []
    for ind in order:
        detectScore = scores[ind]

        # scores index -> (map, pixel).  One anchor per pixel.
        mapNum = 0
        while mapNum < 2 and ind >= mapPixels[mapNum]:
            ind -= mapPixels[mapNum]
            mapNum += 1
        pixels = mapPixels[mapNum]
        mapH, mapW = mapSizes[mapNum]
        y, x = ind // mapW, ind % mapW
        stride = MAP_STRIDES[mapNum]

        # bbox: centre offset in cells, size as log-scale multiples of stride
        loc = np.asarray(locMaps[mapNum][ind::pixels][:4], dtype=np.float64)
        cx = (x + loc[0]) * stride
        cy = (y + loc[1]) * stride
        w = np.exp(loc[2]) * stride
        h = np.exp(loc[3]) * stride
        box = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

        if any(calcIouLTRB(f['box'], box) > maxIou for f in faces):
            continue

        land = np.asarray(landMaps[mapNum][ind::pixels][:10], dtype=np.float64)
        keypoints = np.array([[(x + land[2 * p]) * stride,
                               (y + land[2 * p + 1]) * stride] for p in range(5)])

        faces.append({'box': box, 'landmarks': keypoints, 'score': detectScore})

    return faces
