import numpy as np
import cv2
import os
import json
import onnxruntime

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calcIouXYWH(A,B):
    # A and B are of the format [x center, y center, width, height]
    left = max(A[0]-A[2]/2,B[0]-B[2]/2)
    right = min(A[0]+A[2]/2,B[0]+B[2]/2)
    top = max(A[1]-A[3]/2,B[1]-B[3]/2)
    bottom = min(A[1]+A[3]/2,B[1]+B[3]/2)
    intersect = max(0,right-left) * max(0,bottom-top)
    union = A[2]*A[3] + B[2]*B[3] - intersect
    if union>0:
        return intersect/union
    else:
        return 0

def xywhssToKps(xywhss):
    x, y, w, h, sx, sy = xywhss
    k  = np.array([ x,  y,  x,  y,  x,  y,  x,  y])
    k += np.array([-w, -h,  w, -h, -w,  h,  w,  h])*0.5
    k += np.array([sx, sy, sx,-sy,-sx, sy,-sx,-sy])*0.5
    return k


def ltrbssToKps(ltrbss):
    l, t, r, b, sx, sy = ltrbss
    k = np.array([  l,  t,  r,   t,   l,  b,   r,   b])
    k += np.array([sx, sy, sx, -sy, -sx, sy, -sx, -sy]) * 0.5
    return k

def kpsToXywh(kps):
    l = np.min(kps[0::2])
    r = np.max(kps[0::2])
    t = np.min(kps[1::2])
    b = np.max(kps[1::2])
    x = 0.5 * (l + r)
    y = 0.5 * (t + b)
    w = r - l
    h = b - t
    return np.array([x, y, w, h])

def postprocess_lpd(modelOutput,detectH, detectW, detectThresh, maxIou):
    mapStrides = [32,16,8]
    maxPreDetects = 64    
    assert(detectH % 32 == 0)
    assert(detectW % 32 == 0)
    h32 = (detectH//32) # image height at stride=32
    w32 = (detectW//32)  # image width at stride=32   
    h16 = h32<<1
    w16 = w32<<1
    h8 = h16<<1
    w8 = w16<<1
    lpr_anchors = [[180.0,52.0], [60.0,18.0], [20.0,7.0]]
    mapSizes = [[h32,w32],[h16,w16],[h8,w8]]
    mapPixels = [h32*w32, h16*w16, h8*w8]
    shapedim_y = [9,18,36]
    shapedim_x = [32,64,128]

    objMaps = [modelOutput[0],modelOutput[2],modelOutput[4]]
    shapeOutputs = [modelOutput[1],modelOutput[3],modelOutput[5]]
    #reshaping shapeOutputs
    m=0
    shapes = []
    for shape in shapeOutputs:
        tempshape = shape.reshape(1,6,shapedim_y[m],shapedim_x[m])
        shapes.append(tempshape)
        m+=1

    scores = np.array([])
    for objOut in objMaps:
        outputScores = objOut.flatten()
        scores = np.concatenate((scores, outputScores))
    order = scores.argsort().astype('int32')[::-1][:maxPreDetects]
    order = order[scores[order] > detectThresh]



# each output layer is [anchor][channel][y][x]
    objs = []
    for ind in order:
        detectScore = scores[ind]  # omit sigmoid()

        # get map (which output layer)
        mapNum = 0
        while ind >= mapPixels[mapNum]:
            ind -= mapPixels[mapNum]
            mapNum += 1
        # get anchor
        pixels = mapPixels[mapNum]
        anchNum = 0
        while ind >= pixels:
            ind -= pixels
            anchNum += 1
        # get pixel x,y 
        y = ind // mapSizes[mapNum][1]
        x = ind - y * mapSizes[mapNum][1]

        raw = shapes[mapNum][anchNum,:,y,x]

        # shape [center x, center y, width, height, shear x, shear y]
        xywhss = np.zeros((6))
        xywhss[0] = (sigmoid(raw[0])*2 + x - .5) * mapStrides[mapNum]
        xywhss[1] = (sigmoid(raw[1])*2 + y - .5) * mapStrides[mapNum]
        xywhss[2] = ((sigmoid(raw[2])*2) ** 2) * lpr_anchors[mapNum][anchNum*2]
        xywhss[3] = ((sigmoid(raw[3])*2) ** 2) * lpr_anchors[mapNum][anchNum*2+1]
        xywhss[4] = raw[4] * 0.25 * mapStrides[mapNum]
        xywhss[5] = raw[5] * mapStrides[mapNum]
        kps = xywhssToKps(xywhss)
        box = kpsToXywh(kps)

        # NMS
        passNms = True
        for obj in objs:
            if calcIouXYWH(obj['box'], box) > maxIou:
                passNms = False
                break
        if not passNms:
            continue

        # add detection to list of objects
        objs.append({'box':box, 'kps':kps, 'detectScore':detectScore, 'stride':mapStrides[mapNum], 'anchNum':anchNum})

    return objs 