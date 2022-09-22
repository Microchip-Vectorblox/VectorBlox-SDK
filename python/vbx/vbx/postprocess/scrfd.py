# import vbx.sim
# import openvino.inference_engine as ie
# import argparse
# import os
import numpy as np
import cv2
import json
# from  vbx.postprocess.blazeface import blazeface
from math import ceil
from itertools import product as product



    
def calcIouLTRB(A,B):
    # A and B are of the format [left, top, right, bottom]
    left = max(A[0],B[0])
    top = max(A[1],B[1])
    right = min(A[2],B[2])
    bottom = min(A[3],B[3])
    intersect = max(0,right-left) * max(0,bottom-top)
    union = (A[2]-A[0])*(A[3]-A[1]) + (B[2]-B[0])*(B[3]-B[1]) - intersect
    if union>0:
        return intersect/union
    else:
        return 0
    
def scrfd(modelOut, inputWidth, inputHeight, detectThresh=0.75, maxIou=0.35):
    mapStrides = [8,16,32]
    maxPreDetects = 64
    
    assert(inputHeight % 32 == 0)
    assert(inputWidth % 32 == 0)
    h32 = (inputHeight//32) # image height at stride=32
    w32 = (inputWidth//32)  # image width at stride=32
    h8,h16 = 4*h32, 2*h32
    w8,w16 = 4*w32, 2*w32
    mapSizes = [[h8,w8],[h16,w16],[h32,w32]]
    mapPixels = np.array([h8*w8, h16*w16, h32*w32])

    # each map shape is [anchor][channel][y][x]
    #   there are 2 anchors per pixel
    #   location map has 4 channels; confidence has 2 channels; landmarks has 10 channels
    #   the range of x and y pixel are given by mapSizes[...]
    confMaps = [modelOut[0], modelOut[1], modelOut[2]]
    locMaps = [modelOut[3], modelOut[4], modelOut[5]]
    landMaps = [modelOut[6], modelOut[7], modelOut[8]]
    
    scores = np.concatenate(confMaps)
        
    # alternatively, this argsort could be moved inside the scores loop above
    #   this would reduce memory, but complicate the code and make optimization/vectorization more difficult
    order = scores.argsort()[::-1][:maxPreDetects]
    order = order[scores[order]>detectThresh]
    
    faces = []
    for n,ind in enumerate(order):
        detectScore = scores[ind]
        # scores index to other indices
        mapNum = 0
        if ind>=2*mapPixels[0]:
            ind -= 2*mapPixels[0]
            mapNum += 1
            if ind>2*mapPixels[1]:
                ind -= 2*mapPixels[1]
                mapNum += 1
        # get anchor per pixel
        pixels = mapPixels[mapNum]
        anchNum = 0
        if ind >= pixels:
            anchNum = 1
            ind = ind - pixels
        # get pixel
        mapSize = mapSizes[mapNum]
        y = ind//mapSize[1]
        x = ind - y*mapSize[1]
        # get prior
        stride = mapStrides[mapNum]
        anchor = np.array([x, y]) * stride # anchor center, units of pixels
        
        # get location data
        location = locMaps[mapNum][anchNum*4*pixels+ind::pixels][:4] * stride
        box = np.concatenate((anchor - location[:2], anchor + location[2:]))
        # [left, top, right, bottom]
        
        # NMS
        passNms = True
        for face in faces:
            if calcIouLTRB(face['box'],box)>maxIou:
                passNms = False
                break
        if not passNms:
            continue
        
        # get landmarks
        landmark = landMaps[mapNum][anchNum*10*pixels+ind::pixels][:10] * stride
        keypoints = np.array([anchor + landmark[:2],
                              anchor + landmark[2:4],
                              anchor + landmark[4:6],
                              anchor + landmark[6:8],
                              anchor + landmark[8:10]])
        
        # add detection to list of faces
        faces.append({'box':box,'landmarks':keypoints,'score':detectScore})
    
    return faces
    

    



if __name__ == "__main__":
    main()
