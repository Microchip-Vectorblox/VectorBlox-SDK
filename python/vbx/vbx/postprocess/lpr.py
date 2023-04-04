
import numpy as np
import cv2
import os
import json

CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-']
MAX_PLATE_LETTERS = 9
MIN_PLATE_LETTERS = 4


def PlateDecodeIndices(indices):
    label = list()
    prev = -1
    for n in indices:
        if n != prev and n != len(CHARS)-1:
            label.append(n)
        prev = n    
    PlateID = ""
    for c in label:
        PlateID += CHARS[c]
    return PlateID

# returns: label, confidence
def PlateDecodeCStyle(raw):
    COLS = raw.shape[1]
    x = raw.copy()      # this gets modified, so make a local copy

    maxVal = -1000*np.ones(COLS)        # max value for each column
    maxInd = np.zeros(COLS, dtype=int)  # index of the max value for each column
    secVal = -1000*np.ones(COLS)        # second highest value for each column
    secInd = np.zeros(COLS, dtype=int)  # index of the second highest value for each column
    for c in range(COLS):
        for i in range(len(CHARS)):
            if x[i, c] > maxVal[c]:
                secVal[c] = maxVal[c]
                secInd[c] = maxInd[c]
                maxVal[c] = x[i, c]
                maxInd[c] = i
            elif x[i, c] > secVal[c]:
                secVal[c] = x[i, c]
                secInd[c] = i

    label = PlateDecodeIndices(maxInd)
    if (len(label) > MAX_PLATE_LETTERS) or (len(label) < MIN_PLATE_LETTERS):    # these should be made constants in C
        return label, 0.0

    diffVal = np.zeros(COLS)    # difference between highest two values for each column
    for c in range(COLS):
        diffVal[c] = maxVal[c] - secVal[c]

    while 1:
        testInd = maxInd.copy()
        minDiff = diffVal[0]
        minCol = 0
        for c in range(1, COLS):
            if diffVal[c] < minDiff:
                minDiff = diffVal[c]        # find the minimum diffVal, which is the most likely error
                minCol = c

        testInd[minCol] = secInd[minCol]        # altered version of max indices
        testLabel = PlateDecodeIndices(testInd)     # see what label the altered indices decode to
        if testLabel != label:
            # the altered label is different, so return the minDiff as the confidence
            return label, minDiff
        else:
            # the altered label is the same, find the next most likely error
            x[secInd[minCol], minCol] = -1000   # clear this option, since it decodes to the same label
            secVal[minCol] = -1000
            for i in range(len(CHARS)):
                if (i != maxInd[minCol]) and (x[i, minCol] > secVal[minCol]):
                    secVal[minCol] = x[i, minCol]   # update the second highest value
                    secInd[minCol] = i              # and its index
            diffVal[minCol] = maxVal[minCol] - secVal[minCol]