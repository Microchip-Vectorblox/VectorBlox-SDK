import numpy as np
import cv2

def one_elem(l):
    "get first element from  list of one element"
    assert len(l) == 1
    return l[0]
    
def pad_input(image, inputDims):
    img = cv2.imread(image)
    imgDims = np.array(img.shape[:2])

    resizeRatio = np.min(inputDims/imgDims)
    resizeDims = np.round(imgDims * resizeRatio).astype('int')
    imgResize = cv2.resize(img.astype('float32'), (resizeDims[1],resizeDims[0]), interpolation=cv2.INTER_LINEAR)
    padTop = int((inputDims[0]-resizeDims[0])/2)
    padBottom = inputDims[0]-resizeDims[0] - padTop
    padLeft = int((inputDims[1]-resizeDims[1])/2)
    padRight = inputDims[1]-resizeDims[1] - padLeft
    imgPad = cv2.copyMakeBorder(imgResize/255.0, padTop, padBottom, padLeft, padRight, cv2.BORDER_CONSTANT, value=[0.5,0.5,0.5])
    modelInput = imgPad.swapaxes(0,2).swapaxes(1,2)
    return modelInput

def convert_to_fixedpoint(data, dtype):
    # this should go away eventually, and always input uint8 rather than fixedpoint Q1.7
    if dtype == np.int16:
        shift_amt = 13
    elif dtype == np.int8:
        shift_amt = 7
    clip_max, clip_min = (1 << shift_amt)-1, -(1 << shift_amt)
    float_img = data.astype(np.float32)/255 * (1 << shift_amt) + 0.5

    fixedpoint_img = np.clip(float_img, clip_min, clip_max).astype(dtype)
    return fixedpoint_img