import argparse
import os
import numpy as np
import cv2
import glob
import faceDemoClass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelDet', default='../../../tutorials/pytorch/retinaface.mobilenet/retinaface.mobilenet.onnx')
    parser.add_argument('--modelAtr', default='../../../tutorials/onnx/genderage/genderage.vnnx')
    parser.add_argument('--imageDir')
    parser.add_argument('--sampleDir', default='attributeSamples')
    args = parser.parse_args()

    fdc = faceDemoClass.faceDemo(args.modelDet, None, args.modelAtr, None, createDict=True, debugImages=False)
    
    if args.sampleDir:
        if not os.path.exists(args.sampleDir):
            os.mkdir(args.sampleDir)
        print('saving attribute sample images to folder '+args.sampleDir)

    images = sorted(glob.glob(os.path.join(args.imageDir, '*.jpg')))
    images += sorted(glob.glob(os.path.join(args.imageDir, '*.png')))
    for fileName in images:
        #if fileName[1] == '_':
        #    continue
        img = cv2.imread(fileName)
        faces = fdc.detectFaces(img)
        if len(faces)!=1:
            print("detected {} faces on file: {}".format(len(faces),fileName))
            continue
        
        if args.sampleDir:
            imgFace = fdc.cropAtrFace(img,faces[0])
            baseFileName = os.path.basename(os.path.normpath(fileName))
            cv2.imwrite(args.sampleDir+'/'+baseFileName, imgFace)

if __name__ == "__main__":
    main()
