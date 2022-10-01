import argparse
import os
import numpy as np
import cv2
import glob
import faceDemoClass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelDet', default='../../../tutorials/onnx/scrfd_500m_bnkps/scrfd_500m_bnkps.xml')
    parser.add_argument('--modelRec', default='../../../tutorials/mxnet/mobilefacenet-arcface/model-0000.xml')
    parser.add_argument('--modelAtr', default='../../../tutorials/onnx/genderage/genderage.xml')
    parser.add_argument('--imageDir', default='dbImages')
    parser.add_argument('--debugDir', default='dbDebug')
    parser.add_argument('--db', default='faceDb')
    args = parser.parse_args()

    fdc = faceDemoClass.faceDemo(args.modelDet, args.modelRec, args.modelAtr, None, createDict=True, debugImages=False)
    results = []
    folders = sorted(glob.glob(os.path.join(args.imageDir,'*')))
    if args.debugDir:
        if not os.path.exists(args.debugDir):
            os.mkdir(args.debugDir)
        print('saving debug cropped images to folder '+args.debugDir)

    for f,folder in enumerate(folders):
        folderName = os.path.basename(os.path.normpath(folder))
        print('folder {} of {}: {}'.format(f,len(folders),folderName))
        images = sorted(glob.glob(os.path.join(folder, '*.jpg')))
        images += sorted(glob.glob(os.path.join(folder, '*.png')))
        if args.debugDir:
            if not os.path.exists(args.debugDir+'/'+folderName):
                os.mkdir(args.debugDir+'/'+folderName)
        embedding = np.array([])
        for fileName in images:
            faces,img = fdc.processImage(fileName)
            if len(faces)!=1:
                print("detected {} faces on file: {}".format(len(faces),fileName))
                continue
            if embedding.size==0:
                embedding = faces[0]['embedding']
            else:
                embedding += faces[0]['embedding']  # combine embeddings
            
            if args.debugDir:
                imgFace = fdc.cropFace(img,faces[0])
                baseFileName = os.path.basename(os.path.normpath(fileName))
                cv2.imwrite(args.debugDir+'/'+folderName+'/'+baseFileName, imgFace)
        embedding = embedding/np.sqrt(sum(pow(embedding,2)))    # normalize
        results.append({'name':folderName, 'embedding':embedding})
        
    np.save(args.db+'.npy', results)  #db = np.load('lfwDict.npy',allow_pickle='TRUE')
    print('saved database to '+args.db+'.npy')

if __name__ == "__main__":
    main()
