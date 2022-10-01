import argparse
import os
import numpy as np
import cv2
import glob
import faceDemoClass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelDet', default='../../../tutorials/onnx/scrfd_500m_bnkps/scrfd_500m_bnkps.vnnx')
    parser.add_argument('--modelRec', default='../../../tutorials/mxnet/mobilefacenet-arcface/mobilefacenet-arcface.vnnx')
    parser.add_argument('--modelAtr', default='../../../tutorials/onnx/genderage/genderage.vnnx')
    parser.add_argument('--image', default='garden.jpg')
    parser.add_argument('-d', '--debug', default=False)
    parser.add_argument('--db', default='faceDb.npy')
    args = parser.parse_args()

    fdc = faceDemoClass.faceDemo(args.modelDet, args.modelRec, args.modelAtr, None, createDict=False, debugImages=args.debug)
    
    faces,img = fdc.processImage(args.image)
    db = np.load(args.db,allow_pickle='TRUE')
    width = int(np.ceil(min(img.shape[0],img.shape[1])/300))
    
    for face in faces:
        similarity = np.zeros(len(db))
        for n,ref in enumerate(db):
            similarity[n] = np.dot(face['embedding'],ref['embedding'])
        maxInd = np.argmax(similarity)
        face['similarity'] = similarity[maxInd]
        face['name'] = db[maxInd]['name']
        
    for face in faces:
        if face['similarity']>0.475:
            resText = '{:.3}% {}'.format(face['similarity'], face['name'])
            print(face['similarity'], face['name'])
            color = (0,250,0)
            if face['similarity']<0.525:
                color = (0,250,250)
        else:
            resText = '{:.3}%'.format(face['similarity'])
            color = (0,0,250)
        if face['gender']>0:
            atrText = 'F'
        else:
            atrText = 'M'
        atrText += ' {}'.format(int(face['age']))
        print(atrText)
        box = face['box']
        p1 = (int(box[0]),int(box[1]))
        p2 = (int(box[2]),int(box[3]))
        cv2.rectangle(img, p1, p2, color, width)
        p3 = (int(box[0]),int(box[3]+12*width))
        cv2.putText(img, resText, p3, cv2.FONT_HERSHEY_SIMPLEX, width*.35, color, width, cv2.LINE_AA)
        p4 = (int(box[0]),int(box[1]-4*width))
        cv2.putText(img, atrText, p4, cv2.FONT_HERSHEY_SIMPLEX, width*.35, color, width, cv2.LINE_AA)
        
    cv2.imwrite("demoOutput.png",img)
    print('saved output to demoOutput.png')

if __name__ == "__main__":
    main()
