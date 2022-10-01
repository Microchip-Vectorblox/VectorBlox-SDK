import argparse
import numpy as np
import cv2
import faceDemoClass
import faceTracker

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelDet', default='../../../tutorials/onnx/scrfd_500m_bnkps/scrfd_500m_bnkps.vnnx')
    parser.add_argument('--modelRec', default='../../../tutorials/mxnet/mobilefacenet-arcface/mobilefacenet-arcface.vnnx')
    parser.add_argument('--modelAtr', default='../../../tutorials/onnx/genderage/genderage.vnnx')
    parser.add_argument('--video', default='gardenIn.mp4')
    parser.add_argument('--videoOut', default='demoOutput.mp4')
    parser.add_argument('--imgOutDir',default='videoOut')
    parser.add_argument('--db', default='faceDb.npy')
    parser.add_argument('--crop', default=True) # if true, image is cropped to match model shape; otherwise padded
    parser.add_argument('--maxFrame', type=int) # set max frame
    args = parser.parse_args()
    
    vidIn = cv2.VideoCapture(args.video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vidOut = cv2.VideoWriter(args.videoOut, fourcc, vidIn.get(cv2.CAP_PROP_FPS), (int(vidIn.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vidIn.get(cv2.CAP_PROP_FRAME_HEIGHT))),True)
    
    fdc = faceDemoClass.faceDemo(args.modelDet, args.modelRec, args.modelAtr, None, createDict=False, debugImages=False)
    db = np.load(args.db,allow_pickle='TRUE')
    
    vidInSuccess,img = vidIn.read()
    if not vidInSuccess:
        print('could not read video file {}'.format(args.video))
        return
    width = int(np.ceil(min(img.shape[0],img.shape[1])/400))
    frameNum = 1
    recPerFrame = 1
    atrPerFrame = 1
    
    tracker = faceTracker.faceTracker()
    while vidInSuccess:
        print('frame {}'.format(frameNum))
        objects = fdc.detectFaces(img)
        objects = tracker.updateDetection(objects)   # each object now has an assigned track

        for obj in objects[:recPerFrame]:
            obj = fdc.recognizeFace(img, obj)
                
            similarity = np.zeros(len(db))
            for n,ref in enumerate(db):
                similarity[n] = np.dot(obj['embedding'],ref['embedding'])
            maxInd = np.argmax(similarity)  #ind = np.argsort(-similarity)
            obj['name'] = db[maxInd]['name']
            obj['similarity'] = similarity[maxInd]
                
            tracker.updateRecognition(obj)

        for obj in objects[:atrPerFrame]:
            obj = fdc.attributeFace(img, obj)
            tracker.updateAttribution(obj)
        
        if args.crop:
            cropLeft = max(0,-fdc.meta['padLeft']*fdc.meta['imageX']/fdc.meta['resizeX'])
            cropRight = max(0,-fdc.meta['padRight']*fdc.meta['imageX']/fdc.meta['resizeX'])
            img[:,0:round(cropLeft),:] = img[:,0:round(cropLeft),:]>>1
            img[:,fdc.meta['imageX']-round(cropRight):,:] = img[:,fdc.meta['imageX']-round(cropRight):,:]>>1
            
        for obj in objects:
            track = obj['track']
                
            box = track['filter'].box()
                
            if track['name']:
                color = (0,250,0)
                resText = '{:02}% {}'.format(round(100*track['similarity']),track['name'])
            else:
                color = (250,0,0) #(0,0,250)
                resText = ''    #'{:.3f}'.format(track['similarity'])

            p1 = (int(box[0]),int(box[1]))
            p2 = (int(box[2]),int(box[3]))
            cv2.rectangle(img, p1, p2, color, width)
            p3 = (int(box[0]),int(box[3]+12*width))
            cv2.putText(img, resText, p3, cv2.FONT_HERSHEY_SIMPLEX, width*.35, color, width, cv2.LINE_AA)
            
            if not track['gender'] is None:
                if track['gender']>0.2:
                    atrText = 'F'
                elif track['gender']<-0.4:
                    atrText = 'M'
                else:
                    atrText = '?'
                #atrText += ' {:6.3f} {}'.format(track['gender'],int(track['age']))
                atrText += ' {}'.format(int(track['age']))
            else:
                atrText = ''
            p4 = (int(box[0]),int(box[1]-4*width))
            if p4[1]<0:
                p4 = (int(box[0]),int(box[1]+4*width))
            cv2.putText(img, atrText, p4, cv2.FONT_HERSHEY_SIMPLEX, width*.35, color, width, cv2.LINE_AA)

            # additional info for detection
            # detText = 'det {:0.4f}'.format(obj['detectScore'])
            # p4 = (int(box[0]),int(box[1]-4*width))
            # cv2.putText(img, detText, p4, cv2.FONT_HERSHEY_SIMPLEX, width*.25, color, int(width*.75), cv2.LINE_AA)
            
            # additional info for tracking
            # trackText = '{:05.2f} {}'.format(track['res'], track['frames'])
            # p4 = (int(box[0]),int(box[1]-4*width))
            # cv2.putText(img, trackText, p4, cv2.FONT_HERSHEY_SIMPLEX, width*.25, color, int(width*.75), cv2.LINE_AA)

        cv2.putText(img,'{}'.format(frameNum),(5,45),cv2.FONT_HERSHEY_SIMPLEX, width*.35, [0,0,255], int(width*.75), cv2.LINE_AA)
        vidOut.write(img)
        if args.imgOutDir:
            cv2.imwrite(args.imgOutDir+'/frame{}.jpg'.format(frameNum),img)
        if args.maxFrame and frameNum > args.maxFrame:
            break
        vidInSuccess,img = vidIn.read()
        frameNum += 1
        
    vidIn.release()
    vidOut.release()
    
if __name__ == "__main__":
    main()
