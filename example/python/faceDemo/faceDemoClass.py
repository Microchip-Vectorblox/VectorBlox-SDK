import numpy as np
import cv2
import vbx.postprocess.retinaface
import vbx.postprocess.scrfd
import os

# SphereFace uses five facial landmarks (two eyes, nose point and two mouth corners)
# https://github.com/wy1iu/sphereface/blob/master/preprocess/code/face_align_demo.m
sphereface_coord5 = np.array([[30.2946, 51.6963],
                  [65.5318, 51.5014],
                  [48.0252, 71.7366],
                  [33.5493, 92.3655],
                  [62.7299, 92.2041]])
# The two mouth corners are averaged to make one mouth point 
sphereface_coord4 = np.array([[30.2946, 51.6963],
                   [65.5318, 51.5014],
                   [48.0252, 71.7366],
                   [48.1396, 92.2848]])
# https://github.com/deepinsight/insightface/blob/master/recognition/common/face_align.py
arcface_coord5 = np.array([[38.2946, 51.6963],
                           [73.5318, 51.5014],
                           [56.0252, 71.7366],
                           [41.5493, 92.3655],
                           [70.7299, 92.2041]])
arcface_coord4 = np.array([[38.2946, 51.6963],
                           [73.5318, 51.5014],
                           [56.0252, 71.7366],
                           [56.1396, 92.284805]])

class faceDemo:
    def __init__(self, modelDet, modelRec, modelAtr, anchorsFile, createDict=False, debugImages=False, crop=False):
        self.modelDet = modelDet
        self.modelRec = modelRec
        self.modelAtr = modelAtr
        self.anchorsFile = anchorsFile
        self.createDict = createDict
        self.debugImages = debugImages
        self.useRotation = True
        self.rotationMode = 'estimate'
        self.crop = crop
        if anchorsFile:
            self.anchors = np.load(anchorsFile)

        if modelDet:        
            if isinstance(modelDet,list):   # mxnet
                import mxnet as mx
                ctx = mx.cpu()
                self.detector = mx.gluon.nn.SymbolBlock.imports(modelDet[0], ['data'], modelDet[1], ctx=ctx)
                self.detectorInputDims = [256,256]  # how to check mxnet input shape?
            else:
                if 'mtcnn' == modelDet.lower():
                    from mtcnn.mtcnn import MTCNN
                    if self.createDict:
                        self.detector = MTCNN(steps_threshold=[0.5,0.6,0.6])
                    else:
                        self.detector = MTCNN()
                elif '.vnnx' in modelDet:
                    import vbx.sim
                    self.detector = vbx.sim.model.Model(open(modelDet,"rb").read())
                    if 'retina' in modelDet.lower():
                        if self.detector.input_lengths[0] == 512*288*3:   # retinaface 512x288
                            self.detectorInputDims = [288,512]
                        elif self.detector.input_lengths[0] == 320*320*3:   # retinaface 320
                            self.detectorInputDims = [320,320]    
                        elif self.detector.input_lengths[0] == 640*640*3: # retinaface 640 
                            self.detectorInputDims = [640,640]
                        else:
                            assert(0)
                    elif 'scrfd' in modelDet.lower():
                        if self.detector.input_lengths[0] == 512*288*3:   # retinaface 512x288
                            self.detectorInputDims = [288,512]
                        else:
                            assert(0)
                    else:
                        if self.detector.input_lengths[0] == 256*256*3:   # Blazeface back
                            self.detectorInputDims = [256,256]    
                        elif self.detector.input_lengths[0] == 128*128*3: # Blazeface front
                            self.detectorInputDims = [128,128]
                        else:
                            assert(0)
                elif '.xml' in modelDet:
                    import openvino.inference_engine as ie
                    weights=modelDet.replace('.xml', '.bin')
                    core = ie.IECore()
                    net = core.read_network(model=modelDet, weights=weights)
                    assert(len(net.input_info) == 1)
                    self.detector = core.load_network(network=net, device_name="CPU")
                    inputName = list(self.detector.requests[0].input_blobs.keys())[0]
                    self.detectorInputDims = self.detector.requests[0].input_blobs[inputName].buffer.shape[2:4]
                elif '.onnx' in modelDet:
                    import onnxruntime
                    self.detector = onnxruntime.InferenceSession(modelDet, None)
                    self.detectorInputDims = self.detector.get_inputs()[0].shape[2:4]
                elif '.tflite' in modelDet:
                    import tensorflow as tf
                    self.detector = tf.lite.Interpreter(model_path=modelDet)
                    self.detector.allocate_tensors()
                    input_details = self.detector.get_input_details()
                    self.detectorInputDims = input_details[0]['shape'][1:3]
                else:
                    assert(0)
        
        if modelRec:
            if isinstance(modelRec,list):   # mxnet
                import mxnet as mx
                ctx = mx.cpu()
                self.recognizer = mx.gluon.nn.SymbolBlock.imports(modelRec[0], ['data'], modelRec[1], ctx=ctx)
                self.recognizerInputDims = [112,112]    # assume arcface; can we get these from the model?
                self.coord5 = arcface_coord5
                self.coord4 = arcface_coord4
            else:    
                if '.vnnx' in modelRec:
                    import vbx.sim
                    self.recognizer = vbx.sim.model.Model(open(modelRec,"rb").read())
                    if self.recognizer.input_lengths[0] == 112*112*3:   # arcface
                        self.recognizerInputDims = [112,112]
                    elif self.recognizer.input_lengths[0] == 112*96*3:  # sphereface
                        self.recognizerInputDims = [112,96]
                    else:
                        assert(0)
                elif '.xml' in modelRec:
                    import openvino.inference_engine as ie
                    weights=modelRec.replace('.xml', '.bin')
                    core = ie.IECore()
                    net = core.read_network(model=modelRec, weights=weights)
                    assert(len(net.input_info) == 1)
                    self.recognizer = core.load_network(network=net, device_name="CPU")
                    inputName = list(self.recognizer.requests[0].input_blobs.keys())[0]
                    self.recognizerInputDims = self.recognizer.requests[0].input_blobs[inputName].buffer.shape[2:4]
                elif '.onnx' in modelRec:
                    import onnxruntime
                    self.recognizer = onnxruntime.InferenceSession(modelRec, None)
                    self.recognizerInputDims = self.recognizer.get_inputs()[0].shape[2:4]
                else:
                    assert(0)
                if 'sphere' in modelRec.lower():
                    self.coord5 = sphereface_coord5
                    self.coord4 = sphereface_coord4
                elif 'arc' in modelRec.lower():
                    self.coord5 = arcface_coord5
                    self.coord4 = arcface_coord4
                else:
                    assert(0)
        
        if modelAtr:
            if '.vnnx' in modelAtr:
                import vbx.sim
                self.attributer = vbx.sim.model.Model(open(modelAtr,"rb").read())
                if self.attributer.input_lengths[0] == 96*96*3:
                    self.attributerInputDims = [96,96]
                else:
                    assert(0)
            elif '.xml' in modelAtr:
                import openvino.inference_engine as ie
                weights=modelAtr.replace('.xml', '.bin')
                core = ie.IECore()
                net = core.read_network(model=modelAtr, weights=weights)
                assert(len(net.input_info) == 1)
                self.attributer = core.load_network(network=net, device_name="CPU")
                inputName = list(self.attributer.requests[0].input_blobs.keys())[0]
                self.attributerInputDims = self.attributer.requests[0].input_blobs[inputName].buffer.shape[2:4]
            elif '.onnx' in modelAtr:
                import onnxruntime
                self.attributer = onnxruntime.InferenceSession(modelAtr, None)
                self.attributerInputDims = self.attributer.get_inputs()[0].shape[2:4]
            else:
                assert(0)

            
    def detectFaces(self, img):
        if 'retina' in self.modelDet.lower():
            faces = self.runRetinaFace(img)
        elif 'scrfd' in self.modelDet.lower():
            faces = self.runScrfd(img)
        elif 'mtcnn' == self.modelDet.lower():
            faces = self.detector.detect_faces(img)
            for f in faces:
                b = f['box']
                f['box'] = [b[0], b[1], b[0]+b[2], b[1]+b[3]] # x0,y0,x1,y1
        else:
            faces = self.runBlazeFace(img)
        return faces

    def runRecognitionModel(self, img):
        modelInput = img.transpose(2,0,1)
        modelInput = np.expand_dims(modelInput, axis=0)
        if type(self.recognizer).__name__=='SymbolBlock': #mxnet
            import mxnet as mx
            modelInput = mx.nd.array(modelInput)
            outputs = self.recognizer(modelInput)
            output = outputs[0].asnumpy()
        elif '.vnnx' in self.modelRec:
            inputFlat = modelInput.flatten().astype(np.uint8)
            outputs = self.recognizer.run([inputFlat])
            outputs = [o.astype(np.float32)/(1<<16) for o in outputs]
            output = outputs[0]
        elif '.xml' in self.modelRec:
            inputName = list(self.recognizer.requests[0].input_blobs.keys())[0]
            self.recognizer.requests[0].input_blobs[inputName].buffer[:] = modelInput
            self.recognizer.requests[0].infer()
            outputNames = list(self.recognizer.requests[0].output_blobs.keys())
            outputs = [self.recognizer.requests[0].output_blobs[o].buffer.flatten() for o in outputNames]
            output = outputs[0]
        elif '.onnx' in self.modelRec:
            input_name = self.recognizer.get_inputs()[0].name
            outputs = self.recognizer.run([], {input_name: modelInput.astype(np.float32)})
            output = outputs[0][0]
        return output

    def runAttributeModel(self, img):
        modelInput = img.transpose(2,0,1)
        modelInput = np.expand_dims(modelInput, axis=0)
        if '.vnnx' in self.modelAtr:
            inputFlat = modelInput.flatten().astype(np.uint8)
            outputs = self.attributer.run([inputFlat])
            outputs = [o.astype(np.float32)/(1<<16) for o in outputs]
            outputs = np.concatenate((outputs[1],outputs[0]))
        elif '.xml' in self.modelAtr:
            inputName = list(self.attributer.requests[0].input_blobs.keys())[0]
            self.attributer.requests[0].input_blobs[inputName].buffer[:] = modelInput
            self.attributer.requests[0].infer()
            outputNames = list(self.attributer.requests[0].output_blobs.keys())
            outputs = [self.attributer.requests[0].output_blobs[o].buffer.flatten() for o in outputNames]
            outputs = np.concatenate((outputs[0],outputs[1]))
        elif '.onnx' in self.modelAtr:
            input_name = self.attributer.get_inputs()[0].name
            #modelInput = (modelInput[:,::-1,:,:].astype(np.float32)-127.5)/128.0 # this is for the onnx export from pytorch
            modelInput = modelInput[:,::-1,:,:].astype(np.float32)
            outputs = self.attributer.run([], {input_name: modelInput})
            outputs = outputs[0][0]
        return outputs

    def blazeFacePostProc(self, raw_score_tensor, raw_box_tensor, anchors, scale=256.0, thresh=0.75):
        
        def calcIou(A,B):
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
        
        anchorsPixels = anchors*scale
        min_suppression_threshold = 0.3
        raw_thresh = -np.log((1-thresh)/thresh)
        raw_scores = raw_score_tensor.squeeze()
        raw_box_tensor = raw_box_tensor.squeeze()
        N = 0   # number of detects
        for n,raw_score in enumerate(raw_scores):
            if raw_score > raw_thresh:
                raw_box_tensor[n,0] += anchorsPixels[n, 0]
                raw_box_tensor[n,1] += anchorsPixels[n, 1]
                raw_box_tensor[n,np.arange(4,16,2)] += anchorsPixels[n, 0]
                raw_box_tensor[n,np.arange(5,16,2)] += anchorsPixels[n, 1]
                raw_scores[n] = 1/(1 + np.exp(-raw_score))
                N += 1
            else:
                raw_scores[n] = 0
        ind = np.argsort(-raw_scores)[:N]
        used = np.zeros((N),dtype='bool')
        
        # combine detects with significant IOU (ind[2] and ind[6] should match)
        detects = []
        scores = []
        for n1 in range(N):
            if used[n1]:
                continue
            totalScore = raw_scores[ind[n1]]
            blendPoints = raw_box_tensor[ind[n1],:] * raw_scores[ind[n1]]
            for n2 in range(n1+1,N):
                if used[n2]:
                    continue
                iou = calcIou(raw_box_tensor[ind[n1],:4],raw_box_tensor[ind[n2],:4])
                if iou > min_suppression_threshold:
                    used[n2] = True
                    weight = raw_scores[ind[n2]]    # weight based on score
                    blendPoints += weight * raw_box_tensor[ind[n2],:]
                    totalScore += raw_scores[ind[n2]]
            
            blendPoints *= 1/totalScore   # scale back
            detects.append(blendPoints)
            scores.append(raw_scores[ind[n1]])
            
        return detects,scores
    
    def runBlazeFace(self, img):
        inputImg,meta = self.preProcess(img,self.detectorInputDims)
        self.meta = meta
        if '.vnnx' in self.modelDet:
            modelInput = inputImg.transpose(2,0,1).copy()
            outputs = self.detector.run([modelInput.flatten()])
            outputs = [o/(1<<16) for o in outputs]
            outputs[1] = outputs[1].reshape(896,16)
        elif '.onnx' in self.modelDet:            
            modelInput = inputImg.transpose(2,0,1).copy()
            modelInput = np.expand_dims(modelInput, axis=0)
            inputName = self.detector.get_inputs()[0].name
            outputs = self.detector.run([], {inputName: modelInput.astype(np.float32)})
        elif '.tflite' in self.modelDet:
            input_details = self.detector.get_input_details()
            output_details = self.detector.get_output_details()
            modelInput = cv2.cvtColor(inputImg, cv2.COLOR_BGR2RGB)
            modelInput = (modelInput.astype('float32')-127.5)/127.5
            modelInput = np.expand_dims(modelInput, axis=0)
            self.detector.set_tensor(input_details[0]['index'], modelInput)
            self.detector.invoke()
            outputs = [self.detector.get_tensor(output_details[1]['index']),
                       self.detector.get_tensor(output_details[0]['index'])]
        else:
            assert(0)
        
        if self.createDict:
            detections,scores = self.blazeFacePostProc(outputs[0].copy(), outputs[1].copy(), self.anchors, self.detectorInputDims[0], 0.7)
            if len(detections)==0:
                detections,scores = self.blazeFacePostProc(outputs[0], outputs[1], self.anchors, self.detectorInputDims[0], 0.54)
        else:
            detections,scores = self.blazeFacePostProc(outputs[0], outputs[1], self.anchors, self.detectorInputDims[0])
        
        faces = []
        for di,d in enumerate(detections):
            x = d[0]
            y = d[1]
            w = d[2]
            h = d[3]
            d[0] = x-w/2    # convert from (x,y,w,h) to (left,top,right,bottom)
            d[1] = y-h/2
            d[2] = x+w/2
            d[3] = y+h/2
            for n in np.arange(0,12,2):
                d[n] = (d[n] - meta['padLeft'])/meta['resizeX']*meta['imageX']  # scale to original image
            for n in np.arange(1,12,2):
                d[n] = (d[n] - meta['padTop'])/meta['resizeY']*meta['imageY']
            face = dict(box=d[0:4], keypoints=d[4:12].reshape((4,2)), detectScore=scores[di])
            faces.append(face)
        return faces

    def runRetinaFace(self, img):
        inputImg,meta = self.preProcess(img,self.detectorInputDims)
        self.meta = meta
        if '.vnnx' in self.modelDet:
            modelInput = inputImg.transpose(2,0,1).copy()
            outputs = self.detector.run([modelInput.flatten()])
            outputs = [o/(1<<16) for o in outputs]
            raw_faces = vbx.postprocess.retinaface.retinafaceVnnx(outputs,self.detectorInputDims[1],self.detectorInputDims[0],detectThresh=0.76, maxIou=0.34)
        elif '.onnx' in self.modelDet:            
            modelInput = inputImg.transpose(2,0,1).copy()
            modelInput = np.expand_dims(modelInput, axis=0)
            inputName = self.detector.get_inputs()[0].name
            outputs = self.detector.run([], {inputName: modelInput.astype(np.float32)})
            raw_faces = vbx.postprocess.retinaface.retinaface(outputs,self.detectorInputDims[1],self.detectorInputDims[0],confidence_threshold=0.75, nms_threshold=0.35)
        else:
            assert(0)
       
        faces = []
        for f in raw_faces:
            f['box'][0] = (f['box'][0] - meta['padLeft'])/meta['resizeX']*meta['imageX']  # scale to original image
            f['box'][2] = (f['box'][2] - meta['padLeft'])/meta['resizeX']*meta['imageX']  # scale to original image
            f['box'][1] = (f['box'][1] - meta['padTop'])/meta['resizeY']*meta['imageY']  # scale to original image
            f['box'][3] = (f['box'][3] - meta['padTop'])/meta['resizeY']*meta['imageY']  # scale to original image

            for n in range(len(f['landmarks'])):
                f['landmarks'][n] = (f['landmarks'][n] - [meta['padLeft'],meta['padTop']])
                f['landmarks'][n] = (f['landmarks'][n] / [meta['resizeX'],meta['resizeY']])
                f['landmarks'][n] = (f['landmarks'][n] * [meta['imageX'],meta['imageY']])
            face = dict(box=f['box'], keypoints=f['landmarks'], detectScore=f['score'])
            faces.append(face)

        return faces
    
    def runScrfd(self, img):
        inputImg,meta = self.preProcess(img,self.detectorInputDims)
        self.meta = meta

        if '.vnnx' in self.modelDet:
            modelInput = inputImg.transpose(2,0,1).copy()
            outputs = self.detector.run([modelInput.flatten()])
            outputs = [o/(1<<16) for o in outputs]
            outputs = [outputs[2],outputs[5],outputs[8],outputs[1],outputs[4],outputs[7],outputs[0],outputs[3],outputs[6]]
        elif '.xml' in self.modelDet:
            modelInput = inputImg.transpose(2,0,1).copy()
            inputName = list(self.detector.requests[0].input_blobs.keys())[0]
            self.detector.requests[0].input_blobs[inputName].buffer[:] = modelInput
            self.detector.requests[0].infer()
            outputNames = list(self.detector.requests[0].output_blobs.keys())
            outputs = [self.detector.requests[0].output_blobs[o].buffer.flatten() for o in outputNames]
            outputs = [outputs[0],outputs[3],outputs[6],outputs[1],outputs[4],outputs[7],outputs[2],outputs[5],outputs[8]]
        elif '.onnx' in self.modelDet:            
            modelInput = inputImg.transpose(2,0,1).copy()
            modelInput = np.expand_dims(modelInput, axis=0)
            inputName = self.detector.get_inputs()[0].name
            outputs = self.detector.run([], {inputName: modelInput.astype(np.float32)})
            outputs = [outputs[2],outputs[5],outputs[8],outputs[1],outputs[4],outputs[7],outputs[0],outputs[3],outputs[6]]
        else:
            assert(0)

        raw_faces = vbx.postprocess.scrfd.scrfd(outputs,self.detectorInputDims[1],self.detectorInputDims[0],detectThresh=0.76, maxIou=0.34)
       
        faces = []
        for f in raw_faces:
            f['box'][0] = (f['box'][0] - meta['padLeft'])/meta['resizeX']*meta['imageX']  # scale to original image
            f['box'][2] = (f['box'][2] - meta['padLeft'])/meta['resizeX']*meta['imageX']  # scale to original image
            f['box'][1] = (f['box'][1] - meta['padTop'])/meta['resizeY']*meta['imageY']  # scale to original image
            f['box'][3] = (f['box'][3] - meta['padTop'])/meta['resizeY']*meta['imageY']  # scale to original image

            for n in range(len(f['landmarks'])):
                f['landmarks'][n] = (f['landmarks'][n] - [meta['padLeft'],meta['padTop']])
                f['landmarks'][n] = (f['landmarks'][n] / [meta['resizeX'],meta['resizeY']])
                f['landmarks'][n] = (f['landmarks'][n] * [meta['imageX'],meta['imageY']])
            face = dict(box=f['box'], keypoints=f['landmarks'], detectScore=f['score'])
            faces.append(face)

        return faces

    def preProcess(self, img, inputDims):
        imgDims = np.array(img.shape[:2])
        
        if self.crop:
            resizeRatio = np.max(np.array(inputDims)/imgDims)
        else: # pad
            resizeRatio = np.min(np.array(inputDims)/imgDims)
            
        resizeDims = np.round(imgDims * resizeRatio).astype('int')
        imgResize = cv2.resize(img, (resizeDims[1],resizeDims[0]), interpolation=cv2.INTER_LINEAR)
        padTop = int((inputDims[0]-resizeDims[0])/2)    # if cropping, these values may be negative
        padBottom = inputDims[0]-resizeDims[0] - padTop
        padLeft = int((inputDims[1]-resizeDims[1])/2)
        padRight = inputDims[1]-resizeDims[1] - padLeft
        
        if self.crop:
            inputImg = imgResize[-padTop:resizeDims[0]+padBottom,-padLeft:resizeDims[1]+padRight,:].copy()
        else:
            inputImg = cv2.copyMakeBorder(imgResize, padTop, padBottom, padLeft, padRight, cv2.BORDER_CONSTANT, value=[0,0,0])
        meta = {'imageX':imgDims[1],
                'imageY':imgDims[0],
                'inputX':inputDims[1],
                'inputY':inputDims[0],
                'resizeX':resizeDims[1],
                'resizeY':resizeDims[0],
                'padTop':padTop,
                'padBottom':padBottom,
                'padLeft':padLeft,
                'padRight':padRight}
        return inputImg, meta

    def plotDetections(self, img, faces):
        width = int(np.ceil(min(img.shape[0],img.shape[1])/300))
        for f in faces:
            p1 = (int(f['box'][0]),int(f['box'][1]))
            p2 = (int(f['box'][2]),int(f['box'][3]))
            cv2.rectangle(img, p1, p2, (0,250,0), width)
            for k in f['keypoints']:
                p = (int(k[0]),int(k[1]))
                cv2.circle(img, p, 2*width, (0,0,255), width)
        return img

    def processImage(self, imageFile):
        if isinstance(imageFile,np.ndarray):
            img = imageFile
        else:
            img = cv2.imread(imageFile)
        
        faces = self.detectFaces(img)
        
        if self.createDict and len(faces)>1: # pick best face for dictionary
            # approximate box for LFW dataset
            centerBox = np.array([0.3*self.meta['imageX'],0.3*self.meta['imageY'],0.7*self.meta['imageX'],0.7*self.meta['imageY']])
            err = np.zeros(len(faces))    
            for n,f in enumerate(faces):
                err[n] = np.sum(np.square(f['box'] - centerBox))
            faces = [faces[np.argmin(err)]]

        if self.debugImages:
            imgOut = self.plotDetections(img.copy(), faces)
            cv2.imwrite("detections.png",imgOut)
        
        for face in faces:
            self.recognizeFace(img, face)
            self.attributeFace(img, face)
            
        return faces,img

    def cropFace(self, img, face):
        inputDims = self.recognizerInputDims
        numK = face['keypoints'].shape[0]   # number of keypoints
        keypoints = face['keypoints']
        if numK==4:
            coord = self.coord4
        else:
            coord = self.coord5
        M = cv2.estimateAffinePartial2D(keypoints,coord,method=cv2.LMEDS,maxIters=10000, confidence=0.001)[0]
        imgFace = cv2.warpAffine(img,M,(inputDims[1],inputDims[0]))
        return imgFace

    def cropAtrFace(self, img, face):
        inputDims = self.attributerInputDims
        # method from https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/attribute.py
        # starting at line 73
        bbox = face['box']
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        radius = max(w, h) * 0.75
        keypoints = np.array([center-radius,center+radius])
        coordAtr = np.array([[0,0],inputDims])
        M = cv2.estimateAffinePartial2D(keypoints,coordAtr,method=cv2.LMEDS,maxIters=10000, confidence=0.001)[0]
        imgFace = cv2.warpAffine(img,M,(inputDims[1],inputDims[0]))
        return imgFace

    def attributeFace(self, img, face):
        if self.modelAtr:
            imgFace = self.cropAtrFace(img, face)

            attributes = self.runAttributeModel(imgFace)
            face['gender'] = attributes[0]
            face['age'] = 100 * attributes[2]

            if self.debugImages:
                cv2.imwrite("faceAtrCrop.png",imgFace)

        return face

        
    def recognizeFace(self, img, face):
        inputDims = self.recognizerInputDims
        numK = face['keypoints'].shape[0]   # number of keypoints
        keypoints = face['keypoints']
        if numK==4:
            coord = self.coord4
        else:
            coord = self.coord5
        M = cv2.estimateAffinePartial2D(keypoints,coord,method=cv2.LMEDS,maxIters=10000, confidence=0.001)[0]
        imgFace = cv2.warpAffine(img,M,(inputDims[1],inputDims[0]))


        if self.useRotation:   # rotation
            if self.rotationMode == 'estimate':
                M = cv2.estimateAffinePartial2D(keypoints,coord,method=cv2.LMEDS,maxIters=10000, confidence=0.001)[0]
                if True:
                    imgFace = cv2.warpAffine(img,M,(inputDims[1],inputDims[0]))
                else: # produces same result to first 8 decimal places
                    M_ = np.zeros((3,3))
                    M_[:2] = M
                    M_[2] = [0,0,1]
                    imgFace = cv2.warpPerspective(img,M_,(inputDims[1],inputDims[0]))
            elif self.rotationMode == 'affine':
                assert(len(keypoints) == 4)
                a = np.float32([keypoints[0], keypoints[1], keypoints[3]])
                b = np.float32([coord[0], coord[1], coord[3]])
                M = cv2.getAffineTransform(a,b)
                imgFace = cv2.warpAffine(img,M,(inputDims[1],inputDims[0]))
            elif self.rotationMode == 'perspective':
                assert(len(keypoints) == 4)
                M = cv2.getPerspectiveTransform(np.float32(keypoints),np.float32(coord))
                imgFace = cv2.warpPerspective(img,M,(inputDims[1],inputDims[0]))
        else:   # no rotation
            A = np.zeros((2*numK,3))
            A[0:numK,0] = keypoints[:,0]
            A[0:numK,1] = 1
            A[numK:2*numK,0] = keypoints[:,1]
            A[numK:2*numK,2] = 1
            b = coord.reshape((2*numK,1),order='F').copy()
            x = np.linalg.lstsq(A,b,rcond=None)[0]
            M = np.zeros((2,3))
            M[0,0] = M[1,1] = x[0]
            M[0,2] = x[1]
            M[1,2] = x[2]
            imgFace = cv2.warpAffine(img,M,(inputDims[1],inputDims[0]))
        
        embedding = self.runRecognitionModel(imgFace)
        face['embedding'] = embedding/np.sqrt(sum(pow(embedding,2)))     #normalize emebdding

        if self.debugImages:
            cv2.imwrite("faceRecCrop.png",imgFace)
            temp = np.column_stack((keypoints,np.ones(numK)))
            keypointsWarp = np.matmul(temp,M.transpose())
            for kp in keypointsWarp:
                cv2.circle(imgFace, (int(kp[0]),int(kp[1])), 1, (0,255,0), 1)
            for kp in self.coord5:
                cv2.circle(imgFace, (int(kp[0]),int(kp[1])), 2, (0,0,255), 1)
            cv2.imwrite("faceRecPoints.png",imgFace)
        
        return face

