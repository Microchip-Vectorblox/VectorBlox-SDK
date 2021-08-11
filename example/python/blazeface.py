import vbx.sim
import argparse
import os
import numpy as np
import cv2
from  vbx.postprocess.blazeface import blazeface


def plot_detections(img, detections, with_keypoints=True):
        output_img = img


        print("Found %d faces" % len(detections))
        for i in range(len(detections)):
            ymin = detections[i][ 0] * img.shape[0]
            xmin = detections[i][ 1] * img.shape[1]
            ymax = detections[i][ 2] * img.shape[0]
            xmax = detections[i][ 3] * img.shape[1]
            p1 = (int(xmin),int(ymin))
            p2 = (int(xmax),int(ymax))
            print(p1,p2)
            cv2.rectangle(output_img, p1, p2, (0,50,0))

            if with_keypoints:
                for k in range(6):
                    kp_x = int(detections[i, 4 + k*2    ] * img.shape[1])
                    kp_y = int(detections[i, 4 + k*2 + 1] * img.shape[0])
                    cv2.circle(output_img,(kp_x,kp_y),1,(0,0,255))
        cv2.imwrite("output.png",output_img)

        return output_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    parser.add_argument('-a', '--anchors', default='BlazeFace-PyTorch/anchors.npy')
    parser.add_argument('-s', '--size', type=int, default=128)
    parser.add_argument('-t', '--threshold', type=float, default=0.75)

    args = parser.parse_args()

    model = vbx.sim.model.Model(open(args.model,"rb").read())
    image = cv2.imread(args.image)
    if image.shape != (args.size,args.size,3):
        image = cv2.resize(image,(args.size,args.size))
    input_array = image.transpose(2,0,1)
    outputs = model.run([input_array.flatten()])
    outputs = [o/(1<<16) for o in outputs]

    if len(outputs) == 4:
        a = np.concatenate((outputs[0], outputs[1]))
        b = np.concatenate((outputs[2], outputs[3]))
    else:
        a = outputs[0]
        b = outputs[1]

    anchors = np.load(args.anchors)
    detections = blazeface(a, b, anchors, args.size, args.threshold)
    plot_detections(cv2.imread(args.image), detections[0])



if __name__ == "__main__":
    main()
