import argparse
import cv2
import os
import matplotlib.pyplot as plt
import posenetProc as pn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image', default='../../test_images/ski.273.481.jpg')
    parser.add_argument('-o', '--out_dir', default='output')
    parser.add_argument('-n', '--out_name', default='output')
    parser.add_argument('-b', '--bgr', action='store_true')
    parser.add_argument('-m', '--mean', type=float, nargs='+', default=0.)
    parser.add_argument('-sc', '--scale', type=float, nargs='+', default=1.)
    args = parser.parse_args()
    fileName = os.path.splitext(os.path.basename(args.image))[0]
    outputName = fileName+"_"+args.out_name+".png"

    img = cv2.imread(args.image)
    modelInput, meta = pn.preprocessImage(img, (not args.bgr), args.mean, args.scale)
    prediction = pn.model_infer(args.model, modelInput)
    cocoRes = pn.postProcess(prediction, meta)
    print('detected {} set(s) of keypoints'.format(len(cocoRes)))
    
    fig = plt.figure()
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pn.drawRes(cocoRes)
    plt.savefig(outputName)
    plt.close(fig)
    print("saved "+outputName)

if __name__ == "__main__":
    main()
