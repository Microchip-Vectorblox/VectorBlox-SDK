import argparse
import cv2
import os
import matplotlib.pyplot as plt
import posenetProc as pn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('-i', '--image', default='../../ski.273.481.png')
    parser.add_argument('-o', '--out_dir', default='output')
    parser.add_argument('-n', '--out_name', default='output')
    args = parser.parse_args()
    fileName = os.path.splitext(os.path.basename(args.image))[0]
    outputName = args.out_dir+'/'+fileName+"_"+args.out_name+".png"

    img = cv2.imread(args.image)
    modelInput, meta = pn.preprocessImage(img)
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
