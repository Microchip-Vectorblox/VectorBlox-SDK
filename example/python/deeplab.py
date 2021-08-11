import vbx.postprocess.dataset as dataset
import vbx.sim
import argparse
import cv2
import os,os.path
import math
import numpy as np
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    parser.add_argument('--dataset',choices=['COCO','VOC'],default='VOC')
    parser.add_argument('-o', '--output', default="output.png")
    args = parser.parse_args()

    with open(args.model, "rb") as mf:
        model = vbx.sim.Model(mf.read())
    input_size = int(math.sqrt(model.input_lengths[0]/3))
    if not os.path.isfile(args.image):
        print('Error: {} could not be read'.format(args.image))
        os._exit(1)
    img = cv2.imread(args.image)
    if img.shape != (input_size, input_size, 3):
        img_resized = cv2.resize(img, (input_size, input_size)).clip(0, 255)
    else:
        img_resized = img

    flattened = img_resized.swapaxes(1, 2).swapaxes(0, 1).flatten()
    outputs = model.run([flattened])
    assert(len(outputs)==1)
    output = outputs[0].reshape((513,513))
    #add None Colour at start of array
    colours = np.asarray([[0, 0, 0]] + dataset.voc_colors, dtype="uint8")

    #get top category, map that to colour
    mask = colours[output]
    output_img=((0.3 * img_resized) + (0.7 * mask)).astype("uint8")
    cv2.imwrite(args.output, output_img)
    bw = model.get_bandwidth_per_run()
    print("Bandwidth per run = {} Bytes ({:.3} MB/s at 100MHz)".format(bw,bw/100E6))
    print("Estimated {} seconds at 100MHz".format(model.get_estimated_runtime(100E6)))
    print("If running at another frequency, scale these numbers appropriately")
