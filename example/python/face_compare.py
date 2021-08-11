import vbx.sim
import cv2
import numpy as np
import argparse
import os
import math


def cosine_distance(arr0, arr1):
    return np.sum(arr0*arr1)/(np.sqrt(np.sum(arr0*arr0)) * np.sqrt(np.sum(arr1*arr1)))


def get_embeddings_from_image(model, image_path, image_height, image_width):
    if not os.path.isfile(image_path):
        print('Error: {} could not be read'.format(image_path))
        os._exit(1)
    img = cv2.imread(image_path)
    if img.shape != (image_height, image_width, 3):
        img_resized = cv2.resize(img, (image_width, image_height)).clip(0, 255)
    else:
        img_resized = img
    flattened = img_resized.swapaxes(1, 2).swapaxes(0, 1).flatten()
    outputs = model.run([flattened])
    return outputs[0]*model.output_scale_factor[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image1')
    parser.add_argument('image2')
    parser.add_argument('--height', type=int, default=112)
    parser.add_argument('--width', type=int, default=96)
    args = parser.parse_args()

    with open(args.model, 'rb') as mf:
        model = vbx.sim.Model(mf.read())

    image1_out = get_embeddings_from_image(model,args.image1,args.height,args.width)
    image2_out = get_embeddings_from_image(model,args.image2,args.height,args.width)

    print("image similiarity = {:.3f}".format(cosine_distance(image2_out,image1_out)))
    
    bw = model.get_bandwidth_per_run()
    print("Bandwidth per run = {} Bytes ({:.3} MB/s at 100MHz)".format(bw,bw/100E6))    
    print("Estimated {} seconds at 100MHz".format(model.get_estimated_runtime(100E6)))
    print("If running at another frequency, scale these numbers appropriately")
