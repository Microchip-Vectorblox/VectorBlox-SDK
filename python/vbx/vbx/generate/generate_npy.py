import os
import numpy as np
import cv2
import random
import argparse

'''
This script is used to generate numpy array files (.npy) for use in TF Lite calibration.
If user wants to calibrate with their own data, they can do so by using this script to generate a numpy array.
'''

# Example usage:
# python generate_npy.py [path_to_data_dir] --count 20 --shape 227 227 --output_name imagenetv2_20x227x227x3.npy

random_seed = 342

def read_images(directory_path, num_images, rgb=False, grayscale=False, debug=False):
    random.seed(random_seed)

    files = os.listdir(directory_path)

    image_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))]

    selected_images = random.sample(image_files, min(num_images, len(image_files)))
    if debug:
        print(selected_images)

    images = []
    for image_file in selected_images:
        image_path = os.path.join(directory_path, image_file)
        image = cv2.imread(image_path)
        if grayscale:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)

    return images

def preprocess_images(images, shape, grayscale=False, norm=False):
    height = shape[0]
    width = shape[1]
    if grayscale:
        images_array = np.zeros((len(images), height, width), dtype=np.float32)
    else:
        images_array = np.zeros((len(images), height, width, 3), dtype=np.float32)

    for i, image in enumerate(images):
        resized = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        if norm:
            resized = resized/255.
        images_array[i] = resized
    return images_array

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir')
    parser.add_argument('--output_name', type=str, default='')
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--shape', nargs=2, type=int, default=[224,224]) # height width
    parser.add_argument('--bgr', action='store_true')
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    images = read_images(args.image_dir, args.count, (not args.bgr), args.grayscale, args.debug)
    images_array = preprocess_images(images, args.shape, args.grayscale, args.norm)
    if args.debug:
        print("Shape of the numpy array:", images_array.shape)

    channels = 3
    if args.grayscale:
        channels = 1

    if args.output_name == '':
        bgr = ''
        norm = ''
        if args.bgr:
            bgr = 'bgr_'
        if args.norm:
            norm = 'norm_'
        args.output_name = 'tflite_calibration_images_{}{}{}x{}x{}x{}.npy'.format(bgr,norm,args.count,args.shape[0],args.shape[1],channels)
    np.save(args.output_name, images_array)

if __name__ == '__main__':
    main()
