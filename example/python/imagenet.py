import vbx.sim
import vbx.postprocess.dataset as dataset
import vbx.postprocess.classifier as classifier
import cv2
import numpy as np
import argparse
import os
import math

def convert_to_fixedpoint(data, dtype):
    # this should go away eventually, and always input uint8 rather than fixedpoint Q1.7
    if dtype == np.int16:
        shift_amt = 13
    elif dtype == np.int8:
        shift_amt = 7
    clip_max, clip_min = (1 << shift_amt)-1, -(1 << shift_amt)
    float_img = data.astype(np.float32)/255 * (1 << shift_amt) + 0.5

    fixedpoint_img = np.clip(float_img, clip_min, clip_max).astype(dtype)
    return fixedpoint_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    parser.add_argument('--output', '-o', default="output.png")
    args = parser.parse_args()

    with open(args.model, 'rb') as mf:
        model = vbx.sim.Model(mf.read())

    input_size = int(math.sqrt(model.input_lengths[0]/3))
    input_dtype = model.input_dtypes[0]
    if not os.path.isfile(args.image):
        print('Error: {} could not be read'.format(args.image))
        os._exit(1)
    img = cv2.imread(args.image)
    if img.shape != (input_size, input_size, 3):
        img_resized = cv2.resize(img, (input_size, input_size)).clip(0, 255)
    else:
        img_resized = img
    flattened = img_resized.swapaxes(1, 2).swapaxes(0, 1).flatten()
    if input_dtype != np.uint8:
        flattened = convert_to_fixedpoint(flattened, input_dtype)

    outputs = model.run([flattened])

    scaled_outputs = outputs[0].astype(np.float32)*model.output_scale_factor[0]
    sorted_classes = classifier.topk(scaled_outputs)

    i = 0
    assert(len(scaled_outputs) in (1001,1000))
    if len(scaled_outputs)==1001:
        classes = dataset.imagenet_classes_with_nul
    else:
        classes = dataset.imagenet_classes
    output_img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    for cls, score in list(zip(*sorted_classes))[:3]:
        class_name = classes[cls]
        short_name = class_name.split(',')[0]
        print(cls, short_name, score)

        p3 = (4, (i+1)*(32+4))
        cv2.putText(output_img, '{} {}'.format(cls, short_name), p3,
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        i += 1

    cv2.imwrite(args.output, output_img)
    bw = model.get_bandwidth_per_run()
    print("Bandwidth per run = {} Bytes ({:.3} MB/s at 100MHz)".format(bw,bw/100E6))
    print("Estimated {} seconds at 100MHz".format(model.get_estimated_runtime(100E6)))
    print("If running at another frequency, scale these numbers appropriately")
