import argparse
import numpy as np
import cv2
import vbx.postprocess.ssd as ssd
import vbx.postprocess.dataset as dataset
import vbx.sim
import os
import math


def convert_to_fixedpoint(data, dtype):
    # this should go away eventually, and always input uint8 rather than fixedpoint Q1.7
    if dtype == np.int16:
        shift_amt = 13
    elif dtype == np.int8:
        shift_amt = 7
    clip_max, clip_min = (1 << shift_amt)-1, -(1 << shift_amt)
    float_img = flattened.astype(np.float32)/255 * (1 << shift_amt) + 0.5

    fixedpoint_img = np.clip(float_img, clip_min, clip_max).astype(dtype)
    return fixedpoint_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    parser.add_argument('-o', '--output', default="output.png")
    args = parser.parse_args()

    with open(args.model, "rb") as mf:
        model = vbx.sim.Model(mf.read())
    input_size = 300
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

    predictions = ssd.ssdv2_predictions(outputs, model.output_scale_factor, confidence_threshold=0.5, nms_threshold=0.4, top_k=1)
    
    output_img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    output_scale = 1024. / 300

    classes = ssd.coco91
    colors = dataset.coco_colors
    for p in predictions:
        print("{}\t{}\t({}, {}, {}, {})".format(classes[p['class_id']],
                                                int(100*p['confidence']),
                                                int(p['xmin']), int(p['xmax']),
                                                int(p['ymin']), int(p['ymax'])))
        p1 = (int(p['xmin'] * output_scale), int(p['ymin'] * output_scale))
        p2 = (int(p['xmax'] * output_scale), int(p['ymax'] * output_scale))
        color = colors[p['class_id']]
        cv2.rectangle(output_img, p1, p2, color, 2)

        p3 = (max(p1[0]-4, 4), max(p1[1]-4, 4))
        class_name = classes[p['class_id']]
        short_name = class_name.split(',')[0]
        cv2.putText(output_img, short_name, p3,
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(args.output, output_img)
