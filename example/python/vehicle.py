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
    parser.add_argument('-p', '--priors', default='vehicle_priors.npy')
    parser.add_argument('-o', '--output', default="output.png")
    args = parser.parse_args()

    with open(args.model, "rb") as mf:
        model = vbx.sim.Model(mf.read())
    input_size = 256
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
    outputs = [o/(1<<16) for o in outputs]

    reshaped_outputs = []
    reshaped_outputs.append(outputs[10].reshape((1,8,16,16)))
    reshaped_outputs.append(outputs[11].reshape((1,6,16,16)))
    reshaped_outputs.append(outputs[8].reshape((1,12,8,8)))
    reshaped_outputs.append(outputs[9].reshape((1,9,8,8)))
    reshaped_outputs.append(outputs[6].reshape((1,12,4,4)))
    reshaped_outputs.append(outputs[7].reshape((1,9,4,4)))
    reshaped_outputs.append(outputs[4].reshape((1,4,2,2)))
    reshaped_outputs.append(outputs[5].reshape((1,3,2,2)))
    reshaped_outputs.append(outputs[2].reshape((1,4,1,1)))
    reshaped_outputs.append(outputs[3].reshape((1,3,1,1)))
    reshaped_outputs.append(outputs[0].reshape((1,12,1,1)))
    reshaped_outputs.append(outputs[1].reshape((1,9,1,1)))

    priors = np.load(args.priors).reshape((2,-1,4))
    predictions = ssd.predictions(reshaped_outputs, priors, 256, confidence_threshold=0.4, nms_threshold=0.45, top_k=400, num_classes=3)
    
    output_img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    output_scale = 1024. / input_size

    classes = ['unlabeled', 'car', 'plate']
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
    print("bandwidth per run = {}".format(model.get_bandwidth_per_run()))
    print("estimated {} ms at 133MHz".format(model.get_estimated_runtime(133*1E6)*1000))
