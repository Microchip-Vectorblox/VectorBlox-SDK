import argparse
import numpy as np
import cv2
import vbx.postprocess.ssd as ssd
import vbx.postprocess.dataset as dataset
import vbx.sim
import os
import math

from vbx.generate.onnx_infer import onnx_infer, load_input


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


def get_vnnx_io_shapes(vnxx):
    with open(vnxx, 'rb') as mf:
        model = vbx.sim.Model(mf.read())
    return model.input_dims[0], model.output_dims


def vnnx_infer(vnxx, input_array):
    with open(vnxx, 'rb') as mf:
        model = vbx.sim.Model(mf.read())
    flattened = input_array.flatten().astype('uint8')
    outputs = model.run([flattened])

    bw = model.get_bandwidth_per_run()
    print("Bandwidth per run = {} Bytes ({:.3} MB/s at 100MHz)".format(bw,bw/100E6))
    print("Estimated {} seconds at 100MHz".format(model.get_estimated_runtime(100E6)))
    print("If running at another frequency, scale these numbers appropriately")

    return [o.astype('float32') * sf for o,sf in zip(outputs, model.output_scale_factor)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    parser.add_argument('-p', '--priors', default='vehicle_priors.npy')
    parser.add_argument('-o', '--output', default="output.png")
    args = parser.parse_args()

    input_shape, _ = get_vnnx_io_shapes(args.model)
    input_array = load_input(args.image, 1., input_shape)
    outputs = vnnx_infer(args.model, input_array)

    reshaped_outputs = []
    reshaped_outputs.append(outputs[11].reshape((1,8,16,16)))
    reshaped_outputs.append(outputs[10].reshape((1,6,16,16)))
    reshaped_outputs.append(outputs[9].reshape((1,12,8,8)))
    reshaped_outputs.append(outputs[8].reshape((1,9,8,8)))
    reshaped_outputs.append(outputs[7].reshape((1,12,4,4)))
    reshaped_outputs.append(outputs[6].reshape((1,9,4,4)))
    reshaped_outputs.append(outputs[5].reshape((1,4,2,2)))
    reshaped_outputs.append(outputs[4].reshape((1,3,2,2)))
    reshaped_outputs.append(outputs[3].reshape((1,4,1,1)))
    reshaped_outputs.append(outputs[2].reshape((1,3,1,1)))
    reshaped_outputs.append(outputs[1].reshape((1,12,1,1)))
    reshaped_outputs.append(outputs[0].reshape((1,9,1,1)))

    priors = np.load(args.priors).reshape((2,-1,4))
    predictions = ssd.predictions(reshaped_outputs, priors, 256, confidence_threshold=0.4, nms_threshold=0.3, top_k=3, num_classes=3)
    
    img = cv2.imread(args.image)
    output_img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    output_scale_h = 1024. / input_shape[1]
    output_scale_w = 1024. / input_shape[2]

    classes = ['unlabeled', 'car', 'plate']
    colors = dataset.coco_colors
    for p in predictions:
        print("{}\t{}\t({}, {}, {}, {})".format(classes[p['class_id']],
                                                int(100*p['confidence']),
                                                int(p['xmin']), int(p['xmax']),
                                                int(p['ymin']), int(p['ymax'])))
        p1 = (int(p['xmin'] * output_scale_w), int(p['ymin'] * output_scale_h))
        p2 = (int(p['xmax'] * output_scale_w), int(p['ymax'] * output_scale_h))
        color = colors[p['class_id']]
        cv2.rectangle(output_img, p1, p2, color, 2)

        p3 = (max(p1[0]-4, 4), max(p1[1]-4, 4))
        class_name = classes[p['class_id']]
        short_name = class_name.split(',')[0]
        cv2.putText(output_img, short_name, p3,
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(args.output, output_img)
