import argparse
import numpy as np
import cv2
import vbx.postprocess.ssd as ssd
import vbx.postprocess.dataset as dataset
import vbx.sim
import os
import math
from vbx.generate.utils import load_input



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    parser.add_argument('-b', '--bgr', action='store_true')
    parser.add_argument('-p', '--priors', default='vehicle_priors.npy')
    parser.add_argument('-o', '--output', default="output.png")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if args.model.endswith('.vnnx'):
        with open(args.model, 'rb') as mf:
            model = vbx.sim.Model(mf.read())
        input_shape = model.input_shape[0]
        img_w, img_h = input_shape[-1], input_shape[-2]
        img = cv2.resize(img, (img_w, img_h))
        if not args.bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32)
        img_scaled = (img / model.input_scale_factor[0]) + model.input_zeropoint[0]
        flattened = img_scaled.swapaxes(1, 2).swapaxes(0, 1).flatten().astype(model.input_dtypes[0])

        outputs = model.run([flattened])
        for o,output in enumerate(outputs):
            outputs[o] = model.output_scale_factor[o] * (outputs[o].astype(np.float32) - model.output_zeropoint[o])
            outputs[o] = outputs[o].reshape(model.output_shape[o])

    elif args.model.endswith('.tflite'):
        import tensorflow as tf
        interpreter= tf.lite.Interpreter(model_path=args.model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_shape = input_details[0]['shape']
        img_w, img_h = input_shape[-2], input_shape[-3]
        img = cv2.resize(img, (img_w, img_h))
        if not args.bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)
        input_scale, input_zero_point = input_details[0].get('quantization', (0.0, 0))
        if  input_scale != 0.0:
            img = (img / input_scale) + input_zero_point
        img = img.astype(input_details[0]['dtype'])
        print(img[0,:8])
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        outputs = []
        for o in range(len(output_details)):
            output_scale, output_zero_point = output_details[o].get('quantization', (0.0, 0))
            output = interpreter.get_tensor(output_details[o]['index']).squeeze()
            if  output_scale != 0.0:
                output = output_scale * (output.astype(np.float32) - output_zero_point)
            while len(output.shape) < 4:
                output = np.expand_dims(output, axis=0)
            outputs.append(output.transpose((0,3,1,2)))

    priors = np.load(args.priors).reshape((2,-1,4))

    target_shapes = [(1,8,16,16),
                    (1,6,16,16),
                    (1,12,8,8),
                    (1,9,8,8),
                    (1,12,4,4),
                    (1,9,4,4),
                    (1,4,2,2),
                    (1,3,2,2),
                    (1,4,1,1),
                    (1,3,1,1),
                    (1,12,1,1),
                    (1,9,1,1)]

    reshaped_outputs = []
    for shape in target_shapes:
        for o, output in enumerate(outputs):
            if output.shape == shape:
                reshaped_outputs.append(outputs[o])

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
