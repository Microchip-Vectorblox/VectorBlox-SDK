import vbx.sim
import vbx.postprocess.classifier as classifier
import cv2
import numpy as np
import argparse
import os




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    parser.add_argument('--output', '-o', default="output.png")
    args = parser.parse_args()

    with open(args.model, 'rb') as mf:
        model = vbx.sim.Model(mf.read())

    input_size = 28
    input_dtype = model.input_dtypes[0]
    if not os.path.isfile(args.image):
        print('Error: {} could not be read'.format(args.image))
        os._exit(1)
    img = cv2.imread(args.image)
    if img.shape != (input_size, input_size, 1):
        img_resized = cv2.resize(img, (input_size, input_size)).clip(0, 255)
    else:
        img_resized = img
    flattened = img_resized.swapaxes(1, 2).swapaxes(0, 1).flatten()

    outputs = model.run([flattened])

    scaled_outputs = outputs[0].astype(np.float32)*model.output_scale_factor[0]
    sorted_classes = classifier.topk(scaled_outputs)

    i = 0
    output_img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    for cls, score in list(zip(*sorted_classes))[:3]:
        print("{} {}".format(cls, score))

        p3 = (4, (i+1)*(32+4))
        cv2.putText(output_img, '{}'.format(cls), p3,
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        i += 1

    cv2.imwrite(args.output, output_img)
    print("bandwidth per run = {}".format(model.get_bandwidth_per_run()))
    print("estimated {} seconds at 100MHz".format(model.get_estimated_runtime(100E6)))
