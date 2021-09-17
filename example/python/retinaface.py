import vbx.sim
import argparse
import os
import numpy as np
import cv2
import json
import  vbx.postprocess.retinaface


def openvino_infer(model_file, image):
    import openvino.inference_engine as ie
    weights=model_file.replace('.xml', '.bin')
    core = ie.IECore()
    net = core.read_network(model=model_file, weights=weights)
    assert(len(net.input_info) == 1)
    i0 = [k for k in net.input_info.keys()][0]
    outputs = [k for k in net.outputs.keys()]

    exec_net = core.load_network(network=net, device_name="CPU")
    input_size=exec_net.requests[0].input_blobs[i0].buffer.shape[-1]
    img = cv2.imread(image)
    if img.shape != (input_size,input_size,3):
        img = cv2.resize(img,(input_size,input_size))
    input_array = img.swapaxes(1, 2).swapaxes(0, 1).astype(np.float32)
    input_array = np.expand_dims(input_array, axis=0)
    exec_net.requests[0].input_blobs[i0].buffer[:] = input_array
    exec_net.requests[0].infer()
    return [exec_net.requests[0].output_blobs[o].buffer.flatten() for o in outputs]


def onnx_infer(model_file, image, scale=1.0, shift=0.0, io=None):
    import onnxruntime
    session = onnxruntime.InferenceSession(model_file, None)
    input_name = session.get_inputs()[0].name
    input_size = session.get_inputs()[0].shape[-1]

    if not io is None:
        with open(io) as f:
            io_dict = json.load(f)
            output_scale_factors = io_dict['output_scale_factors']
            input_scale_factors = io_dict['input_scale_factors']
    if '.npy' in image:
        input_array = np.load(image)
    else:
        img = cv2.imread(image)
        if img.shape != (input_size,input_size,3):
            img = cv2.resize(img,(input_size,input_size))
        input_array = img.swapaxes(1, 2).swapaxes(0, 1).astype(np.float32)
        if not(io is None):
            input_array /= input_scale_factors[0]
        else:
            input_array = (input_array / scale) - shift

    input_array = np.expand_dims(input_array, axis=0)
    if io is None:
        return [o.flatten() for o in session.run([], {input_name: input_array})]
    else:
        return [o.flatten() * sf for o,sf in zip(session.run([], {input_name: input_array}), output_scale_factors)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    parser.add_argument('-a', '--anchors')
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=320)
    parser.add_argument('-s', '--scale', type=float, default=255.)
    parser.add_argument('-sh', '--shift', type=float, default=0.)
    parser.add_argument('-t', '--threshold', type=float, default=0.8)
    parser.add_argument('-nms', '--nms-threshold', type=float, default=0.4)
    parser.add_argument('-tn', '--transpose', action='store_true')
    parser.add_argument('--io')

    args = parser.parse_args()

    if '.onnx' in args.model:
        outputs = onnx_infer(args.model, args.image, args.scale, args.shift, args.io)
    else:
        model = vbx.sim.model.Model(open(args.model,"rb").read())
        input_dtype = model.input_dtypes[0]

        image = cv2.imread(args.image).astype(np.float32)
        if image.shape != (args.height,args.width,3):
            image = cv2.resize(image,(args.width,args.height))
        input_array = image.transpose(2,0,1)
        input_array = (input_array / args.scale) - args.shift
        input_array = (input_array * 255.).astype(np.uint8)

        outputs = model.run([input_array.flatten()])
        outputs = [o/(1<<16) for o in outputs]
    faces = vbx.postprocess.retinaface.retinaface(outputs, args.width, args.height,args.threshold, args.nms_threshold)
    img = cv2.imread(args.image)
    if img.shape != (args.height,args.width,3):
        img = cv2.resize(img,(args.width,args.height))

    for f in faces:
        text = "{:.4f}".format(f['score'])
        box = list(map(int, f['box']))

        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cx = box[0]
        cy = box[1] + 12
        cv2.putText(img, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        for l in f['landmarks']:
            cv2.circle(img, (int(l[0]), int(l[1])), 1, (0, 0, 255), 4)
        print("face found at", *box, 'w/ confidence {:3.4f}'.format(f['score']))
        for l in f['landmarks']:
            print("face feature at",*l)
        print()
    # save image
    print("{} faces found".format(len(faces)))
    name = "test.jpg"
    cv2.imwrite(name, img)


    if '.vnnx' in args.model:
        bw = model.get_bandwidth_per_run()
        print("Bandwidth per run = {} Bytes ({:.3} MB/s at 100MHz)".format(bw,bw/100E6))
        print("Estimated {} seconds at 100MHz".format(model.get_estimated_runtime(100E6)))
        print("If running at another frequency, scale these numbers appropriately")


if __name__ == "__main__":
    main()
