from cmath import rect
import vbx.sim
import argparse
import os
import numpy as np
import cv2
import json
import  vbx.postprocess.retinaface

from vbx.generate.openvino_infer import openvino_infer, openvino_infer_multi, get_model_input_shape as get_xml_input_shape
from vbx.generate.onnx_infer import onnx_infer, onnx_infer_multi, load_input
from vbx.generate.onnx_helper import get_model_input_shape as get_onnx_input_shape


def get_vnnx_io_shapes(vnxx):
    with open(vnxx, 'rb') as mf:
        model = vbx.sim.Model(mf.read())
    return model.input_dims[0], model.output_dims


def vnnx_infer(vnnx_model, input_array):
    model = vbx.sim.model.Model(open(vnnx_model,"rb").read())

    input_array = input_array.astype(np.uint8)
    outputs = model.run([input_array.flatten()])
    outputs = [o/(1<<16) for o in outputs]

    bw = model.get_bandwidth_per_run()
    print("Bandwidth per run = {} Bytes ({:.3} MB/s at 100MHz)".format(bw,bw/100E6))
    print("Estimated {} seconds at 100MHz".format(model.get_estimated_runtime(100E6)))
    print("If running at another frequency, scale these numbers appropriately")

    return outputs


def vnnx_infer_gaze(vnnx_model, input_array1, input_array2, input_angles):
    model = vbx.sim.model.Model(open(vnnx_model,"rb").read())

    input_array1 = input_array1.astype(np.uint8).flatten()
    input_array2 = input_array2.astype(np.uint8).flatten()
    input_angles = input_angles.astype(np.int8).flatten()
    outputs = model.run([input_angles,input_array1,input_array2])
    outputs = [o/(1<<16) for o in outputs]

    bw = model.get_bandwidth_per_run()
    print("Bandwidth per run = {} Bytes ({:.3} MB/s at 100MHz)".format(bw,bw/100E6))
    print("Estimated {} seconds at 100MHz".format(model.get_estimated_runtime(100E6)))
    print("If running at another frequency, scale these numbers appropriately")

    return outputs

def get_rect_and_rotation(face):
    # TODO unify with get_roi in hands.py

    # blazeface reference for our retinaface detector:
    # https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt

    # face[box] = [xmin ymin xmax ymax]
    
    rect = {}

    box = face['box'] # 0,0 is top left corner of image
    landmarks = face['landmarks']

    xmin = box[0]
    xmax = box[2]
    ymin = box[1]
    ymax = box[3]

    rect['width'] = xmax - xmin
    rect['height'] = ymax - ymin
    rect['x_center'] = (xmin + xmax) / 2
    rect['y_center'] = (ymin + ymax) / 2

    x0, y0 = landmarks[0] # left eye keypoint / (eye) left bound of eye
    x1, y1 = landmarks[1] # right eye keypoint / (eye) right bound of eye

    # rotation vector target angle : 0
    target_angle = 0
    rotation = target_angle - np.arctan2(-(y1 - y0), x1 - x0)

    def normalize_radians(angle):
        return angle - 2 * np.pi * np.floor((angle + np.pi) / (2 * np.pi))
    rotation = normalize_radians(rotation)

    return rect, rotation


def transform_rect(rect, rotation, w, h):
    # TODO unify with get_roi in hands.py
    # https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt
    # https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/rect_transformation_calculator.cc

    # mediapipe uses 1.5x scale, intel/openvino uses 1.2x scale, can switch around (those scalings are based on their own networks for face)
    # openvino additionally adjusts the bbox after making the detection
    # https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/gaze_estimation_demo/cpp/src/face_detector.cpp#L41
    scale_x = 1.2
    scale_y = 1.2
    shift_x = 0
    shift_y = 0
    square_long = True

    # we need to account for original img w and h for getting ROI
    # for our retinaface postprocessing, we want to avoid scaling everything by w and h as it is already scaled
    # perhaps may want inputs scaled already to universalize with hands.py
    if rotation == 0:
        rect['x_center'] = (rect['x_center'] + rect['width'] * shift_x)
        rect['y_center'] = (rect['y_center'] + rect['height'] * shift_y)
    else:
        x_shift = (rect['width'] * shift_x * np.cos(rotation) - rect['height'] * shift_y * np.sin(rotation))
        y_shift = (rect['width'] * shift_x * np.sin(rotation) + rect['height'] * shift_y * np.cos(rotation))
        rect['x_center'] = rect['x_center']+ x_shift
        rect['y_center'] = rect['y_center']+ y_shift

    if square_long:
        long_side = max(rect['width'], rect['height'])
        rect['width'] = long_side * scale_x
        rect['height'] = long_side * scale_y
    else:
        short_side = min(rect['width'], rect['height'])
        rect['width'] = short_side * scale_x
        rect['height'] = short_side * scale_y

    return rect


def get_roi_corners(rect, rotation):
    # TODO should be same with get_roi in hands.py, unify
    # get corners of ROI box with center of box translated to 0, 0
    points = []

    # bottom left
    x0 = - rect['width']/2
    y0 = rect['height']/2
    points.append((x0,y0))

    # top left
    x1 = - rect['width']/2
    y1 = - rect['height']/2
    points.append((x1,y1))
    
    # top right
    x2 = rect['width']/2
    y2 = - rect['height']/2
    points.append((x2,y2))

    # bottom right
    x3 = rect['width']/2
    y3 = rect['height']/2
    points.append((x3,y3))

    for idx, point in enumerate(points):
        x_rotated = point[0] * np.cos(rotation) - point[1] * np.sin(rotation)
        y_rotated = point[0] * np.sin(rotation) + point[1] * np.cos(rotation)

        points[idx] = (x_rotated + rect['x_center'], y_rotated + rect['y_center'])

    return points


def get_roi(face, w, h, do_rotation=False):
    # TODO unify with get_roi in hands.py
    rect, rotation = get_rect_and_rotation(face)
    if not do_rotation:
        rotation=0.0
    rect = transform_rect(rect,rotation,w,h)
    roi_points = get_roi_corners(rect,rotation)
    return roi_points


def warp_affine_roi_to_image(img, rect_points, input_len, roi_idx):
    # TODO unify with get_roi in hands.py
    # pass in top left, top right, bottom right roi corners
    # based on https://github.com/geaxgx/openvino_hand_tracker/blob/main/mediapipe_utils.py
    src = np.array(rect_points[1:], dtype=np.float32) # rect_points[0] is left bottom point !
    dst = np.array([(0, 0), (input_len, 0), (input_len, input_len)], dtype=np.float32)
    mat = cv2.getAffineTransform(src, dst)
    warp_dst = cv2.warpAffine(img, mat, (input_len, input_len))

    region_name = "region{}.png".format(roi_idx)
    cv2.imwrite(region_name, warp_dst)
    return region_name


# assumes image has width == height
def model_infer(model, image, input_len):
    if '.vnnx' in model:
        input_shape, _ = get_vnnx_io_shapes(args.model)
        input_array = load_input(image, 1., input_shape)
        outputs = vnnx_infer(model, input_array)
    elif '.xml' in model:
        weights=model.replace('.xml', '.bin')
        input_shape = get_xml_input_shape(model, weights)
        input_array = load_input(image, 1., input_shape)
        outputs = openvino_infer(model, input_array)
    elif '.onnx' in model:
        input_shape = get_onnx_input_shape(model)
        input_array = load_input(image, 1., input_shape)  
        outputs = onnx_infer(model, input_array)

    return outputs


def get_eye_meta(left_lm, right_lm, scale):
    eye_size = cv2.norm(left_lm-right_lm)
    bbox_size = eye_size*scale
    midpoint = (left_lm+right_lm)/2
    bbox_min = midpoint - (bbox_size/2)
    bbox_max = midpoint + (bbox_size/2)
    eye = {'box':np.concatenate([bbox_min,bbox_max]),'landmarks':np.array([left_lm,right_lm])}
    return eye


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=320)
    parser.add_argument('--channels', type=int, default=3) 
    parser.add_argument('-t', '--threshold', type=float, default=0.8)
    parser.add_argument('-nms', '--nms-threshold', type=float, default=0.4)
    parser.add_argument('--age-gender-model', default=None)
    parser.add_argument('--emotion-model',default=None)
    parser.add_argument('--landmark-model',default=None)
    parser.add_argument('--eye-state-model',default=None)
    parser.add_argument('--head-pose-model',default=None)
    parser.add_argument('--gaze-estimation-model',default=None)

    args = parser.parse_args()
    if '.vnnx' in args.model:
        input_shape = (args.channels, args.height, args.width)
        input_array = load_input(args.image, 1., input_shape)
        outputs = vnnx_infer(args.model, input_array)
    elif '.xml' in args.model:
        weights=args.model.replace('.xml', '.bin')
        input_shape = get_xml_input_shape(args.model, weights)
        input_array = load_input(args.image, 1., input_shape)
        outputs = openvino_infer(args.model, input_array)
    elif '.onnx' in args.model:
        input_shape = get_onnx_input_shape(args.model)
        input_array = load_input(args.image, 1., input_shape)  
        outputs = onnx_infer(args.model, input_array)

    faces = vbx.postprocess.retinaface.retinaface(outputs, args.width, args.height,args.threshold, args.nms_threshold)
    img = cv2.imread(args.image)

    if img.shape != (args.height,args.width,args.channels):
        img = cv2.resize(img,(args.width,args.height))

    orig_img_width=img.shape[1]
    orig_img_height=img.shape[0]
    
    output_img = img.copy()
    for f in faces:
        # region(s) of interest formatted as [bottom left, top left, top right, bottom right]
        f['roi'] = get_roi(f, orig_img_width, orig_img_height, do_rotation=False)

        text = "{:.4f}".format(f['score'])
        box = list(map(int, f['box']))

        # [xmin ymin, xmax, ymax]
        cv2.rectangle(output_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cx = box[0]
        cy = box[1] + 12
        cv2.putText(output_img, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        for l in f['landmarks']:
            cv2.circle(output_img, (int(l[0]), int(l[1])), 1, (0, 0, 255), 4)
        print("face found at", *box, 'w/ confidence {:3.4f}'.format(f['score']))
        for l in f['landmarks']:
            print("face feature at",*l)
        print()
    # save image
    print("{} faces found".format(len(faces)))
    name = "test.jpg"
    cv2.imwrite(name, output_img)


    if args.age_gender_model:
        if 'genderage' in args.age_gender_model: # insightface genderage - runs thru post.onnx well
            exp_input_len = 96
        else:   # openvino age-gender-recognition-retail
            exp_input_len = 62
        for idx,face in enumerate(faces):
            cropped_img_name=warp_affine_roi_to_image(img, face['roi'], exp_input_len, idx)
            outputs = model_infer(args.age_gender_model, cropped_img_name, exp_input_len)

            gender_out=outputs[0] if len(outputs[0]) == 2 else outputs[1]
            age_out = outputs[1] if len(outputs[1]) == 1 else outputs[0]
            gender= 'female' if np.argmax(gender_out) == 0 else 'male'
            age = round(age_out[0] * 100)
            print("face {} \ngender:\t {} (female vs. male {} vs. {})".format(idx,gender, gender_out[0], gender_out[1]))
            print("age : {}\n".format(age))


    if args.emotion_model:
        exp_input_len=64
        for idx,face in enumerate(faces):
            cropped_img_name=warp_affine_roi_to_image(img, face['roi'], exp_input_len, idx)
            outputs = model_infer(args.emotion_model, cropped_img_name, exp_input_len)

            # outputs - 1 output vector softmax across 5 classes
            # [neutral, happy, sad, surprise, anger]
            print("face {} emotion scores:".format(idx))
            print("neutral:  {}\nhappy:    {}\nsad:      {}\nsurprise: {}\nanger:    {}\n".format(*outputs[0]))


    if args.head_pose_model:
        exp_input_len=60
        output_img = img.copy()
        for idx,face in enumerate(faces):
            cropped_img_name=warp_affine_roi_to_image(img, face['roi'], exp_input_len, idx)
            outputs = model_infer(args.head_pose_model, cropped_img_name, exp_input_len)

            # xml ordering : pitch roll yaw
            # onnx ordering : likely same as vnnx (need to confirm)
            # vnnx ordering : yaw roll pitch
            if '.vnnx' or '.onnx' in args.head_pose_model:
                yaw = outputs[0][0]
                roll = outputs[1][0]
                pitch = outputs[2][0]
            else:
                pitch = outputs[0][0]
                roll = outputs[1][0]
                yaw = outputs[2][0]

            print("face {} angles in degrees:".format(idx))
            print("yaw: {}\npitch: {}\nroll: {}\n".format(yaw,pitch,roll))

            # based on https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/gaze_estimation_demo/cpp/src/results_marker.cpp
            # convert degrees to radians   
            yaw *= np.pi / 180
            roll *= np.pi / 180
            pitch *= np.pi / 180

            # we will draw axis from nose landmark instead of center of bbox
            x, y = face['landmarks'][2] # nose

            # X-Axis pointing to right. drawn in red
            x1 = x + exp_input_len * (np.cos(roll) * np.cos(yaw) + np.sin(yaw) * np.sin(pitch) * np.sin(roll))
            y1 = y + exp_input_len * np.cos(pitch) * np.sin(roll)

            # Y-Axis pointing up drawn in green
            x2 = x + exp_input_len * (np.cos(roll) * np.sin(yaw) * np.sin(pitch) + np.cos(yaw) * np.sin(roll))
            y2 = y - exp_input_len * np.cos(pitch) * np.cos(roll)

            # Z-Axis (out of the screen) drawn in blue
            x3 = x + exp_input_len * np.sin(yaw) * np.cos(pitch)
            y3 = y + exp_input_len * np.sin(pitch)

            cv2.line(output_img, (int(x), int(y)), (int(x1),int(y1)),(0,0,255),2)
            cv2.line(output_img, (int(x), int(y)), (int(x2),int(y2)),(0,255,0),2)
            cv2.line(output_img, (int(x), int(y)), (int(x3),int(y3)),(255,0,0),2)

            if args.gaze_estimation_model:
                face['yaw'] = yaw
                face['pitch'] = pitch
                face['roll'] = roll

        name = "headpose.jpg"
        cv2.imwrite(name, output_img)
            

    if args.landmark_model:
        if '2d106det' in args.landmark_model: # insightface 2d106det
            exp_input_len = 192
        else: # openvino facial-landmarks-35-adas-0002 (needs softmax onnx norm fix, or cut graph in two)
            exp_input_len = 60

        output_img = img.copy()
        for idx,face in enumerate(faces):
            cropped_img_name=warp_affine_roi_to_image(img, face['roi'], exp_input_len, idx)
                        
            if '2d106det' in args.landmark_model:
                outputs = model_infer(args.landmark_model, cropped_img_name, exp_input_len)

            else: # openvino landmark
                # NOTE softmax has issues during norm step
                outputs = model_infer(args.landmark_model, cropped_img_name, exp_input_len)

                # TODO debug/test code for openvino landmark model splitting
                # xml_model = 'face-lm-softmax.xml'
                # onnx_model = 'keep_temp/face-lm-softmax.post.onnx'
                # vnnx_model = 'face-lm-softmax.vnnx'
                # outputs_a = model_infer(xml_model, cropped_img_name, exp_input_len) 
                # outputs_b = model_infer(onnx_model, cropped_img_name, exp_input_len)
                # outputs_b.reverse()
                # outputs_c = model_infer(vnnx_model, cropped_img_name, exp_input_len)
                # outputs_c.reverse()

                # import pdb; pdb.set_trace()

                # # check if outputs match
                # for i, j in zip(outputs_a, outputs_b):
                #     if np.allclose(i, j, atol=1e-05):
                #         pass
                #     else:
                #         print("fail")

                # # xml outs are not in order - last 8 outs need to be moved to front
                # def my_concat(outputs):
                #     concat_a=np.expand_dims(np.concatenate(outputs[-8:]), axis=(1,2))
                #     concat_b=np.expand_dims(np.concatenate(outputs[:-8]), axis=(1,2))
                #     res = np.concatenate((concat_a, concat_b), axis=0)
                #     return res

                # xml_lm = 'face-lm-concat.xml'
                # onnx_lm = 'keep_temp/face-lm-concat.post.onnx'
                # vnnx_lm = 'face-lm-concat.vnnx'
                # concat_res_a = my_concat(outputs_a)
                # concat_res_b = np.expand_dims(my_concat(outputs_b), axis=0)
                # concat_res_c = np.expand_dims(my_concat(outputs_c), axis=0)
                # outputs = openvino_infer(xml_lm, concat_res_b)
                # # outputs = onnx_infer(onnx_lm, concat_res_b)
                # # outputs = vnnx_infer(vnnx_lm, concat_res_b)

            top_left_x, top_left_y = face['roi'][1]
            bottom_right_x, bottom_right_y = face['roi'][3]
            bbox_w = bottom_right_x-top_left_x
            bbox_h = bottom_right_y-top_left_y

            bbox_w_h = bbox_w/2
            bbox_h_h = bbox_h/2
            center_x = top_left_x + bbox_w_h
            center_y = top_left_y + bbox_h_h
            landmarks = outputs[0].reshape(-1,2)
            face['eye_lms'] = []
            for idx, l in enumerate(landmarks):
                if '2d106det' in args.landmark_model:
                    lm_x = (bbox_w_h * l[0]) + center_x
                    lm_y = (bbox_h_h * l[1]) + center_y
                else:
                    lm_x = (bbox_w * l[0]) + top_left_x
                    lm_y = (bbox_h * l[1]) + top_left_y

                cv2.circle(output_img, (int(lm_x), int(lm_y)), 1, (0, 0, 255), -1)

                # need eye landmarks for eye cropping
                if (args.eye_state_model or args.gaze_estimation_model):
                    if '2d106det' in args.landmark_model: # insightface landmark
                        if idx in [35,39,89,93]:
                            face['eye_lms'].append([lm_x, lm_y])
                    else: # openvino landmark model
                        if idx < 4:
                            face['eye_lms'].append([lm_x, lm_y])
            face['eye_lms'] = np.asarray(face['eye_lms'])

        name = "landmarks.jpg"
        cv2.imwrite(name, output_img)


    if args.eye_state_model:
        if args.landmark_model is None :
            print("need a landmark model in order to run eye state model.")
            return
        exp_input_len=32
        scale = 1.2

        for idx,face in enumerate(faces):
            eye_lms = face['eye_lms']

            if '2d106det' in args.landmark_model: # for insightface lm model
                l_eye=get_eye_meta(eye_lms[0], eye_lms[1], scale) 
            else: # openvino landmark model
                l_eye=get_eye_meta(eye_lms[1], eye_lms[0], scale)
            eye_roi = get_roi(l_eye, orig_img_width, orig_img_height, do_rotation=True)
            cropped_roi_name = warp_affine_roi_to_image(img, eye_roi, exp_input_len, idx*2)
            outputs = model_infer(args.eye_state_model, cropped_roi_name, exp_input_len)
            # softmax 
            exp = np.exp(outputs[0])
            l_eye_state = exp / sum(exp)

            r_eye=get_eye_meta(eye_lms[2],eye_lms[3], scale)
            eye_roi = get_roi(r_eye, orig_img_width, orig_img_height, do_rotation=True)
            cropped_roi_name = warp_affine_roi_to_image(img, eye_roi, exp_input_len, (idx*2)+1)
            outputs = model_infer(args.eye_state_model, cropped_roi_name, exp_input_len)
            # softmax 
            exp = np.exp(outputs[0])
            r_eye_state = exp / sum(exp)

            print("face {} eye states:".format(idx))
            print("left  eye:\t closed:{} \t open:{}".format(*l_eye_state))
            print("right eye:\t closed:{} \t open:{}\n".format(*r_eye_state))


    if args.gaze_estimation_model:
        if args.landmark_model is None or args.head_pose_model is None:
            print("need a landmark model and head pose model in order to run gaze model.")
            return
        exp_input_len=60
        scale=1.2
        output_img = img.copy()
        for idx, face in enumerate(faces):
            eye_lms = face['eye_lms']
            yaw = face['yaw'] * 180 / np.pi # scale back to degrees
            pitch = face['pitch'] * 180 / np.pi
            roll = face['roll'] * 180 / np.pi

            if '2d106det' in args.landmark_model: # for insightface lm model
                l_eye=get_eye_meta(eye_lms[0], eye_lms[1], scale) 
            else: # # if using openvino intel landmark model
                l_eye=get_eye_meta(eye_lms[1], eye_lms[0], scale)
            eye_roi = get_roi(l_eye, orig_img_width, orig_img_height, do_rotation=True)
            l_cropped_roi_name = warp_affine_roi_to_image(img, eye_roi, exp_input_len, idx*2)

            r_eye=get_eye_meta(eye_lms[2],eye_lms[3], scale)
            eye_roi = get_roi(r_eye, orig_img_width, orig_img_height, do_rotation=True)
            r_cropped_roi_name = warp_affine_roi_to_image(img, eye_roi, exp_input_len, (idx*2)+1)

            head_pose_angles = np.array([yaw,pitch,roll])

            input_shape = (1,3,exp_input_len,exp_input_len)
            input_array1 = load_input(l_cropped_roi_name, 1., input_shape)
            input_array2 = load_input(r_cropped_roi_name, 1., input_shape)
            head_pose_angles=np.expand_dims(head_pose_angles, axis=0)

            if '.vnnx' in args.gaze_estimation_model:
                outputs = vnnx_infer_gaze(args.gaze_estimation_model, input_array1, input_array2, head_pose_angles)
            elif '.xml' in args.gaze_estimation_model:
                # NOTE input names changed accordingly depending on openvino xml
                xml_input_feed = {'left_eye_image':input_array1, 'right_eye_image':input_array2, 'head_pose_angles':head_pose_angles}
                outputs = openvino_infer_multi(args.gaze_estimation_model, xml_input_feed)
            elif '.onnx' in args.gaze_estimation_model:
                # NOTE input names changed accordingly depending on onnx
                onnx_input_feed = {'2:0':input_array1, '1:0':input_array2, '0:0':head_pose_angles.astype(np.float32)}
                outputs = onnx_infer_multi(args.gaze_estimation_model, onnx_input_feed)

            gaze_vector = outputs[0] / cv2.norm(outputs[0])

            gaze_length = exp_input_len * 0.8
            gaze_x = int(gaze_vector[0] * gaze_length)
            gaze_y = int(-gaze_vector[1] * gaze_length)
            
            l_midpoint = ((eye_lms[0]+eye_lms[1])/2).astype(np.int)
            r_midpoint = ((eye_lms[2]+eye_lms[3])/2).astype(np.int)

            cv2.arrowedLine(output_img, l_midpoint, l_midpoint+(gaze_x,gaze_y), (255,0,0), 1, tipLength=0.3)
            cv2.arrowedLine(output_img, r_midpoint, r_midpoint+(gaze_x,gaze_y), (255,0,0), 1, tipLength=0.3)

        name = "gaze.jpg"
        cv2.imwrite(name, output_img)

if __name__ == "__main__":
    main()
