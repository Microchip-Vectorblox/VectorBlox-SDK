/*!
 * \file
 * \brief Sample post-processing code for popular model types: (yolo, imagenet, mnist)
 */

#ifndef __POSTPROCESS_H_
#define __POSTPROCESS_H_
#define Q32 16
#define Q16 13
#define Q8 7
#define U8 8

#include "libfixmath/fixmath.h"
#include "vbx_cnn_api.h"
#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

//#define fix16_exp fix16_exp_lut
fix16_t fix16_logistic_activate(fix16_t x);

typedef unsigned char uchar;

#define OBJECT_NAME_LENGTH 12
typedef struct {
    fix16_t box[4];
    fix16_t points[6][2];
    fix16_t detect_score;
    fix16_t recognition_score;
    char name[OBJECT_NAME_LENGTH];
    short track_val;
} object_t;

typedef struct {
    fix16_t keypoints[17][2];
    fix16_t scores[17];
    fix16_t poseScore;
} poses_t;

typedef struct {
    int points[2];
    fix16_t scores;
    int id;
} queue_element_t;

typedef struct {
    int xmin;
    int xmax;
    int ymin;
    int ymax;
    int x;
    int y;
    int w;
    int h;
    fix16_t angle;
    fix16_t confidence;
    int class_id;
    const char* class_name;
} fix16_box;

typedef struct {
    int version;
    int input_dims[3];
    int output_dims[3];
    int coords;
    int classes;
    int num;
    int anchors_length;
    fix16_t* anchors;
    int mask_length;
    int* mask;
} yolo_info_t;

void reverse(fix16_t* output_buffer[], int len);
uint32_t fletcher32(const uint16_t *data, size_t len);
void print_json(model_t* model,vbx_cnn_io_ptr_t* io_buffers,int use_int8);
void preprocess_inputs(uint8_t* input, fix16_t scale, int32_t zero_point, int input_length,int int8_flag);
typedef void (*file_write)(const char*,int);
extern char *imagenet_classes[];
void post_process_classifier(fix16_t *outputs, const int output_size, int16_t* output_index, int topk);

/**
 * @brief Post processing for classifiers such as MNIST or ImageNet networks. Returns the topk indices of the
outputs, sorted from highest to lowest
 * 
 * @param outputs Unsorted outputs obtained from the model
 * @param output_size Number of outputs
 * @param[out] output_index The retrned indicies, sorted by scores lowest to highest
 * @param topk Number of indices to return sorted
 */
void post_process_classifier_int8(int8_t *outputs, const int output_size, int16_t* output_index, int topk);
/**
 * @brief Performs non-maximal suppression on detected objects
 * 
 * @param output Array of boxes obtained from postprocessing
 * @param output_boxes Max number of boxes for outputs
 * @param input_h Input height of the model
 * @param input_w Input width of the model
 * @param f16_scale Scale value multiplierin fix16 format for int8
 * @param zero_point Mean offset for int8
 * @param thresh Condifence threshold, ranged from 0-1.0
 * @param overlap Non-max suppression fo overlapping detections, ranging from 0-1.0
 * @param fix16_boxes Boxes obtained from non-maximal suppression
 * @param boxes_len Max length of boxes to be returned
 * @param num_classes Number of classes for detection
 * @return int Number of valid detections
 */
int post_process_ultra_nms_int8(int8_t *output, int output_boxes, int input_h, int input_w,fix16_t f16_scale, int32_t zero_point, fix16_t thresh, fix16_t overlap, fix16_box fix16_boxes[], int boxes_len, const int num_classes);
/**
 * @brief Performs non-maximal suppression on detected objects
 * 
 * @param output Array of boxes obtained from postprocessing
 * @param output_boxes Max number of boxes for outputs
 * @param input_h Input height of the model
 * @param input_w Input width of the model
 * @param thresh Condifence threshold, ranged from 0-1.0
 * @param overlap Non-max suppression fo overlapping detections, ranging from 0-1.0
 * @param fix16_boxes Boxes obtained from non-maximal suppression
 * @param poses Poses obtained from non-maximal suppression if pose
 * @param boxes_len Max length of boxes to be returned
 * @param num_classes Number of classes for detection
 * @param is_obb Check if nms is done for oriented-bounding boxes
 * @param is_pose Check if nms is done for pose detection
 * @return int Number of valid detections
 */
int post_process_ultra_nms(fix16_t *output, int output_boxes, int input_h, int input_w, fix16_t thresh, fix16_t overlap, fix16_box fix16_boxes[], poses_t poses[], int boxes_len, const int num_classes, const int is_obb, const int is_pose);

/**
 * @brief Returns number of detected objects and their respective classes
 * 
 * @param outputs Output buffers obtained from VectorBlox
 * @param outputs_shape Index of the shapes corresponding to the outputs
 * @param post Boxes obtained from postprocessing
 * @param thresh Condifence threshold, ranged from 0-1.0
 * @param zero_points Mean offset for int8
 * @param scale_outs fix16 scaling multiplier for int8
 * @param max_boxes Max number of boxes 
 * @param is_obb Check if postprocess is done for oriented-bounding boxes
 * @param is_pose Check if postprocess is done for pose detection
 * @param num_outputs Number of outputs the model contains
 * @return int Number of detected boxes
 */
int post_process_ultra_int8(int8_t **outputs, int* outputs_shape[], fix16_t *post, fix16_t thresh, int zero_points[], fix16_t scale_outs[], const int max_boxes, const int is_obb, const int is_pose, int num_outputs);
/**
 * @brief Post-processing for Yolov2/V3/V4/V5. Returns number of detected objects, along with  boxes containing coordinates, confidence, and class information.
 * 
 * @param outputs Output buffers obtained from VectorBlox
 * @param num_outputs Number of outputs the model contains
 * @param zero_points Mean offset for int8
 * @param scale_outs fix16 scaling multiplier for int8
 * @param cfg Configuration containing output sizes and anchors
 * @param thresh Confidence threshold, ranged from 0-1.0
 * @param overlap IOU overlap threshold, ranged from 0-1.0
 * @param fix16_boxes Array of object detectiong boxes, one per detection
 * @param max_boxes Limit to the number of possible detection boxes found
 * @return int Number of detected boxes
 */
int post_process_yolo_int8(int8_t **outputs, const int num_outputs, int zero_points[], fix16_t scale_outs[], 
	 yolo_info_t *cfg, fix16_t thresh, fix16_t overlap, fix16_box fix16_boxes[], int max_boxes);
int post_process_yolo(fix16_t **outputs, const int num_outputs, yolo_info_t *cfg,
                      fix16_t thresh, fix16_t overlap, fix16_box fix16_boxes[], int max_boxes);

extern char *dota_classes[15];
extern char *imagenet_classes[1000];
extern char *voc_classes[20];
extern char *coco_classes[80];
extern char* coco91_classes[92];
extern char* vehicle_classes[3];

fix16_t calcIou_LTRB(fix16_t* A, fix16_t* B);
fix16_t calcIou_XYWH(fix16_t* A, fix16_t* B);
void fix16_softmax(fix16_t *input, int n, fix16_t *output);
void fix16_do_nms(fix16_box *boxes, int total, fix16_t iou_thresh);
int fix16_clean_boxes(fix16_box *boxes, poses_t *poses, int total, int width, int height);
void fix16_sort_boxes(fix16_box *boxes, poses_t *poses, int total);
void privacy_draw(int split);
void pixel_draw(model_t *model, vbx_cnn_io_ptr_t* o_buffers, vbx_cnn_t *the_vbx_cnn);
int post_process_blazeface(object_t faces[],fix16_t* scores,fix16_t* points,int scoresLength,int max_faces, fix16_t anchorsScale);
int post_process_retinaface(object_t faces[],int max_faces, fix16_t *network_outputs[9],int image_width,int image_height,
                            fix16_t confidence_threshold, fix16_t nms_threshold);
int post_process_scrfd(object_t faces[], int max_faces, fix16_t *network_outputs[9], int image_width, int image_height, 
                            fix16_t confidence_threshold, fix16_t nms_threshold);
/**
 * @brief Postprocessing on detected objects that stores bounding box and keypoints within the face object
 * 
 * @param faces Array of faces found, one per detection
 * @param max_faces Max number of faces that can be found in a model
 * @param network_outputs Output buffers obtained from VectorBlox
 * @param zero_points Mean offset for int8
 * @param scale_outs fix16 scaling multiplier for int8
 * @param image_width Width of input image sent to the network
 * @param image_height Height of input image sent to the network
 * @param confidence_threshold Confidence threshold, ranged from 0-1.0
 * @param nms_threshold Non-max suppression for overlapping detectiongs, ranged from 0-1.0
 * @param model Network model
 * @return * int Number of detected objects
 */
int post_process_scrfd_int8(object_t faces[],int max_faces, int8_t *network_outputs[9],int zero_points[], fix16_t scale_outs[], 
                            int image_width, int image_height,fix16_t confidence_threshold, fix16_t nms_threshold, model_t *model);
int post_process_lpd_int8(object_t objects[],int max_objects, int8_t *detectOutputs[9], int image_width, int image_height,
					fix16_t confidence_threshold, fix16_t nms_threshold,int detectNumOutputs, int zero_points[], fix16_t scale_outs[]);
int post_process_lpd(object_t plates[],int max_plates, fix16_t *detectOutputs[9], int image_width, int image_height,
                            fix16_t confidence_threshold, fix16_t nms_threshold,int detectNumOutputs);
void int8_to_fix16(fix16_t* output, int8_t* input, int size, fix16_t f16_scale, int32_t zero_point);
fix16_t int8_to_fix16_single(int8_t input,fix16_t scale, int32_t zero_point);
fix16_t post_process_lpr_int8(int8_t *output, model_t *model, char *label);
fix16_t post_process_lpr(fix16_t *output, int output_length, char *label);

int post_process_ssdv2(fix16_box *boxes, int max_boxes,
                       fix16_t *network_outputs[12], int num_classes,
                       fix16_t confidence_threshold, fix16_t nms_threshold);
/**
 * @brief Post-processing for SSD V2. Returns number of detected objects, along with boxes, confidence, and class information
 * 
 * @param boxes Array of object detciong boxes, one per detection
 * @param max_boxes Max number of boxes that can be found
 * @param network_outputs Output buffers obtained from VectorBlox
 * @param network_scales fix16 scaling multipliers for int8
 * @param network_zeros Mean offsets for int8
 * @param num_classes Number of classes in the model
 * @param confidence_threshold Confidence threshold, from 0-1.0
 * @param nms_threshold Non-max suppression overlap threshold, from 0-1.0
 * @return * int Number of detected objects
 */
int post_process_ssd_torch_int8(fix16_box *boxes, int max_boxes, 
                       int8_t *network_outputs[12],
                       fix16_t network_scales[12],
                       int32_t network_zeros[12],
		       int num_classes,
                       fix16_t confidence_threshold, fix16_t nms_threshold);
int post_process_ssd_torch(fix16_box *boxes, int max_boxes,
                       fix16_t *network_outputs[12], int num_classes,
                       fix16_t confidence_threshold, fix16_t nms_threshold);
int post_process_vehicles(fix16_box *boxes, int max_boxes,
                       fix16_t *network_outputs[12], int num_classes,
                       fix16_t confidence_threshold, fix16_t nms_threshold);   
int decodeMultiplePoses(poses_t poses[],fix16_t *scores, fix16_t *offsets,fix16_t *displacementsFwd, fix16_t *displacementsBwd, int outputStride,int maxPoseDetections, fix16_t scoreThreshold,int nmsRadius,fix16_t minPoseScore,int height, int width);

/**
 * @brief Post processing for posenet networks that should return number of detected poses with keypoint, score, and displacement information
 * 
 * @param poses Array of poses found
 * @param scores Output buffer from model
 * @param offsets Output buffer from model
 * @param displacementsFwd Output buffer from model
 * @param displacementsBwd Output buffer from model
 * @param outputStride Stride used from model
 * @param maxPoseDetections Max number of detected poses
 * @param scoreThreshold Score threshold ranged from 0-1.0
 * @param nmsRadius Non-maximal suppression radius in pixels
 * @param minPoseScore Threshold for individual points ranged from 0-1.0
 * @param height Height of model output
 * @param width Width of model output
 * @param zero_points Mean offset for int8
 * @param scale_outs fix16 scaling multiplier for int8
 * @return int Number of detected poses
 */
int decodeMultiplePoses_int8(poses_t poses[],int8_t *scores, int8_t *offsets,int8_t *displacementsFwd, int8_t *displacementsBwd, int outputStride,int maxPoseDetections, fix16_t scoreThreshold,int nmsRadius, fix16_t minPoseScore,int height, int width, int zero_points[], fix16_t scale_outs[]);

/**
 * @brief Post process wrapper function
 * 
 * @param name name of the model
 * @param pptype model postprocessing type
 * @param model model object
 * @param o_buffers output buffers
 * @param int8_flag indicate whether the output is int8 or not
 * @param fps FPS number calculated from main run function or set if not in video demo
 * @param the_vbx_cnn vectorblox cnn object
 * @return int return 0 if successfully able to run postprocessing, -1 if postprocessing type is not valid.
 */
int pprint_post_process(const char *name, const char *pptype, model_t *model, fix16_t **o_buffers,int int8_flag, int fps);

#ifdef __cplusplus
}
#endif

#endif // __POSTPROCESS_H_
