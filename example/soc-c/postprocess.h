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
#define PDMA 1
//#define fix16_exp fix16_exp_lut
fix16_t fix16_logistic_activate(fix16_t x);


typedef struct {
    int xmin;
    int xmax;
    int ymin;
    int ymax;
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
void preprocess_inputs(uint8_t* input, fix16_t scale, int32_t zero_point, int input_length,int int8_flag);
typedef void (*file_write)(const char*,int);
extern char *imagenet_classes[];
void post_process_classifier(fix16_t *outputs, const int output_size, int16_t* output_index, int topk);
void post_process_classifier_int8(int8_t *outputs, const int output_size, int16_t* output_index, int topk);
//int post_process_ultra_nms_uint8(uint8_t *output, int input_h, int input_w,fix16_t f16_scale, int32_t zero_point, fix16_t thresh, fix16_t overlap, fix16_box fix16_boxes[], int max_boxes);
int post_process_ultra_nms_int8(int8_t *output, int output_boxes, int input_h, int input_w,fix16_t f16_scale, int32_t zero_point, fix16_t thresh, fix16_t overlap, fix16_box fix16_boxes[], int boxes_len);
int post_process_ultra_nms(fix16_t *output, int output_boxes, int input_h, int input_w, fix16_t thresh, fix16_t overlap, fix16_box fix16_boxes[], int boxes_len);
int post_process_ultra_int8(int8_t **outputs, int* outputs_shape[], fix16_t *post, fix16_t thresh, int zero_points[], fix16_t scale_outs[], const int max_boxes);
//int post_process_ultra(fix16_t **outputs, fix16_t *post, fix16_t thresh);
int post_process_yolo_int8(int8_t **outputs, const int num_outputs, int zero_points[], fix16_t scale_outs[], 
	 yolo_info_t *cfg, fix16_t thresh, fix16_t overlap, fix16_box fix16_boxes[], int max_boxes);
int post_process_yolo(fix16_t **outputs, const int num_outputs, yolo_info_t *cfg,
                      fix16_t thresh, fix16_t overlap, fix16_box fix16_boxes[], int max_boxes);

extern char *imagenet_classes[1000];
extern char *voc_classes[20];
extern char *coco_classes[80];
extern char* coco91_classes[92];
extern char* vehicle_classes[3];

typedef unsigned char uchar;

#define NAME_LENGTH 12
typedef struct {
    fix16_t box[4];
    fix16_t points[6][2];
    fix16_t detect_score;
    fix16_t recognition_score;
    char name[NAME_LENGTH];
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

fix16_t calcIou_LTRB(fix16_t* A, fix16_t* B);
fix16_t calcIou_XYWH(fix16_t* A, fix16_t* B);
void fix16_softmax(fix16_t *input, int n, fix16_t *output);
void fix16_do_nms(fix16_box *boxes, int total, fix16_t iou_thresh);
int fix16_clean_boxes(fix16_box *boxes, int total, int width, int height);
void fix16_sort_boxes(fix16_box *boxes, int total);

int post_process_blazeface(object_t faces[],fix16_t* scores,fix16_t* points,int scoresLength,int max_faces, fix16_t anchorsScale);
int post_process_retinaface(object_t faces[],int max_faces, fix16_t *network_outputs[9],int image_width,int image_height,
                            fix16_t confidence_threshold, fix16_t nms_threshold);
int post_process_scrfd(object_t faces[], int max_faces, fix16_t *network_outputs[9], int image_width, int image_height, 
                            fix16_t confidence_threshold, fix16_t nms_threshold);
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
int decodeMultiplePoses_int8(poses_t poses[],int8_t *scores, int8_t *offsets,int8_t *displacementsFwd, int8_t *displacementsBwd, int outputStride,int maxPoseDetections, fix16_t scoreThreshold,int nmsRadius, fix16_t minPoseScore,int height, int width, int zero_points[], fix16_t scale_outs[]);
int pprint_post_process(const char *name, const char *str, model_t *model, vbx_cnn_io_ptr_t *io_buffers, int int8_flag);

#ifdef __cplusplus
}
#endif

#endif // __POSTPROCESS_H_
