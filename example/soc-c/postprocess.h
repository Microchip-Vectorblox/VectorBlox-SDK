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

typedef void (*file_write)(const char*,int);
extern char *imagenet_classes[];
void post_process_classifier(fix16_t *outputs, const int output_size, int16_t* output_index, int topk);
int post_process_yolo(fix16_t **outputs, const int num_outputs, yolo_info_t *cfg,
                      fix16_t thresh, fix16_t overlap, fix16_box fix16_boxes[], int max_boxes);

extern char *imagenet_classes[1000];
extern char *voc_classes[20];
extern char *coco_classes[80];
extern char* coco91_classes[92];
extern char* vehicle_classes[3];

typedef unsigned char uchar;

typedef struct
{
    uchar* data;
    int rows;
    int cols;
    int channels;
}cMat;

typedef struct {
    fix16_t box[4]; // (left, top, right, bottom)
    fix16_t points[6][2]; // (left eye, right eye, nose, mouth)(x,y)
    fix16_t detectScore;
} face_t;

#define PLATE_LENGTH 12
typedef struct {
    fix16_t lpbox[4];
    fix16_t keypoints[8];
    fix16_t detect_Score;
    int stride;
    fix16_t recScore;
    char name[PLATE_LENGTH];
    short track_val;
} plateT;


fix16_t calcIou_LTRB(fix16_t* A, fix16_t* B);
fix16_t calcIou_XYWH(fix16_t* A, fix16_t* B);

int post_process_blazeface(face_t faces[],fix16_t* scores,fix16_t* points,int scoresLength,int max_faces, fix16_t anchorsScale);
int post_process_retinaface(face_t faces[],int max_faces, fix16_t *network_outputs[9],int image_width,int image_height,
                            fix16_t confidence_threshold, fix16_t nms_threshold);
int post_process_scrfd(face_t faces[], int max_faces, fix16_t *network_outputs[9], int image_width, int image_height, 
                            fix16_t confidence_threshold, fix16_t nms_threshold);
int post_process_lpd(plateT plates[],int max_plates, fix16_t *detectOutputs[9], int image_width, int image_height,
                            fix16_t confidence_threshold, fix16_t nms_threshold,int detectNumOutputs);

int post_process_ssdv2(fix16_box *boxes, int max_boxes,
                       fix16_t *network_outputs[12], int num_classes,
                       fix16_t confidence_threshold, fix16_t nms_threshold);
int post_process_vehicles(fix16_box *boxes, int max_boxes,
                       fix16_t *network_outputs[12], int num_classes,
                       fix16_t confidence_threshold, fix16_t nms_threshold);

int pprint_post_process(const char *name, const char *str, model_t *model, vbx_cnn_io_ptr_t *io_buffers);

#ifdef __cplusplus
}
#endif

#endif // __POSTPROCESS_H_
