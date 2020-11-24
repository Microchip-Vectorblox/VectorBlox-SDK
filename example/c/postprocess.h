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

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int xmin;
    int xmax;
    int ymin;
    int ymax;
    fix16_t confidence;
    int class_id;
	const char* class_name;
} fix16_box;

typedef void (*file_write)(const char*,int);
extern char *imagenet_classes[];
void post_process_classifier(fix16_t *outputs, const int output_size, int16_t* output_index, int topk);
void post_process_tiny_yolov2_voc(fix16_t *outputs, int *detections, fix16_box fix16_boxes[],int max_boxes);
void post_process_yolov2_voc(fix16_t *outputs, int *detections, fix16_box fix16_boxes[],int max_boxes);
void post_process_tiny_yolov3_coco(fix16_t *outputs0,
                                   fix16_t *outputs1,
                                   int *detections,
                                   fix16_box fix16_boxes[],int max_boxes);
int post_process_yolo(fix16_t **outputs, int *output_sizes, const int num_outputs,
					  float **biases, int* dims,
                      float thresh, float overlap, const int num, const int classes,
                      fix16_box fix16_boxes[], int max_boxes,const int version);


#ifdef __cplusplus
}
#endif

#endif // __POSTPROCESS_H_
