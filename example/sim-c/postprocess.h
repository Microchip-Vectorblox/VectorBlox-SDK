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

typedef struct {
    int version;
    int input_dims[3];
    int output_dims[3];
    int coords;
    int classes;
    int num;
    int anchors_length;
    float* anchors;
    int mask_length;
    int* mask;
} yolo_info_t;

typedef void (*file_write)(const char*,int);
extern char *imagenet_classes[];
void post_process_classifier(fix16_t *outputs, const int output_size, int16_t* output_index, int topk);
int post_process_yolo(fix16_t **outputs, const int num_outputs, yolo_info_t *cfg,
                      float thresh, float overlap, fix16_box fix16_boxes[], int max_boxes);

extern char *imagenet_classes[1000];
extern char *voc_classes[20];
extern char *coco_classes[80];

#ifdef __cplusplus
}
#endif

#endif // __POSTPROCESS_H_
