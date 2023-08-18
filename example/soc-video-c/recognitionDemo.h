#ifndef __RECOGNITION_DEMO_H
#define __RECOGNITION_DEMO_H

#include "libfixmath/fixmath.h"
#include "vbx_cnn_api.h"
#include "postprocess.h"
#include "model_descr.h"
#include "tracking.h"

#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned char uchar;
void append_name(char* name_entered);
bool not_duplicate(char* id);
void print_list();
void delete_embedding(char* input_buf,struct model_descr_t* models,uint8_t modelIdx);
void tracksInit(struct model_descr_t* models);
void trackClean(struct model_descr_t* models, uint8_t modelIdx);

short recognitionDemoInit(vbx_cnn_t* the_vbx_cnn, struct model_descr_t* models, uint8_t modelIdx, int has_attribute_model, int screen_height, int screen_width, int screen_y_offset, int screen_x_offset);
void matchEmbedding(fix16_t embedding[],fix16_t* similarity, char** name);
void recognizeObject(vbx_cnn_t* the_vbx_cnn, struct model_descr_t* model, object_t* object, const char* post_process_type, int screen_height, int screen_width, int screen_stride, int screen_y_offset, int screen_x_offset);
int runRecognitionDemo(struct model_descr_t* models, vbx_cnn_t* the_vbx_cnn, uint8_t modelIdx, int use_attribute_model, int screen_height, int screen_width, int screen_y_offset, int screen_x_offset);
#ifdef __cplusplus
}
#endif

#endif //__RECOGNITION_DEMO_H
