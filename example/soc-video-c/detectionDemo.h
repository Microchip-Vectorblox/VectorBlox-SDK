#ifndef __DETECTION_DEMO_H
#define __DETECTION_DEMO_H

#include "vbx_cnn_api.h"
#include "postprocess.h"
#include "model_descr.h"

#ifdef __cplusplus
extern "C" {
#endif

short detectionDemoInit(vbx_cnn_t* the_vbx_cnn, struct model_descr_t* models, uint8_t modelIdx);
int runDetectionDemo(struct model_descr_t* models, vbx_cnn_t* the_vbx_cnn, uint8_t modelIdx);

#ifdef __cplusplus
}
#endif

#endif //__DETECTION_DEMO_H
