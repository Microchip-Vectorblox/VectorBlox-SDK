#ifndef __MODEL_DESCR_H
#define __MODEL_DESCR_H

#include "tracking.h"

#ifdef __cplusplus
extern "C" {
#endif

struct model_descr_t{
    const char *name;
    const char *fname;
    int spi_offset;
    const char* post_process_type;
    model_t* model;
    short modelSetup_done;
    int time_ms;
    vbx_cnn_io_ptr_t* model_io_buffers;
    uint8_t* model_input_buffer;
    fix16_t* model_output_buffer[64];
    size_t model_output_length[64];
    // the following pipelined I/O buffers used for objectDetection
    uint8_t* pipelined_input_buffer[2];
    fix16_t* pipelined_output_buffers[2][64];
    int buf_idx;
    int is_running;
    track_t** pTracks;
    Tracker_t* pTracker;
    track_t* tracks;
    fix16_t *coord4;
    fix16_t x_ratio;
    fix16_t y_ratio;
};

#ifdef __cplusplus
}
#endif

#endif //__MODEL_DESCR_H
