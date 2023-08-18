#ifndef __WARP_H
#define __WARP_H

#include "fix16.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  uint32_t in_addr;
  uint32_t out_addr;
  uint32_t out_width;
  uint32_t out_height;
  uint32_t m[9];
  uint32_t _;
  uint32_t start;
  uint32_t done;
} warp_registers_t;


void warp_affine_image(volatile uint32_t* hsl_warp_base_addr,
                       uint8_t* image_in, uint8_t* image_out,int out_w,int out_h,
                       const uint32_t* transform_matrix_2x3);


/* src and dst are fix16_t arrays of 3 x,y points [x0, y0, x1, y1, x2, y2] */
/* M is 2x3 mf16 matrix */
/* taken from https://github.com/opencv/opencv/blob/68d15fc62edad980f1ffa15ee478438335f39cc3/modules/imgproc/src/imgwarp.cpp#
L3325 */

void warp_image_with_points(volatile uint32_t* scale_base_addr,
                            volatile uint32_t* warp_base_addr,
                            uint32_t* image_in, uint8_t* image_out,uint8_t* temp_buffer,
                            fix16_t src_points[6],
                            fix16_t dest_points[6],
                            unsigned src_width,unsigned src_height,unsigned src_stride,
                            unsigned dst_width,unsigned dst_height);

#define HLS_WARP_BASE_ADDRESS ((volatile uint32_t*)0x78040000)
#ifdef __cplusplus
}
#endif

#endif //#ifndef __WARP_H
