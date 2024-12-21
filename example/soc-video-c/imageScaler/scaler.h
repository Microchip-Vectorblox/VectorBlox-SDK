#ifndef __SCALER_H
#define __SCALER_H

#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

void resize_strided_image(uint32_t* image_in,int in_w,int in_h,int in_stride,
                  uint8_t* image_out,int out_w,int out_h);
void resize_image_hls_start(volatile uint32_t* hls_scale_base_addr,
                            uint32_t* image_in,int in_w,int in_h,int in_stride,int x_offset,int y_offset,
                            uint8_t* image_out,int out_w,int out_h);
void resize_image_hls_wait(volatile uint32_t* hls_scale_base_addr);
void resize_image_hls(volatile uint32_t* hls_scale_base_addr,
                      uint32_t* image_in,int in_w,int in_h,int in_stride,int x_offset,int y_offset,
                      uint8_t* image_out,int out_w,int out_h);
#define HLS_SCALER_BASE_ADDRESS ((volatile uint32_t*)0x78030000)

#define SCALER_FRAME_ADDRESS	 	0x68000000 

#ifdef __cplusplus
}
#endif

#endif //#ifndef __SCALER_H
