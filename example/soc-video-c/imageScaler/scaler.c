

/**
 * bilinear interpolation with interleaved input and planer output
 */
#include "scaler.h"

void resize_strided_image(uint32_t* image_in,int in_w,int in_h,int in_stride,
                 uint8_t* image_out,int out_w,int out_h)
{
    in_stride/=sizeof(image_in[0]);
  for(int h=0;h<out_h;h++){
    for(int w=0;w<out_w;w++){
      int x1=w*in_w/out_w;
      int x2=x1+1;
      int x_alpha = w*in_w % out_w;
      int y1=h*in_h/out_h;
      int y2=y1+1;
      int y_alpha=w*in_h % out_h;
      uint32_t p11 = image_in[in_stride*y1 + x1];
      uint32_t p12 = image_in[in_stride*y1 + x2];
      uint32_t p21 = image_in[in_stride*y2 + x1];
      uint32_t p22 = image_in[in_stride*y2 + x2];
      for (int c =0;c <3;c++){
#define get_chan(pix,chan) ((pix >> (8*(chan))) &0xFF)
        int a = get_chan(p11,c) *(out_w-x_alpha) /out_w + get_chan(p12,c) *x_alpha/out_w ;
        int b = get_chan(p21,c) *(out_w-x_alpha) /out_w + get_chan(p22,c) *x_alpha/out_w ;
        int p = a *(out_h-y_alpha) /out_h + b *y_alpha/out_h ;
        image_out[c*out_w*out_h+h*out_w+w ] = p;
      }
    }
  }

}

#define SCALE_IN_ADDR 0
#define SCALE_OUT_ADDR 1
#define SCALE_XRATIO 2
#define SCALE_YRATIO 3
#define SCALE_IN_STRIDE 4
#define SCALE_IN_WIDTH 5
#define SCALE_IN_HEIGHT 6
#define SCALE_OUT_WIDTH 7
#define SCALE_OUT_HEIGHT 8
#define SCALE_CONTROL 10
int32_t calc_ratio(int32_t in,int32_t out){
  int64_t in64 = in-1;
  in64<<=16;
  return in64/out;
}
#include "scaler.h"
void resize_image_hls_start(volatile uint32_t* hls_scale_base_addr,
                            uint32_t* image_in,int in_w,int in_h,int in_stride,int x_offset,int y_offset,
                            uint8_t* image_out,int out_w,int out_h)
{
  //setup parameters
  int32_t x_ratio =calc_ratio(in_w, out_w);
  int32_t y_ratio =calc_ratio(in_h, out_h);

  hls_scale_base_addr[SCALE_XRATIO] = x_ratio;
  hls_scale_base_addr[SCALE_YRATIO] = y_ratio;
  hls_scale_base_addr[SCALE_IN_ADDR]  = (uint64_t)(uintptr_t)(image_in +y_offset*in_stride/4 + x_offset);
  hls_scale_base_addr[SCALE_OUT_ADDR] = (uint64_t)(uintptr_t)image_out;
  hls_scale_base_addr[SCALE_IN_WIDTH] =in_w;
  hls_scale_base_addr[SCALE_IN_HEIGHT] =in_h;
  hls_scale_base_addr[SCALE_IN_STRIDE] = in_stride;
  hls_scale_base_addr[SCALE_OUT_WIDTH] = out_w;
  hls_scale_base_addr[SCALE_OUT_HEIGHT] = out_h;

  //start the kernel
  hls_scale_base_addr[SCALE_CONTROL] = 1;
}
void resize_image_hls_wait(volatile uint32_t* hls_scale_base_addr)
{
  //wait for the kernel
  while(!hls_scale_base_addr[SCALE_CONTROL]);
}


void resize_image_hls(volatile uint32_t* hls_scale_base_addr,
                      uint32_t* image_in,int in_w,int in_h,int in_stride,int x_offset,int y_offset,
                      uint8_t* image_out,int out_w,int out_h)
{
    resize_image_hls_start(hls_scale_base_addr,
                           image_in,in_w,in_h,in_stride,x_offset,y_offset,
                           image_out,out_w,out_h);
    resize_image_hls_wait(hls_scale_base_addr);
}
