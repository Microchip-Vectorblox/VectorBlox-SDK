#ifndef DRAW_H
#define DRAW_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
void draw_dma_memcpy(int cols,int rows,
                     uint32_t *dst,int dst_stride,
                     uint32_t *src,int src_stride);
void draw_wait_for_draw();
#define GET_COLOUR(r,g,b,a)  ((uint32_t)(	  \
	(((r)&0xFF)<<16) | \
	(((g)&0xFF)<<8) | \
	(((b)&0xFF)<<0) | \
	(((a)&0xFF)<<24)))

void draw_clear_frame( uint32_t* f_buf,int f_width,int f_height);

void draw_rectangle(int x,int y,int w,int h,uint32_t colour,
                    uint32_t* f_buf,int f_width,int f_height);


void draw_box(int x,int y,int w,int h,int thickness,uint32_t colour,
              uint32_t* f_buf,int f_width,int f_height);

// x,y == bottom left_corner
    typedef enum{
        BLUE,
        WHITE,
        ORANGE,
        RED,
        GREEN} label_colour_e;
void draw_label(const char* label,int x,int y,
                uint32_t* f_buf,int f_width,int f_height,label_colour_e colour);

uint32_t get_colour_modulo(int i);
#ifdef __cplusplus
}
#endif

#endif //DRAW_H
