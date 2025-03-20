#ifndef __DRAW_ASSIST_H
#define __DRAW_ASSIST_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif //#ifdef __cplusplus

  int draw_assist_not_done(volatile void *draw_assist_base_address);
  
  void draw_assist_dma_memcpy(volatile void *draw_assist_base_address,
                              int columns, int rows,
                              uint32_t *dst, int dst_stride,
                              uint32_t *src, int src_stride);

  void draw_assist_dma_memset(volatile void *draw_assist_base_address,
                              int columns, int rows,
                              uint32_t *dst, int dst_stride,
                              uint32_t colour);

  //Draw a line of apparent thickness one one pixel.
  //
  //Note that the end point is inclusive.
  //
  //The line is centered on the start/end pixels.
  void draw_assist_draw_line(volatile void *draw_assist_base_address,
                             int start_column, int start_row,
                             int end_column, int end_row,
                             uint32_t *dst, int dst_stride,
                             uint32_t colour, int alpha);

  void draw_assist_draw_line_pp(//volatile void *draw_assist_base_address,
                              int start_column, int start_row,
                              int end_column, int end_row,
                              uint32_t *dst, int dst_stride,
                              uint32_t colour, int thickness, int alpha);

  void draw_assist_plot_pixel(int x, int y, float brightness, uint32_t *dst, int dst_stride,
                              uint32_t colour );
#ifdef __cplusplus
}
#endif //#ifdef __cplusplus

#define DRAW_ASSIST_CONTROL_REGISTER               (0x00>>2)
#define DRAW_ASSIST_INPUT_ADDRESS_REGISTER         (0x08>>2)
#define DRAW_ASSIST_INPUT_COLUMNS_MINUS1_REGISTER  (0x10>>2)
#define DRAW_ASSIST_INPUT_ROWS_REGISTER            (0x14>>2)
#define DRAW_ASSIST_INPUT_STRIDE_REGISTER          (0x18>>2)
#define DRAW_ASSIST_OUTPUT_ADDRESS_REGISTER        (0x20>>2)
#define DRAW_ASSIST_OUTPUT_COLUMNS_MINUS1_REGISTER (0x28>>2)
#define DRAW_ASSIST_OUTPUT_ROWS_REGISTER           (0x2C>>2)
#define DRAW_ASSIST_OUTPUT_STRIDE_REGISTER         (0x30>>2)

#define DRAW_ASSIST_CONTROL_START_BIT              0x01
#define DRAW_ASSIST_CONTROL_FULL_BIT               0x02
#define DRAW_ASSIST_CONTROL_RUNNING_BIT            0x04

#define DRAW_ASSIST_STRIDE_FRACTIONAL_BITS         16

#endif //#ifndef __DRAW_ASSIST_H