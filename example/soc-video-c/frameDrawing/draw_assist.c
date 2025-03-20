#include "draw_assist.h"
#include <math.h>
#include <stdbool.h>
#include "draw.h"

int draw_assist_not_done(volatile void *draw_assist_base_address){
  volatile uint32_t *draw_assist_register = (volatile uint32_t *)draw_assist_base_address;
  
  return (draw_assist_register[DRAW_ASSIST_CONTROL_REGISTER] & DRAW_ASSIST_CONTROL_RUNNING_BIT) ? 1 : 0;
}

void draw_assist_dma_memcpy(volatile void *draw_assist_base_address,
                            int columns, int rows,
                            uint32_t *dst, int dst_stride,
                            uint32_t *src, int src_stride){
  volatile uint32_t *draw_assist_register = (volatile uint32_t *)draw_assist_base_address;

  draw_assist_register[DRAW_ASSIST_INPUT_ADDRESS_REGISTER]           = (uint32_t)((uintptr_t)src);
  if(src_stride == columns){
    draw_assist_register[DRAW_ASSIST_INPUT_COLUMNS_MINUS1_REGISTER]  = (rows*columns)-1;
    draw_assist_register[DRAW_ASSIST_INPUT_ROWS_REGISTER]            = 1;
  } else {
    draw_assist_register[DRAW_ASSIST_INPUT_COLUMNS_MINUS1_REGISTER]  = columns-1;
    draw_assist_register[DRAW_ASSIST_INPUT_ROWS_REGISTER]            = rows;
    draw_assist_register[DRAW_ASSIST_INPUT_STRIDE_REGISTER]          = src_stride << DRAW_ASSIST_STRIDE_FRACTIONAL_BITS;
  }
  draw_assist_register[DRAW_ASSIST_OUTPUT_ADDRESS_REGISTER]          = (uint32_t)((uintptr_t)dst);
  if(dst_stride == columns){
    draw_assist_register[DRAW_ASSIST_OUTPUT_COLUMNS_MINUS1_REGISTER] = (rows*columns)-1;
    draw_assist_register[DRAW_ASSIST_OUTPUT_ROWS_REGISTER]           = 1;
  } else {
    draw_assist_register[DRAW_ASSIST_OUTPUT_COLUMNS_MINUS1_REGISTER] = columns-1;
    draw_assist_register[DRAW_ASSIST_OUTPUT_ROWS_REGISTER]           = rows;
    draw_assist_register[DRAW_ASSIST_OUTPUT_STRIDE_REGISTER]         = dst_stride << DRAW_ASSIST_STRIDE_FRACTIONAL_BITS;
  }

  //Wait for ready then set start
  while(draw_assist_register[DRAW_ASSIST_CONTROL_REGISTER] & DRAW_ASSIST_CONTROL_FULL_BIT){
  }
  draw_assist_register[DRAW_ASSIST_CONTROL_REGISTER] = DRAW_ASSIST_CONTROL_START_BIT;
}

void draw_assist_dma_memset(volatile void *draw_assist_base_address,
                            int columns, int rows,
                            uint32_t *dst, int dst_stride,
                            uint32_t colour){
  volatile uint32_t *draw_assist_register = (volatile uint32_t *)draw_assist_base_address;

  //Splatted value puts colour in address register and sets rows to 0
  draw_assist_register[DRAW_ASSIST_INPUT_ADDRESS_REGISTER]           = colour;
  draw_assist_register[DRAW_ASSIST_INPUT_COLUMNS_MINUS1_REGISTER]    = (rows*columns)-1;
  draw_assist_register[DRAW_ASSIST_INPUT_ROWS_REGISTER]              = 0;
  draw_assist_register[DRAW_ASSIST_OUTPUT_ADDRESS_REGISTER]          = (uint32_t)((uintptr_t)dst);
  if(dst_stride == columns){
    draw_assist_register[DRAW_ASSIST_OUTPUT_COLUMNS_MINUS1_REGISTER] = (rows*columns)-1;
    draw_assist_register[DRAW_ASSIST_OUTPUT_ROWS_REGISTER]           = 1;
  } else {
    draw_assist_register[DRAW_ASSIST_OUTPUT_COLUMNS_MINUS1_REGISTER] = columns-1;
    draw_assist_register[DRAW_ASSIST_OUTPUT_ROWS_REGISTER]           = rows;
    draw_assist_register[DRAW_ASSIST_OUTPUT_STRIDE_REGISTER]         = dst_stride << DRAW_ASSIST_STRIDE_FRACTIONAL_BITS;
  }

  //Wait for ready then set start
  while(draw_assist_register[DRAW_ASSIST_CONTROL_REGISTER] & DRAW_ASSIST_CONTROL_FULL_BIT){
  }
  draw_assist_register[DRAW_ASSIST_CONTROL_REGISTER] = DRAW_ASSIST_CONTROL_START_BIT;
}

//Draw a line of apparent thickness one one pixel.
//
//Note that the end point is inclusive.
//
//The line is centered on the start/end pixels.
void draw_assist_draw_line(volatile void *draw_assist_base_address,
                           int start_column, int start_row,
                           int end_column, int end_row,
                           uint32_t *dst, int dst_stride,
                           uint32_t colour, int alpha){
  //Always go forwards in x
  if(start_column > end_column){
    int temp     = start_column;
    start_column = end_column;
    end_column   = temp;
    temp         = start_row;
    start_row    = end_row;
    end_row      = temp;
  }
  
  uint32_t *dst_start = dst+(start_row*dst_stride)+start_column;
  
  //Special case horizontal line
  if(end_row == start_row){
    draw_assist_dma_memset(draw_assist_base_address,
                           (end_column-start_column)+1, 1,
                           dst_start, dst_stride,
                           colour);
    return;
  }

  int32_t inverse_stride = ((((end_column - start_column) << DRAW_ASSIST_STRIDE_FRACTIONAL_BITS) +
                             (1 << (DRAW_ASSIST_STRIDE_FRACTIONAL_BITS - 1))) /
                            (end_row - start_row));
  uint32_t abs_inverse_stride = inverse_stride;
  if(inverse_stride < 0){
    abs_inverse_stride = 0 - inverse_stride;
  }

  int columns = ((abs_inverse_stride + ((1 << DRAW_ASSIST_STRIDE_FRACTIONAL_BITS) - 1)) >>
                 DRAW_ASSIST_STRIDE_FRACTIONAL_BITS);

  //Correct for line thickness
  end_column = end_column - (columns - 1);

  int rows = (end_row - start_row);
  if(start_row > end_row){
    rows       = (start_row - end_row);
    inverse_stride = abs_inverse_stride - (dst_stride << DRAW_ASSIST_STRIDE_FRACTIONAL_BITS);
  } else {
    inverse_stride = abs_inverse_stride + (dst_stride << DRAW_ASSIST_STRIDE_FRACTIONAL_BITS);
  }

  volatile uint32_t *draw_assist_register = (volatile uint32_t *)draw_assist_base_address;

  //Splatted value puts colour in address register and sets rows to 0
  draw_assist_register[DRAW_ASSIST_INPUT_ADDRESS_REGISTER]           = colour;
  draw_assist_register[DRAW_ASSIST_INPUT_COLUMNS_MINUS1_REGISTER]    = (rows*columns)-1;
  draw_assist_register[DRAW_ASSIST_INPUT_ROWS_REGISTER]              = 0;
  draw_assist_register[DRAW_ASSIST_OUTPUT_ADDRESS_REGISTER]          = (uint32_t)((uintptr_t)dst_start);
  if(inverse_stride == (columns << DRAW_ASSIST_STRIDE_FRACTIONAL_BITS)){
    draw_assist_register[DRAW_ASSIST_OUTPUT_COLUMNS_MINUS1_REGISTER] = (rows*columns)-1;
    draw_assist_register[DRAW_ASSIST_OUTPUT_ROWS_REGISTER]           = 1;
  } else {
    draw_assist_register[DRAW_ASSIST_OUTPUT_COLUMNS_MINUS1_REGISTER] = columns-1;
    draw_assist_register[DRAW_ASSIST_OUTPUT_ROWS_REGISTER]           = rows;
    draw_assist_register[DRAW_ASSIST_OUTPUT_STRIDE_REGISTER]         = inverse_stride;
  }

  //Wait for ready then set start
  while(draw_assist_register[DRAW_ASSIST_CONTROL_REGISTER] & DRAW_ASSIST_CONTROL_FULL_BIT){
  }
  draw_assist_register[DRAW_ASSIST_CONTROL_REGISTER] = DRAW_ASSIST_CONTROL_START_BIT;
}


void draw_assist_draw_line_pp(//volatile void *draw_assist_base_address,
                              int start_column, int start_row,
                              int end_column, int end_row,
                              uint32_t *dst, int dst_stride,
                              uint32_t colour, int thickness, 
                              int alpha){



  int16_t dy = end_row - start_row;
  int16_t dx = end_column - start_column;
  float gradient;
  uint16_t abs_dy = dy;
  uint16_t abs_dx = dx;
  if(dy < 0){
    abs_dy = 0-dy;
  }
  if(dx < 0){
    abs_dx = 0-dx;
  }

  bool steep  = abs_dy > abs_dx;
  if(steep){
    int temp     = start_column;
    start_column = start_row;
    start_row    = temp;
    temp         = end_column;
    end_column   = end_row;
    end_row      = temp;
  }

    //Always go forwards in x
  if(start_column > end_column){
    int temp     = start_column;
    start_column = end_column;
    end_column   = temp;
    temp         = start_row;
    start_row    = end_row;
    end_row      = temp;
  }

  dy = end_row - start_row;
  dx = end_column - start_column;
  if (dx == 0){
    gradient = 0.0;
  } else {
    gradient = (float)dy/dx;
  }
  
  //start pixel
  if(steep){
    for (int w = 0; w <= thickness; ++w){
      draw_assist_plot_pixel(start_row+w, start_column, alpha/100.0, dst, dst_stride, colour);
      draw_assist_plot_pixel(start_row-w, start_column, alpha/100.0, dst, dst_stride, colour);
    }
  } else {
    for (int w = 0; w <= thickness; ++w){
      draw_assist_plot_pixel(start_column, start_row+w, alpha/100.0, dst, dst_stride, colour);
      draw_assist_plot_pixel(start_column, start_row-w, alpha/100.0, dst, dst_stride, colour);
    }
  }
  float intery = start_row + gradient;
  //end pixel
  if(steep){
    for (int w = 0; w <= thickness; ++w){
      draw_assist_plot_pixel(end_row+w, end_column, alpha/100.0, dst, dst_stride, colour);
      draw_assist_plot_pixel(end_row-w, end_column, alpha/100.0, dst, dst_stride, colour);
    }
  } else {
    for (int w = 0; w <= thickness; ++w){
      draw_assist_plot_pixel(end_column, end_row+w, alpha/100.0, dst, dst_stride, colour);
      draw_assist_plot_pixel(end_column, end_row-w, alpha/100.0, dst, dst_stride, colour);
    }
  }
  //main loop
  if(steep){
    for(int i = start_column; i <= end_column; i++){
      for (int w = 0; w <= thickness; ++w){
        draw_assist_plot_pixel(floor(intery)+w, i,  alpha/100.0, dst, dst_stride, colour);
        draw_assist_plot_pixel(floor(intery)-w, i,  alpha/100.0, dst, dst_stride, colour);
      }
    draw_assist_plot_pixel(floor(intery)-thickness-1, i, alpha/100.0 * (1-(intery - floor(intery))), dst, dst_stride, colour);
    draw_assist_plot_pixel(floor(intery)+thickness+1, i, alpha/100.0 *(intery - floor(intery)),  dst, dst_stride, colour);
    intery += gradient;
    }
  } else {
    for(int i = start_column; i < end_column; i++){
      for (int w = 0; w <= thickness; ++w){
        draw_assist_plot_pixel(i, floor(intery)+w,  alpha/100.0, dst, dst_stride, colour);
        draw_assist_plot_pixel(i, floor(intery)-w,  alpha/100.0, dst, dst_stride, colour);
      }
    draw_assist_plot_pixel(i, floor(intery)-thickness-1,  alpha/100.0 * (1-(intery - floor(intery))), dst, dst_stride, colour);
    draw_assist_plot_pixel(i, floor(intery)+thickness+1,  alpha/100.0 * (intery - floor(intery)),  dst, dst_stride, colour);
    intery += gradient;

    }
  }
}



void draw_assist_plot_pixel(int x, int y, float brightness, uint32_t *dst, int dst_stride, uint32_t colour ){
  uint32_t *dst_start  = dst+(y*dst_stride)+x;  
  uint32_t temp_colour = dst_start[0];
  uint8_t r_1,g_1,b_1;
  uint8_t r_2,g_2,b_2;

  r_1 = (temp_colour & 0x00FF0000) >> 16;
  g_1 = (temp_colour & 0x0000FF00) >> 8;
  b_1 = (temp_colour & 0x000000FF) >>  0;

  r_2 = (colour & 0x00FF0000) >> 16;
  g_2 = (colour & 0x0000FF00) >> 8;
  b_2 = (colour & 0x000000FF) >>  0;

  r_1 = r_1 * (1-brightness) + r_2 * brightness;
  g_1 = g_1 * (1-brightness) + g_2 * brightness;
  b_1 = b_1 * (1-brightness) + b_2 * brightness;

  colour = GET_COLOUR(r_1, g_1, b_1, 255);

  dst_start[0] = colour;
}