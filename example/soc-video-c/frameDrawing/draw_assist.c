#include "draw_assist.h"

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
                           uint32_t *dst, int dst_stride,
                           int dst_width, int dst_height,
                           int start_column, int start_row,
                           int end_column, int end_row,
                           uint32_t colour){
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

  int rows = (end_row - start_row) + 1;
  if(start_row > end_row){
    rows       = (start_row - end_row) + 1;
    inverse_stride = inverse_stride - (dst_stride << DRAW_ASSIST_STRIDE_FRACTIONAL_BITS);
  } else {
    inverse_stride = inverse_stride + (dst_stride << DRAW_ASSIST_STRIDE_FRACTIONAL_BITS);
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
