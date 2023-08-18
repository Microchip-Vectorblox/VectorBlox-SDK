#include "ascii_characters.h"
#include "draw.h"
#if  defined(__amd64)
void draw_dma_memcpy(int cols,int rows,
                     uint32_t *dst,int dst_stride,
                     uint32_t *src,int src_stride)
{
  for(int r=0;r<rows;++r){
	for(int c=0;c<cols;++c){
	  dst[r*dst_stride+c] = src[r*src_stride+c];
	}
  }

}

void draw_dma_memset(int cols,int rows,
                     uint32_t* dst,int dst_stride,
                     uint32_t colour)
{
  for(int r=0;r<rows;++r){
	for(int c=0;c<cols;++c){
	  dst[r*dst_stride+c] = colour;
	}
  }
}
void draw_wait_for_draw(){}
#else
extern volatile void *draw_assist_base_address;
#if VBX_SOC_DRIVER
extern void *ascii_characters_base_address;
#else
extern const int CHAR_SPI_OFFSET;
#endif
#include "draw_assist.h"
void draw_dma_memcpy(int cols,int rows,
                     uint32_t *dst,int dst_stride,
                     uint32_t *src,int src_stride)
{
  draw_assist_dma_memcpy(draw_assist_base_address,
						 cols,rows,
						 dst,dst_stride,
						 src,src_stride);
  //draw_wait_for_draw();

}

void draw_dma_memset(int cols,int rows,
                     uint32_t* dst,int dst_stride,
                     uint32_t colour)
{

  draw_assist_dma_memset(draw_assist_base_address,
						 cols,rows,
						 dst,dst_stride,
						 colour);
  //draw_wait_for_draw();

}
void draw_wait_for_draw(){
	while(draw_assist_not_done(draw_assist_base_address));
}
#endif
void draw_clear_frame( uint32_t* f_buf,int f_width,int f_height)
{
  draw_dma_memset(f_width*f_height,1,f_buf,0,0);
}

void draw_rectangle(int x,int y,int w,int h,uint32_t colour,
                    uint32_t* f_buf,int f_width,int f_height)
{
  draw_dma_memset(w,h,
				  f_buf+y*f_width+x,f_width,
				  colour);

}


void draw_box(int x,int y,int w,int h,int thickness,uint32_t colour,
			  uint32_t* f_buf,int f_width,int f_height)
{
  //top
  draw_rectangle(x,y,w,thickness,colour,
				 f_buf,f_width,f_height);
  //bottom
  draw_rectangle(x,y+h-thickness,w,thickness,colour,
				 f_buf,f_width,f_height);
  //left
  draw_rectangle(x,y,thickness,h,colour,
				 f_buf,f_width,f_height);
  //right
  draw_rectangle(x+w-thickness,y,thickness,h,colour,
				 f_buf,f_width,f_height);

}
void initialize_characters();

// x,y == bottom left_corner
void draw_label(const char* label,int x,int y,
                uint32_t* f_buf,int f_width,int f_height,label_colour_e colour)
{
  initialize_characters();
  struct character_t* ascii_characters;
  switch(colour){
  case BLUE:
      ascii_characters=ascii_characters_blue;break;
  case WHITE:
      ascii_characters=ascii_characters_white;break;
  case ORANGE:
      ascii_characters=ascii_characters_orange;break;
  case RED:
      ascii_characters=ascii_characters_red;break;
  case GREEN:
      ascii_characters=ascii_characters_green;break;
  }
  for(int i=0;label[i];++i){
	int c  = label[i]&0x7F;
	struct character_t letter =  ascii_characters[c];
	uint32_t* dst = f_buf +y*f_width + x;
	draw_dma_memcpy(letter.width,letter.height,
					dst,f_width,
					(uint32_t*)letter.data,letter.width);
	x+=letter.width;
  }
}
uint32_t get_colour_modulo(int i){

  static uint32_t voc_colors[] = { GET_COLOUR(128,   0,   0,255),
								   GET_COLOUR(  0, 128,   0,255),
								   GET_COLOUR(128, 128,   0,255),
								   GET_COLOUR(  0,   0, 128,255),
								   GET_COLOUR(128,   0, 128,255),
								   GET_COLOUR(  0, 128, 128,255),
								   GET_COLOUR(128, 128, 128,255),
								   GET_COLOUR( 64,   0,   0,255),
								   GET_COLOUR(192,   0,   0,255),
								   GET_COLOUR( 64, 128,   0,255),
								   GET_COLOUR(192, 128,   0,255),
								   GET_COLOUR( 64,   0, 128,255),
								   GET_COLOUR(192,   0, 128,255),
								   GET_COLOUR( 64, 128, 128,255),
								   GET_COLOUR(192, 128, 128,255),
								   GET_COLOUR(  0,  64,   0,255),
								   GET_COLOUR(128,  64,   0,255),
								   GET_COLOUR(  0, 192,   0,255),
								   GET_COLOUR(128, 192,   0,255),
								   GET_COLOUR(  0,  64, 128,255)};
  uint32_t c = voc_colors[i%sizeof(voc_colors)/sizeof(*voc_colors)];
  return c;
}

static void change_chardata_offset(struct character_t* ascii_characters,char* char_data){
    for(int c =0;c<128;c++){
        intptr_t offset = (intptr_t)ascii_characters[c].data;
        ascii_characters[c].data = char_data + offset;
    }
}
void initialize_characters(){
  static int is_initialized=0;
  static char* char_data;
  if(is_initialized==1){
	return;
  }
  is_initialized=1;
#if VBX_SOC_DRIVER
  char_data = (char*)ascii_characters_base_address;
#else
  char_data = ddr_uncached_allocate(character_bin_length);
  copy_from_flash(NULL, char_data, CHAR_SPI_OFFSET, character_bin_length);
#endif
  change_chardata_offset(ascii_characters_blue ,char_data);
  change_chardata_offset(ascii_characters_white,char_data);
  change_chardata_offset(ascii_characters_orange ,char_data);
  change_chardata_offset(ascii_characters_red  ,char_data);
  change_chardata_offset(ascii_characters_green,char_data);
}
