#include "imageScaler/scaler.h"
#include "detectionDemo.h"
#include "frameDrawing/draw_assist.h"
#include "frameDrawing/draw.h"

//#define DRAW_SCALER

extern volatile uint32_t*  PROCESSING_FRAME_ADDRESS;
extern volatile uint32_t*  PROCESSING_NEXT_FRAME_ADDRESS;
extern volatile uint32_t*  SCALER_BASE_ADDRESS;
extern volatile uint32_t*  SAVED_FRAME_SWAP;
extern int fps;

extern int set_screen_width;
extern int set_screen_height;
extern int set_screen_y_offset;
extern int set_screen_x_offset;
extern int set_screen_stride;


#ifndef HLS_RESIZE
extern volatile uint32_t* RED_DDR_FRAME_START_ADDR;
extern volatile uint32_t* GREEN_DDR_FRAME_START_ADDR;
extern volatile uint32_t* BLUE_DDR_FRAME_START_ADDR;
#endif
static inline void* virt_to_phys(vbx_cnn_t* vbx_cnn,void* virt){
	return (char*)(virt) + vbx_cnn->dma_phys_trans_offset;
}
#ifdef DRAW_SCALER
static void* mmap_buffer(uint32_t start_address){
    int fd = open("/dev/mem", O_RDWR);
    size_t size = 128*1024*1024;
    void* _ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0x3000000000 + start_address);
    close(fd);
    return _ptr;
}


void copy_planer_image(uint32_t* alpha_frame,uint8_t *planer_image,int x_offset,int y_offset,int width,int height){
    int plane_size = width*height;
								
	for(int y=0;y<height;++y){
        for(int x=0;x<width;++x){
			uint32_t pixel = (0x00 | (planer_image[(2*plane_size)+(width*y+x)]<<16) |
									 (planer_image[(1*plane_size)+(width*y+x)]<<8) |
									 (planer_image[(0*plane_size)+(width*y+x)]<<0));			


			alpha_frame[(y+y_offset)*(0x2000/4)+x+x_offset]=pixel;
        }
    }
}

uint32_t* scaler_draw_frame;
uint32_t* scaler_frame_addr[8];

static inline uint32_t* virt_loop() {
  uint32_t offset = (uint32_t)overlay_draw_frame - 0x70000000;
  return (uint32_t*)((char*)(scaler_draw_frame) + offset);
}
#endif

#if VBX_SOC_DRIVER
#include <sys/time.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include "pdma/pdma_helpers.h"


extern uint32_t *linux_draw_frame;
extern uint32_t* overlay_draw_frame;

static inline uint32_t* virt_loop_linux(uint32_t* loop) {
	uint32_t offset = ((uint32_t)(uintptr_t)overlay_draw_frame) - 0x70000000;
	return (uint32_t*)((char*)(linux_draw_frame) + offset);
}

static int gettimediff_us_2(struct timeval start, struct timeval end) {
	int sec = end.tv_sec - start.tv_sec;
	int usec = end.tv_usec - start.tv_usec;
	return sec * 1000000 + usec;
}

extern int8_t* pdma_mmap_t;
extern uint64_t pdma_out;
extern int32_t pdma_channel;

int32_t pdma_ch_transfer(uint64_t output_data_phys, void* source_buffer,int offset,int size,vbx_cnn_t *vbx_cnn,int32_t channel){
	uint64_t srcbuf=0x3000000000 + (uint64_t)(uintptr_t)virt_to_phys(vbx_cnn, source_buffer);
	return pdma_ch_cpy(output_data_phys + offset, srcbuf, size, channel);
}
#else
	
	#include "../tinyprintf.h"
	
#endif


#define BOXES_LEN 1024
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
static fix16_t fix16_two = F16(2);
// Globals Specification


fix16_box boxes[BOXES_LEN];
int valid_boxes = 0;
int m_run_fps=0;;


///////////////////////////////////
short detectionDemoInit(vbx_cnn_t* the_vbx_cnn, struct model_descr_t* models, uint8_t modelIdx) {
	struct model_descr_t *object_model = models+modelIdx;
	object_model->buf_idx=0;
	object_model->is_running = 0;
	object_model->model_io_buffers  = vbx_allocate_dma_buffer(the_vbx_cnn, (1+model_get_num_outputs(object_model->model))*sizeof(object_model->model_io_buffers), 0);
	if(!object_model->model_io_buffers){
		printf("Memory allocation issue for model io buffers.\n");
		return -1;
	}

	size_t input_length = 0;
#ifdef HLS_RESIZE
	input_length = model_get_input_length(object_model->model, 0) *
		((int)model_get_input_datatype(object_model->model, 0) + 1);
	object_model->pipelined_input_buffer[0] = vbx_allocate_dma_buffer(the_vbx_cnn, input_length, 0);
	object_model->pipelined_input_buffer[1] = vbx_allocate_dma_buffer(the_vbx_cnn, input_length, 0);
	object_model->model_io_buffers[0] =(uintptr_t)object_model->pipelined_input_buffer[0];
	if(!object_model->pipelined_input_buffer[0] ||!object_model->pipelined_input_buffer[1]){
			printf("Memory allocation issue for model input buffers.\n");
			return -1;	
	}
#else	
	input_length = model_get_input_length(object_model->model, 0) *
		((int)model_get_input_datatype(object_model->model, 0) + 1);
	object_model->model_input_buffer = vbx_allocate_dma_buffer(the_vbx_cnn, input_length, 0);
	object_model->model_io_buffers[0] = (uintptr_t)object_model->model_input_buffer;
	if(!object_model->model_input_buffer){
		printf("Memory allocation issue for model input buffers.\n");
		return -1;	
	}
#endif

	// Determine the input and output lengths of the Object Model, and re-order output buffers if applicable
	int num_outputs = model_get_num_outputs(object_model->model);
	for(int output = 0; output < num_outputs; output++) {
		int output_length = model_get_output_length(object_model->model, output) * 4; // fix16 outputs are 4 bytes per element?
		object_model->pipelined_output_buffers[0][output] = vbx_allocate_dma_buffer(the_vbx_cnn, output_length, 0);
		object_model->pipelined_output_buffers[1][output] = vbx_allocate_dma_buffer(the_vbx_cnn, output_length, 0);
		object_model->model_io_buffers[output+1] = (uintptr_t)object_model->pipelined_output_buffers[0][output];
		if(!object_model->pipelined_output_buffers[0][output] || !object_model->pipelined_output_buffers[1][output]){
			printf("Memory allocation issue for model output buffers.\n");
			return -1;	
		}
	}


#ifdef DRAW_SCALER
	scaler_draw_frame       = (uint32_t *)mmap_buffer(0x70000000);
	scaler_frame_addr[0]    = (uint32_t *)mmap_buffer(0x68000000);
	scaler_frame_addr[1]	= (uint32_t *)mmap_buffer(0x69000000);
	scaler_frame_addr[2]	= (uint32_t *)mmap_buffer(0x6a000000);
	scaler_frame_addr[3]	= (uint32_t *)mmap_buffer(0x6b000000);
	scaler_frame_addr[4]	= (uint32_t *)mmap_buffer(0x6c000000);
	scaler_frame_addr[5]	= (uint32_t *)mmap_buffer(0x6d000000);
	scaler_frame_addr[6]	= (uint32_t *)mmap_buffer(0x6e000000);
	scaler_frame_addr[7]	= (uint32_t *)mmap_buffer(0x6f000000);
#endif
	return 1;
}

int runDetectionDemo(struct model_descr_t* models, vbx_cnn_t* the_vbx_cnn, uint8_t modelIdx){
	int err, status;
	err = 0;
	status = 0;
	struct timeval m_run1, m_run2;
	struct model_descr_t *object_model = models+modelIdx;
	int* input_dims = model_get_input_shape(object_model->model, 0);	
	uint32_t offset;
	//Start processing the network if not already running - 1st pass only (frame 0 )
	if(!object_model->is_running) {		
#ifdef HLS_RESIZE	
	resize_image_hls(SCALER_BASE_ADDRESS,(uint32_t*)(intptr_t)(*PROCESSING_FRAME_ADDRESS),
				set_screen_width, set_screen_height, set_screen_stride, 
				set_screen_x_offset, set_screen_y_offset,
				(uint8_t*)virt_to_phys(the_vbx_cnn, (void*)object_model->pipelined_input_buffer[object_model->buf_idx]),
				input_dims[3],input_dims[2]);
#else
		*BLUE_DDR_FRAME_START_ADDR  = (2*input_dims[3]*input_dims[2]);
		*GREEN_DDR_FRAME_START_ADDR = (1*input_dims[3]*input_dims[2]);
		*RED_DDR_FRAME_START_ADDR   = (0*input_dims[3]*input_dims[2]); 
		offset = (*PROCESSING_FRAME_ADDRESS) - 0x70000000;
		object_model->model_input_buffer = (uint8_t*)(uintptr_t)(SCALER_FRAME_ADDRESS + offset);
#ifdef DRAW_SCALER
		uint32_t *scaler_frame = virt_loop();
		if ( offset == 0) {
		  copy_planer_image(scaler_frame, (uint8_t*)scaler_frame_addr[0], 100, 100, input_dims[3],input_dims[2]);
		} else if (offset == 0x1000000) {
		  copy_planer_image(scaler_frame, (uint8_t*)scaler_frame_addr[1], 100, 100, input_dims[3],input_dims[2]);
		} else if (offset == 0x2000000) {
		  copy_planer_image(scaler_frame, (uint8_t*)scaler_frame_addr[2], 100, 100, input_dims[3],input_dims[2]);
		} else if (offset == 0x3000000) {
		  copy_planer_image(scaler_frame, (uint8_t*)scaler_frame_addr[3], 100, 100, input_dims[3],input_dims[2]);
		} else if (offset == 0x4000000) {
		  copy_planer_image(scaler_frame, (uint8_t*)scaler_frame_addr[4], 100, 100, input_dims[3],input_dims[2]);
		} else if (offset == 0x5000000) {
		  copy_planer_image(scaler_frame, (uint8_t*)scaler_frame_addr[5], 100, 100, input_dims[3],input_dims[2]);
		} else if (offset == 0x6000000) {
		  copy_planer_image(scaler_frame, (uint8_t*)scaler_frame_addr[6], 100, 100, input_dims[3],input_dims[2]);
		} else if (offset == 0x7000000) {
		  copy_planer_image(scaler_frame, (uint8_t*)scaler_frame_addr[7], 100, 100, input_dims[3],input_dims[2]);
		}
#endif
		object_model->model_io_buffers[0] = (uintptr_t)object_model->model_input_buffer - the_vbx_cnn->dma_phys_trans_offset;	
//		object_model->model_io_buffers[0] = (uintptr_t)model_get_test_input(object_model->model,0);
#endif		
		// Start model
#if VBX_SOC_DRIVER		
		gettimeofday(&m_run1, NULL);
#endif		
		err = vbx_cnn_model_start(the_vbx_cnn, object_model->model, object_model->model_io_buffers); 
#ifdef HLS_RESIZE
		resize_image_hls_start(SCALER_BASE_ADDRESS,(uint32_t*)(intptr_t)(*PROCESSING_NEXT_FRAME_ADDRESS),
                set_screen_width, set_screen_height, set_screen_stride,
                set_screen_x_offset, set_screen_y_offset,
                (uint8_t*)virt_to_phys(the_vbx_cnn, (void*)object_model->pipelined_input_buffer[!object_model->buf_idx]),
                input_dims[3], input_dims[2]);
#endif
		if(err != 0) return err;
		object_model->is_running = 1;
	}

	status = vbx_cnn_model_wfi(the_vbx_cnn); // Check if model done
	
	if(status < 0) {
		return status;
	} else if (status == 0) { // When  model is completed

		//Swap set of pipelined output buffers
#ifdef HLS_RESIZE
		object_model->model_io_buffers[0] = (uintptr_t)object_model->pipelined_input_buffer[!object_model->buf_idx];

        //wait for next frame scaling to finish, if it hasn't already
		resize_image_hls_wait(SCALER_BASE_ADDRESS);
#else		
		offset = (*PROCESSING_NEXT_FRAME_ADDRESS) - 0x70000000;
		object_model->model_input_buffer = (uint8_t*)(uintptr_t)(SCALER_FRAME_ADDRESS + offset);
		object_model->model_io_buffers[0] = (uintptr_t)object_model->model_input_buffer - the_vbx_cnn->dma_phys_trans_offset;	
#endif
		int num_outputs = model_get_num_outputs(object_model->model);
		for (int o = 0; o < num_outputs; o++) {			
			object_model->model_io_buffers[o+1] = (uintptr_t)object_model->pipelined_output_buffers[!object_model->buf_idx][o];
		}	
		//Start model inference
		
		err = vbx_cnn_model_start(the_vbx_cnn, object_model->model, object_model->model_io_buffers); 
		if(err != 0) return err;
		object_model->is_running = 1;
#ifdef HLS_RESIZE
		resize_image_hls_start(SCALER_BASE_ADDRESS,(uint32_t*)(intptr_t)(*PROCESSING_NEXT_FRAME_ADDRESS),
				set_screen_width, set_screen_height, set_screen_stride, 
				set_screen_x_offset, set_screen_y_offset,
				(uint8_t*)virt_to_phys(the_vbx_cnn, (void*)object_model->pipelined_input_buffer[object_model->buf_idx]),
				input_dims[3],input_dims[2]);
#endif

    if(!strcmp(object_model->post_process_type, "PIXEL")){
            pixel_draw(object_model->model, (vbx_cnn_io_ptr_t*)object_model->pipelined_output_buffers[object_model->buf_idx],the_vbx_cnn);
    }
#if VBX_SOC_DRIVER
		gettimeofday(&m_run2, NULL);
		m_run_fps = 1000/ (gettimediff_us_2(m_run1, m_run2) / 1000);
	if (PDMA){
		vbx_cnn_io_ptr_t pdma_buffer[model_get_num_outputs(object_model->model)];
		int output_offset=0;
		for(int o =0; o<(int)model_get_num_outputs(object_model->model);o++){
			int output_length = model_get_output_length(object_model->model, o);
			pdma_ch_transfer(pdma_out,(void*)object_model->pipelined_output_buffers[object_model->buf_idx][o],output_offset,model_get_output_length(object_model->model, o),the_vbx_cnn,pdma_channel);
			pdma_buffer[o] = (vbx_cnn_io_ptr_t)(pdma_mmap_t + output_offset);
			output_offset+= output_length;
		}
		pprint_post_process(object_model->name, object_model->post_process_type, object_model->model, (fix16_t**)(uintptr_t)pdma_buffer,1,fps);
	}
#else
		pprint_post_process(object_model->name, object_model->post_process_type, object_model->model, (fix16_t**)(uintptr_t)object_model->pipelined_output_buffers[object_model->buf_idx],1,fps);

#endif
	object_model->buf_idx = !object_model->buf_idx;
		
	}
	return status;
}
